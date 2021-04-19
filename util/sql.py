"""
Functions that interface with the dlr "liniendatenbank" database
"""

from typing import Any, Tuple, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from pandas import DataFrame
from sqlalchemy import text

from .osm import sql_get_osm_from_line


# OO interface
class RailwayDatabase:
    # TODO add caching by saving already fetched data for each trip id

    def __init__(self, engine):
        self.engine = engine

    def get_trip_shape(self, trip_id: int, crs: Optional[Any] = None) -> GeoDataFrame:
        """
        Get shape id of trip

            Parameters
            ----------
                trip_id : int
                    The id of the trip
                crs
                    if given the geometry is converted to this crs

            Returns
            -------
                GeoDataFrame
                    A GeoDataFrame with 1 row and  columns "shape_id" and "geom"
        """

        sql = """
            SELECT geo_shape_geoms.shape_id , geo_shape_geoms.geom
            FROM geo_trips, geo_shape_geoms
            WHERE geo_trips.shape_id = geo_shape_geoms.shape_id
            AND geo_trips.trip_id = :trip_id
            """
        shape = gpd.read_postgis(text(sql), con=self.engine, params={"trip_id": int(trip_id)}, geom_col='geom')

        if crs:
            shape = shape.to_crs(crs)

        return shape

    def get_trip_timetable(self, trip_id: int, min_stop_duration: float = 30.,
                           round_int: bool = True) -> DataFrame:
        # distance from start
        # station name
        # stop time at station in s
        # driving time to next station in s

        # sqlalchemy cant handle numpy datatypes
        trip_id = int(trip_id)

        sql = """\
        select geo_trips.trip_headsign, geo_stop_times.stop_sequence,\
        geo_stop_times.arrival_time, geo_stop_times.departure_time,\
        geo_stops.stop_name, ST_LineLocatePoint(ST_Transform(geo_shape_geoms.geom,25832),\
        ST_Transform(ST_SetSRID(ST_MakePoint(stop_lon,stop_lat),4326),25832))\
         * ST_length(ST_Transform(geo_shape_geoms.geom,25832)) as dist
        from geo_stop_times, geo_stops, geo_trips, geo_shape_geoms
        where 
        geo_stops.stop_id = geo_stop_times.stop_id
        and geo_stop_times.trip_id = geo_trips.trip_id
        and geo_trips.shape_id = geo_shape_geoms.shape_id
        and geo_trips.trip_id = :trip_id
        order by stop_sequence, departure_time
        """

        time_table = pd.read_sql_query(text(sql), con=self.engine, params={"trip_id": trip_id})

        s15 = pd.Timedelta(min_stop_duration * 0.5, unit="s")

        last_stop = time_table.stop_sequence.iloc[-1]
        first_stop = time_table.stop_sequence.iloc[0]

        # ignore first and last station
        ign_frst_last = (time_table.stop_sequence > first_stop) & (time_table.stop_sequence < last_stop)
        arr_eq_dep = ign_frst_last & (time_table.departure_time == time_table.arrival_time)

        # Assumption: trains stop at least 30s
        # if arrival time = departure time, then arrival time -15 and departure time + 15
        time_table.loc[arr_eq_dep, ["arrival_time"]] -= s15
        time_table.loc[arr_eq_dep, ["departure_time"]] += s15

        # stop duration = departure - arrival
        time_table["stop_duration"] = (time_table.departure_time - time_table.arrival_time).dt.total_seconds()

        # driving time to next station:
        # take arrival time of next station and subtract departure time
        driving_time = time_table.arrival_time[1:].dt.total_seconds().values - time_table.departure_time[
                                                                               :-1].dt.total_seconds().values

        driving_time = np.append(driving_time, 0)

        time_table["driving_time"] = driving_time

        # delete rows where driving time to next station is 0 (except last row)
        keep = time_table["driving_time"].values != 0
        keep[-1] = True
        time_table = time_table[keep]

        if round_int:
            time_table["dist"] = np.rint(time_table["dist"]).astype(int)
            time_table["stop_duration"] = np.rint(time_table["stop_duration"]).astype(int)
            time_table["driving_time"] = np.rint(time_table["driving_time"]).astype(int)

        return time_table[["dist", "stop_name", "stop_duration", "driving_time", "arrival_time", "departure_time"]]

    def get_trip_stops(self, trip_id: int) -> Union[DataFrame, Tuple[str, DataFrame]]:
        sql = """
        select
        stop_name, stop_lat, stop_lon
        from geo_trips, geo_stop_times, geo_stops
        where
        geo_trips.trip_id = geo_stop_times.trip_id
        and geo_stops.stop_id = geo_stop_times.stop_id
        and geo_trips.trip_id = :trip_id
        order by stop_sequence;
        """

        stops = pd.read_sql_query(text(sql), con=self.engine, params={"trip_id": int(trip_id)})

        # stops = gpd.read_postgis(text(sql), geom_col='geom', con=engine, params={"trip_id": trip_id})

        return stops

    def get_trip_osm(self, trip_id: int, **kwargs):

        # get shape from database
        shape: GeoDataFrame = self.get_trip_shape(trip_id)

        trip_geom = shape["geom"]
        osm_data = sql_get_osm_from_line(trip_geom, self.engine, **kwargs)

        return osm_data
