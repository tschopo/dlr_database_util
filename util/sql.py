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

        with self.engine.connect() as connection:
            shape = gpd.read_postgis(text(sql), con=connection, params={"trip_id": int(trip_id)}, geom_col='geom')

        if crs:
            shape = shape.to_crs(crs)

        return shape

    def get_trip_timetable(self, trip_id: int, min_stop_duration: float = 30.,
                           round_int: bool = True, filter=True) -> DataFrame:
        # distance from start
        # station name
        # stop time at station in s
        # driving time to next station in s

        # sqlalchemy cant handle numpy datatypes
        trip_id = int(trip_id)

        # old dist calculation fails because of kopfbahnhöfe and inaccurate stop lat lon
        # ST_LineLocatePoint(ST_Transform(geo_shape_geoms.geom, 25832), \
        #                   ST_Transform(ST_SetSRID(ST_MakePoint(stop_lon, stop_lat), 4326), 25832)) \
        # * ST_length(ST_Transform(geo_shape_geoms.geom, 25832))

        sql = """\
        select geo_trips.trip_headsign, geo_stop_times.stop_sequence,\
        geo_stop_times.arrival_time, geo_stop_times.departure_time,\
        geo_stops.stop_name, shape_dist_traveled as dist
        from geo_stop_times, geo_stops, geo_trips, geo_shape_geoms
        where 
        geo_stops.stop_id = geo_stop_times.stop_id
        and geo_stop_times.trip_id = geo_trips.trip_id
        and geo_trips.shape_id = geo_shape_geoms.shape_id
        and geo_trips.trip_id = :trip_id
        order by stop_sequence, departure_time
        """

        with self.engine.connect() as connection:
            timetable = pd.read_sql_query(text(sql), con=connection, params={"trip_id": trip_id})

        if filter:
            # for '1957658_Wernigerode - Ilfeld'
            # filter stations that end with "001 P1", "002 P2"
            # sometimes in the gtfs the geometries are added as stations
            timetable = timetable[~timetable.stop_name.str.contains(r'\d\d\d P\d')]

        s15 = pd.Timedelta(min_stop_duration * 0.5, unit="s")

        last_stop = timetable.stop_sequence.iloc[-1]
        first_stop = timetable.stop_sequence.iloc[0]

        # ignore first and last station
        ign_frst_last = (timetable.stop_sequence > first_stop) & (timetable.stop_sequence < last_stop)
        arr_eq_dep = ign_frst_last & (timetable.departure_time == timetable.arrival_time)

        # Assumption: trains stop at least 30s
        # if arrival time = departure time, then arrival time -15 and departure time + 15
        timetable.loc[arr_eq_dep, ["arrival_time"]] -= s15
        timetable.loc[arr_eq_dep, ["departure_time"]] += s15

        # stop duration = departure - arrival
        timetable["stop_duration"] = (timetable.departure_time - timetable.arrival_time).dt.total_seconds()

        # driving time to next station:
        # take arrival time of next station and subtract departure time
        driving_time = timetable.arrival_time[1:].dt.total_seconds().values - timetable.departure_time[
                                                                               :-1].dt.total_seconds().values

        driving_time = np.append(driving_time, 0)

        timetable["driving_time"] = driving_time

        # delete rows where driving time to next station is 0 (except last row)
        keep = timetable["driving_time"].values != 0
        keep[-1] = True
        timetable = timetable[keep]

        # reset index
        timetable.reset_index(drop=True, inplace=True)

        if round_int:
            timetable["dist"] = np.rint(timetable["dist"]).astype(int)
            timetable["stop_duration"] = np.rint(timetable["stop_duration"]).astype(int)
            timetable["driving_time"] = np.rint(timetable["driving_time"]).astype(int)

        return timetable[["dist", "stop_name", "stop_duration", "driving_time", "arrival_time", "departure_time"]]

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

        with self.engine.connect() as connection:
            stops = pd.read_sql_query(text(sql), con=connection, params={"trip_id": int(trip_id)})

        # stops = gpd.read_postgis(text(sql), geom_col='geom', con=engine, params={"trip_id": trip_id})

        return stops

    def get_trip_osm(self, trip_id: int, **kwargs):

        # get shape from database
        shape: GeoDataFrame = self.get_trip_shape(trip_id)

        trip_geom = shape["geom"]
        osm_data = sql_get_osm_from_line(trip_geom, self.engine, **kwargs)

        return osm_data
