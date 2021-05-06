"""
Functions that interface with the dlr "liniendatenbank" database
"""

from typing import Any, Tuple, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from ElevationSampler import ElevationProfile
from geopandas import GeoDataFrame
from pandas import DataFrame
from sqlalchemy import text

from .tpt import elevation_pipeline
from .osm import sql_get_osm_from_line, get_osm_prop


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

    def get_trip_brunnels(self, trip_id: int):

        sql = """
            select ldb_brunnels.start_dist, ldb_brunnels.end_dist
            from ldb_trip_candidates, ldb_brunnels
            where ldb_brunnels.trip_id = ldb_trip_candidates.same_geom_candidate
            and ldb_trip_candidates.trip_id = :trip_id;
            """

        with self.engine.connect() as connection:
            brunnels = pd.read_sql_query(text(sql), con=connection, params={"trip_id": int(trip_id)})

        return brunnels

    def get_trip_elevation_profile(self, trip_id: int):

        sql = """
            select ldb_elevation.dist, ldb_elevation.elevation
            from ldb_trip_candidates, ldb_elevation
            where ldb_elevation.trip_id = ldb_trip_candidates.same_geom_candidate
            and ldb_trip_candidates.trip_id = :trip_id;
            """

        with self.engine.connect() as connection:
            elevation = pd.read_sql_query(text(sql), con=connection, params={"trip_id": int(trip_id)})

        elevation_profile = ElevationProfile(elevation.dist, elevation.elevation)

        return elevation_profile

    def get_trip_electrified(self, trip_id: int):

        sql = """
            select ldb_electrified.*
            from ldb_trip_candidates, ldb_electrified
            where ldb_electrified.trip_id = ldb_trip_candidates.same_geom_candidate
            and ldb_trip_candidates.trip_id = :trip_id;
            """

        with self.engine.connect() as connection:
            electrified = pd.read_sql_query(text(sql), con=connection, params={"trip_id": int(trip_id)})

        return electrified

    def get_trip_maxspeed(self, trip_id: int):
        sql = """
            select ldb_maxspeed.maxspeed, ldb_maxspeed.start_dist, ldb_maxspeed.end_dist
            from ldb_trip_candidates, ldb_maxspeed
            where ldb_maxspeed.trip_id = ldb_trip_candidates.same_geom_candidate
            and ldb_trip_candidates.trip_id = :trip_id;
            """

        with self.engine.connect() as connection:
            maxspeed = pd.read_sql_query(text(sql), con=connection, params={"trip_id": int(trip_id)})

        return maxspeed

    def save_trip_osm_tables(self, trip_id: int, crs=25832, replace=True, trip_id_is_candidate_trip_id=False,
                             max_trip_length=500000, brunnel_filter_length=10., **get_osm_kwargs):
        """

        Parameters
        ----------

        trip_id
        crs : int or pyproj CRS object
            Must be a projected crs (unit meters). The osm_data and trip geom are converted to this crs. Is used only
            internally, since this function does not save geo data, but the accuracy of the distance calculations depend
            on this.
        replace : bool
            if True', then replaces the trips in the database on conflict. else an
             error is raised if trip already exists and the values are not added. default True
        trip_id_is_candidate_trip_id: bool
            if False then the candidate trip_id is looked up first. If True it is assumed that trip_id is candidate
            trip_id and no lookup is performed. candidate trips are trips with the same geometry
        max_trip_length : float
            ignore trips longer than this. returns trip_id if longer.
        brunnel_filter_length
            parameter of :func:`get_osm_prop`
        get_osm_kwargs
            parameters of :func:`sql_get_osm_from_line`

        Returns
        -------
            0 if success else trip_id

        """

        # get the candidate trip id with same geom
        if not trip_id_is_candidate_trip_id:
            sql = """
                    select same_geom_candidate 
                    from ldb_trip_candidates 
                    where trip_id = :trip_id
                    """

            with self.engine.connect() as con:
                rs = con.execute(text(sql), {'trip_id': int(trip_id)}).first()
                trip_id = rs[0]

        # get shape from database
        shape: GeoDataFrame = self.get_trip_shape(trip_id, crs=crs)

        trip_geom = shape["geom"]
        osm_data = sql_get_osm_from_line(trip_geom, self.engine, **get_osm_kwargs)

        trip_length = trip_geom.length.iloc[0]

        if trip_length > max_trip_length:
            return trip_id

        electrified = get_osm_prop(osm_data, "electrified", trip_length=trip_length)
        maxspeed = get_osm_prop(osm_data, "maxspeed", trip_length=trip_length)
        brunnels = get_osm_prop(osm_data, "brunnel", brunnel_filter_length=brunnel_filter_length)

        electrified['trip_id'] = trip_id
        maxspeed['trip_id'] = trip_id
        brunnels['trip_id'] = trip_id

        # make sure the columns are in right order
        electrified = electrified[['trip_id', 'electrified', 'start_dist', 'end_dist']]
        maxspeed = maxspeed[['trip_id', 'maxspeed', 'start_dist', 'end_dist']]
        brunnels = brunnels[['trip_id', 'start_dist', 'end_dist']]

        with self.engine.begin() as con:
            # if replace: make sure that trip does not exist by deleting all records
            if replace:
                con.execute(text("delete from ldb_electrified where trip_id = :trip_id"), {'trip_id': trip_id})
                con.execute(text("delete from ldb_maxspeed where trip_id = :trip_id"), {'trip_id': trip_id})
                con.execute(text("delete from ldb_brunnels where trip_id = :trip_id"), {'trip_id': trip_id})

            electrified.to_sql('ldb_electrified', con, if_exists='append', index=False)
            maxspeed.to_sql('ldb_maxspeed', con, if_exists='append', index=False)
            brunnels.to_sql('ldb_brunnels', con, if_exists='append', index=False)

        return 0

    def save_trip_elevation_table(self, trip_id, dem, brunnels, replace=True, trip_id_is_candidate_trip_id=False,
                                  first_sample_distance=10, interpolated=True,
                                  **ele_pipeline_kwargs):

        # get the candidate trip id with same geom
        if not trip_id_is_candidate_trip_id:
            sql = """
                select same_geom_candidate 
                from ldb_trip_candidates 
                where trip_id = :trip_id
                """

            with self.engine.connect() as con:
                rs = con.execute(text(sql), {'trip_id': trip_id}).first()
                trip_id = rs[0]

        trip_geom = self.get_trip_shape(int(trip_id), dem.crs)["geom"]

        elevation_profile = dem.elevation_profile(trip_geom, distance=first_sample_distance,
                                                  interpolated=interpolated)

        elevation_profile = elevation_pipeline(elevation_profile, brunnels, **ele_pipeline_kwargs)

        elevation = pd.DataFrame({'dist': elevation_profile.distances,
                                  'elevation': elevation_profile.elevations})
        elevation['trip_id'] = trip_id

        elevation = elevation[['trip_id', 'dist', 'elevation']]

        with self.engine.connect() as con:
            # if replace: make sure that trip does not exist by deleting all records
            if replace:
                con.execute(text("delete from ldb_elevation where trip_id = :trip_id"), {'trip_id': trip_id})

            elevation.to_sql('ldb_elevation', con, if_exists='append', index=False)

        return 0

    def contains_generated_trip(self, trip_id):
        """
        Check if the data for the trip is stored in the database. Checks if data present in ldb_electrified,
        ldb_elevation and ldb_maxspeed.

        Parameters
        ----------
        trip_id

        Returns
        -------

        """

        with self.engine.connect() as con:
            # get the candidate trip
            sql = """
               select same_geom_candidate 
               from ldb_trip_candidates 
               where trip_id = :trip_id
               """

            # if no trip candidate then also no data
            rs = con.execute(text(sql), {'trip_id': trip_id}).first()
            if rs is not None:
                trip_id = rs[0]
            else:
                return False

            sql = "select exists(select 1 from ldb_electrified where trip_id=:trip_id)"
            in_electrified = con.execute(text(sql), {'trip_id': trip_id}).first()[0]

            sql = "select exists(select 1 from ldb_maxspeed where trip_id=:trip_id)"
            in_maxspeed = con.execute(text(sql), {'trip_id': trip_id}).first()[0]

            sql = "select exists(select 1 from ldb_elevation where trip_id=:trip_id)"
            in_elevation = con.execute(text(sql), {'trip_id': trip_id}).first()[0]

            return in_electrified and in_maxspeed and in_elevation






