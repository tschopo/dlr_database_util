"""
Functions that interface with the dlr "liniendatenbank" database
"""
import re
from typing import Any, Tuple, Optional, Union, List

import geopandas as gpd
import numpy as np
import pandas as pd
from ElevationSampler import ElevationProfile
from geopandas import GeoDataFrame
from pandas import DataFrame
from sqlalchemy import text
from datetime import datetime, timedelta

from .tpt import elevation_pipeline
from .osm import sql_get_osm_from_line, get_osm_prop


# OO interface
class RailwayDatabase:
    # TODO add caching by saving already fetched data for each trip id

    def __init__(self, engine):
        self.engine = engine

        # TODO run, and also the other extensions used by queries
        """
        CREATE
        EXTENSION if not exists
        pg_trgm
        """

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

    def get_trips_from_station(self, station_name: str, return_id_type='same_geom_and_stops_candidate', fuzzy=True, fuzzy_strength=0.5) -> np.ndarray:
        """
        Returns all trips that start at station "station_name". Performs fuzzy search if fuzzy = True.

        Parameters
        ----------
        station_name
        fuzzy

        Returns
        -------

        """

        if not fuzzy:
            fuzzy_strength = 1.

        if return_id_type == 'all':
            return_id_type = 'trip_id'
        elif return_id_type != 'same_geom_and_stops_candidate' and return_id_type != 'same_geom_candidate':
            raise Exception('return_id_type must be "all", "same_geom_and_stops_candidate", or "same_geom_candidate"')

        sql = """select distinct ldb_trip_candidates.{return_id_type}
            from geo_stop_times, geo_stops, ldb_trip_candidates
            where ldb_trip_candidates.trip_id = geo_stop_times.trip_id
            and geo_stop_times.stop_id = geo_stops.stop_id
            and stop_sequence = 0
            and SIMILARITY(stop_name,:start_station) > :fuzzy_strength
        """.format(return_id_type=return_id_type)

        with self.engine.connect() as connection:
            trips = pd.read_sql_query(text(sql), con=connection, params={"start_station": station_name, "fuzzy_strength": fuzzy_strength})

        return trips.values

    def get_trips_to_station(self, station_name: str, return_id_type='same_geom_and_stops_candidate', fuzzy=True, fuzzy_strength=0.5) -> np.ndarray:
        """
        Returns all trips that end at station "station_name". Performs fuzzy search if fuzzy = True.

        Parameters
        ----------
        station_name
        fuzzy

        Returns
        -------

        """

        if not fuzzy:
            fuzzy_strength = 1.

        if return_id_type == 'all':
            return_id_type = 'trip_id'
        elif return_id_type != 'same_geom_and_stops_candidate' and return_id_type != 'same_geom_candidate':
            raise Exception('return_id_type must be "all", "same_geom_and_stops_candidate", or "same_geom_candidate"')

        sql = """select distinct ldb_trip_candidates.{return_id_type}
            from ldb_trip_candidates, geo_stop_times, geo_stops, calc_n_stops
            where ldb_trip_candidates.trip_id = geo_stop_times.trip_id
            and geo_stop_times.stop_id = geo_stops.stop_id
            and calc_n_stops.trip_id = ldb_trip_candidates.trip_id
            and stop_sequence = n_stops
            and SIMILARITY(stop_name,:end_station) > :fuzzy_strength;
        """.format(return_id_type=return_id_type)

        with self.engine.connect() as connection:
            trips = pd.read_sql_query(text(sql), con=connection,
                                      params={"end_station": station_name, "fuzzy_strength": fuzzy_strength})

        return trips.values

    def get_trips_from_to(self, from_station: Optional[str] = None, to_station: Optional[str] = None, return_id_type='same_geom_and_stops_candidate', fuzzy=True, fuzzy_strength=0.5, include_return_trips=False):
        """
        get trips that start at from_station and end at to_station.

        Parameters
        ----------
        from_station
        to_station
        return_id_type
            either 'all', 'same_geom_candidate' or 'same_geom_and_stops_candidate'
        fuzzy
        fuzzy_strength

        Returns
        -------

        """

        if from_station is None and to_station is None:
            raise Exception("Either start_station or end_station must be given!")

        if include_return_trips:
            from_station_return, to_station_return = to_station, from_station
            return_trips = self.get_trips_from_to(from_station_return, to_station_return, return_id_type, fuzzy, fuzzy_strength, include_return_trips=False)

        else:
            return_trips = np.array([], dtype=int)

        if from_station is not None:
            start_trips = self.get_trips_from_station(from_station, return_id_type=return_id_type, fuzzy=fuzzy, fuzzy_strength=fuzzy_strength)
            if to_station is None:
                return np.append(start_trips.flatten(), return_trips)

        if to_station is not None:
            end_trips = self.get_trips_to_station(to_station, return_id_type=return_id_type, fuzzy=fuzzy, fuzzy_strength=fuzzy_strength)
            if from_station is None:
                return np.append(end_trips.flatten(), return_trips)

        return np.append(np.intersect1d(start_trips, end_trips, assume_unique=True), return_trips)

    def get_same_geom_and_stops_candidate(self, trip_id):

        sql = """select
            same_geom_and_stops_candidate
            from ldb_trip_candidates where
            trip_id = :trip_id
        """

        with self.engine.connect() as con:
            rs = con.execute(text(sql), {'trip_id': int(trip_id)}).first()
            trip_id = rs[0]

            return trip_id

    def get_same_geom_candidate(self, trip_id):

        sql = """select
            same_geom_candidate
            from ldb_trip_candidates where
            trip_id = :trip_id
        """

        with self.engine.connect() as con:
            rs = con.execute(text(sql), {'trip_id': int(trip_id)}).first()
            trip_id = rs[0]

            return trip_id

    def get_trips_from_same_geom_and_stops_candidate(self, same_geom_and_stops_candidate):
        """
        Get list of trips that belong to the given same_geom_and_stops_candidate.

        Parameters
        ----------
        same_geom_and_stops_candidate

        Returns
        -------

        """

        sql = """select distinct trip_id
                    from ldb_trip_candidates
                    where same_geom_and_stops_candidate = :same_geom_and_stops_candidate
                    order by trip_id;
                """

        with self.engine.connect() as connection:
            trips = pd.read_sql_query(text(sql), con=connection,
                                      params={"same_geom_and_stops_candidate": same_geom_and_stops_candidate})
        return trips

    def get_trips_from_same_geom_candidate(self, same_geom_candidate):

        sql = """select distinct trip_id
                    from ldb_trip_candidates
                    where same_geom_and_stops_candidate = :same_geom_candidate
                    order by trip_id;
                """

        with self.engine.connect() as connection:
            trips = pd.read_sql_query(text(sql), con=connection,
                                      params={"same_geom_candidate": same_geom_candidate})
        return trips

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
        """
        Returns Dataframe of Stopname, stop_lat, stop_lon of the stops of a trip.

        Parameters
        ----------
        trip_id : int
            The Trip ID

        Returns
        -------
            Dataframe

        """
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

    def get_trip_osm(self, trip_id: int, crs=25832, **kwargs) -> DataFrame:
        """
        Generates a Dataframe containing OSM Data for the trip_id.

        Parameters
        ----------
        trip_id
        crs
        kwargs : dictionary
            kwargs for sql_get_osm_from_line

        Returns
        -------
            Dataframe : output of sql_get_osm_from_line
        """

        # get shape from database
        shape: GeoDataFrame = self.get_trip_shape(trip_id, crs=crs)

        trip_geom = shape["geom"]
        osm_data = sql_get_osm_from_line(trip_geom, self.engine, **kwargs)

        return osm_data

    def get_trip_brunnels(self, trip_id: int):

        sql = """
            select ldb_brunnels.start_dist, ldb_brunnels.end_dist
            from ldb_trip_candidates, ldb_brunnels
            where ldb_brunnels.trip_id = ldb_trip_candidates.same_geom_candidate
            and ldb_trip_candidates.trip_id = :trip_id
            order by start_dist;
            """

        with self.engine.connect() as connection:
            brunnels = pd.read_sql_query(text(sql), con=connection, params={"trip_id": int(trip_id)})

        return brunnels

    def get_trip_elevation_profile(self, trip_id: int):

        sql = """
            select ldb_elevation.dist, ldb_elevation.elevation
            from ldb_trip_candidates, ldb_elevation
            where ldb_elevation.trip_id = ldb_trip_candidates.same_geom_candidate
            and ldb_trip_candidates.trip_id = :trip_id
            order by dist;
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
            and ldb_trip_candidates.trip_id = :trip_id
            order by start_dist;
            """

        with self.engine.connect() as connection:
            electrified = pd.read_sql_query(text(sql), con=connection, params={"trip_id": int(trip_id)})

        return electrified

    def get_trip_maxspeed(self, trip_id: int):
        sql = """
            select ldb_maxspeed.maxspeed, ldb_maxspeed.start_dist, ldb_maxspeed.end_dist
            from ldb_trip_candidates, ldb_maxspeed
            where ldb_maxspeed.trip_id = ldb_trip_candidates.same_geom_candidate
            and ldb_trip_candidates.trip_id = :trip_id
            order by start_dist;
            """

        with self.engine.connect() as connection:
            maxspeed = pd.read_sql_query(text(sql), con=connection, params={"trip_id": int(trip_id)})

        return maxspeed

    def save_trip_osm_tables(self, trip_id: int, crs=25832, replace=True, trip_id_is_candidate_trip_id=False,
                             max_trip_length: Optional[float] = 500000., brunnel_filter_length=10.,
                             set_unknown_electrified_to_no=True, **get_osm_kwargs):
        """
        calculates the osm properties for the trip and saves them into the database.

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
        set_unknown_electrified_to_no : bool
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

        if max_trip_length is not None and trip_length > max_trip_length:
            return trip_id

        electrified = get_osm_prop(osm_data, "electrified", trip_length=trip_length,
                                   set_unknown_electrified_to_no=set_unknown_electrified_to_no)
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

    def get_trip(self, trip_id: int, crs=25832):
        """
        Get a Trip from the Database. Use TripGenerator if trip data is not yet in the Database.

        Parameters
        ----------
        trip_id : int
        crs : int or crs Object
            default epsg 25832

        Returns
        -------

        Trip : A Trip object if all the data for the trip is already generated. Raises Exception if nnot yet generated
        """
        from .trip import Trip

        trip_id = int(trip_id)

        trip_geom = self.get_trip_shape(trip_id, crs)["geom"]

        # get timetable data from database
        timetable = self.get_trip_timetable(trip_id)

        # convert the timetable arrival and start time to datetime instead of timedelta
        date = "2021-01-01"
        timetable["arrival_time"] = np.datetime64(date) + timetable.arrival_time
        timetable["departure_time"] = np.datetime64(date) + timetable.departure_time

        # check if generated trip is already stored in database
        if self.contains_generated_trip(trip_id):
            # if so get the data from database
            brunnels = self.get_trip_brunnels(trip_id)
            electrified = self.get_trip_electrified(trip_id)
            maxspeed = self.get_trip_maxspeed(trip_id)
            elevation_profile = self.get_trip_elevation_profile(trip_id)
            return Trip(trip_id, electrified, maxspeed, brunnels, timetable, elevation_profile, trip_geom)
        else:
            raise Exception('Generated Trip not in Database! Use TripGenerator to generate the Trip.')

    def fix_osm(self, fix_geojson: str, prop: str, crs=25832):
        """
        Fix osm data (in osm_railways) from geojson. The geojson contains the correct data. All osm features that
        intersect the geojson are set to the prop in the geojson.

        Parameters
        ----------
        fix_geojson: str
            path to the geojson file
        prop: str
            the key in the geojson / osm database that is fixed

        Returns
        -------

        """
        fix_osm = gpd.read_file(fix_geojson)
        fix_osm = fix_osm.to_crs(crs)

        # escape prop for sql
        prop = re.sub('[^A-Za-z0-9_]+', '', prop)

        sql = """ UPDATE osm_railways
        set {prop} = :prop_val
        where way_id = :way_id
        """.format(prop=prop)

        for fix_index, fix_row in fix_osm.iterrows():
            geom = fix_row.geometry
            osm_data = sql_get_osm_from_line(geom, self.engine, crs=crs)

            with self.engine.connect() as con:
                for osm_index, osm_row in osm_data.iterrows():
                    print("..." + str(osm_index))
                    print("set ", str(osm_row.way_id), " to ", fix_row[prop])
                    con.execute(text(sql), {'way_id': osm_row.way_id, 'prop_val': fix_row[prop]})

    def calculate_n_fahrten_in_period(self, trip_id: int, period_start: int) -> int:
        """
        Calculates the number of times a trip drives in the 1 week period from period_start.
        Period must start on monday.

        Parameters
        ----------
        trip_id
        period_start : int
            Format YYYYMMDD e.g. 20201109

        Returns
        -------

        Number of times a trip drives.
        """

        weekdays = {0: "monday",
                    1: "tuesday",
                    2: "wednesday",
                    3: "thursday",
                    4: "friday",
                    5: "saturday",
                    6: "sunday"}

        end_date = datetime.strptime(str(period_start), '%Y%m%d') + timedelta(days=6)
        period_end = int(end_date.strftime('%Y%m%d'))
        end_weekday_index = end_date.weekday()
        end_weekday = weekdays[end_weekday_index]

        start_date = datetime.strptime(str(period_start), '%Y%m%d')
        start_weekday_index = start_date.weekday()
        start_weekday = weekdays[start_weekday_index]

        if start_weekday != "monday":
            raise Exception("Period must start in monday!")

        sql = """select * from geo_calendar, geo_trips
            where geo_trips.service_id = geo_calendar.service_id
            and geo_trips.trip_id = :trip_id
            and start_date <= :period_end
            and end_date >= :period_start;
            """

        n_fahrten = 0

        with self.engine.connect() as conn:
            calendar = pd.read_sql_query(text(sql), con=conn, params={'trip_id': trip_id, 'period_start': period_start,
                                                                      'period_end': period_end})

        covered_dates = []
        if calendar.shape[0] > 0:

            assert (calendar.shape[0] == 1)

            calendar = calendar.iloc[0]

            # case 2: trip starts later in the week
            if calendar.start_date > period_start:
                # get day of week of start_date
                start_weekday_index = datetime.strptime(str(int(calendar.start_date)), '%Y%m%d').weekday()
                start_weekday = weekdays[start_weekday_index]

            # case 3. trip ends later in the week
            if calendar.end_date < period_end:
                # get day of week of start_date
                end_weekday_index = datetime.strptime(str(int(calendar.end_date)), '%Y%m%d').weekday()
                end_weekday = weekdays[end_weekday_index]

            # get the dates where the trip is driving
            if calendar.monday and start_weekday_index == 0:
                covered_dates.append(start_date)
            if calendar.tuesday and start_weekday_index <= 1 and end_weekday_index >= 1:
                covered_dates.append(start_date + timedelta(days=1))
            if calendar.wednesday and start_weekday_index <= 2 and end_weekday_index >= 2:
                covered_dates.append(start_date + timedelta(days=2))
            if calendar.thursday and start_weekday_index <= 3 and end_weekday_index >= 3:
                covered_dates.append(start_date + timedelta(days=3))
            if calendar.friday and start_weekday_index <= 4 and end_weekday_index >= 4:
                covered_dates.append(start_date + timedelta(days=4))
            if calendar.saturday and start_weekday_index <= 5 and end_weekday_index >= 5:
                covered_dates.append(start_date + timedelta(days=5))
            if calendar.sunday and start_weekday_index <= 6 and end_weekday_index >= 6:
                covered_dates.append(start_date + timedelta(days=6))

            n_fahrten = np.sum(calendar[start_weekday:end_weekday])

        # get excetptions
        sql = """select * from geo_calendar_dates, geo_trips
            where geo_trips.service_id = geo_calendar_dates.service_id
            and geo_trips.trip_id = :trip_id
            and date <= :period_end
            and date >= :period_start;
            """

        with self.engine.connect() as conn:
            calendar_dates = pd.read_sql_query(text(sql), con=conn,
                                               params={'trip_id': trip_id, 'period_start': period_start,
                                                       'period_end': period_end})

        # assert that gtfs correct and only trip added if not drive on that date and only trip removed if drive on that
        # date
        for index, row in calendar_dates[calendar_dates.exception_type == 2].iterrows():
            if not (datetime.strptime(str(int(row.date)), '%Y%m%d') in covered_dates):
                raise Exception("Error in GTFS data, inconsistent calendar and calendar dates")

        for index, row in calendar_dates[calendar_dates.exception_type == 1].iterrows():
            if not (datetime.strptime(str(int(row.date)), '%Y%m%d') not in covered_dates):
                raise Exception("Error in GTFS data, inconsistent calendar and calendar dates")

        # all exceptions where trip removed --> decrement n_fahrten
        n_fahrten -= np.sum(calendar_dates.exception_type == 2)

        # all exceptions where trip added --> increment n_fahrten
        n_fahrten += np.sum(calendar_dates.exception_type == 1)

        return n_fahrten
