"""
Functions that interface with the dlr "liniendatenbank" database
"""

from typing import Any, Tuple, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from ElevationSampler import ElevationSampler
from geopandas import GeoSeries, GeoDataFrame
from pandas import DataFrame
from shapely.geometry import Point
from shapely.ops import unary_union
from sqlalchemy import text
from sqlalchemy.engine import Engine

from .osm import get_osm_prop
from .osm import sql_get_osm_from_line, filter_overlapping_osm
from .tpt import process_ele


# from util import sql_get_osm_raw_by_shape_id


def sql_get_shape_id(trip_id: int, engine: Engine) -> int:
    """
    Get shape id of trip
    
        Parameters
        ----------
            trip_id : int
                The id of the trip
            engine : sqlalchemy engine
            
        Returns
        -------
            int
                The shape_id of the trip
    """

    sql = """
    SELECT geo_shape_geoms.shape_id 
    FROM geo_trips, geo_shape_geoms
    WHERE geo_trips.shape_id = geo_shape_geoms.shape_id
    AND geo_trips.trip_id = :trip_id
    """
    shape_id = int(pd.read_sql_query(text(sql), con=engine, params={"trip_id": trip_id}).iloc[0][0])

    return shape_id


def sql_get_geometry(shape_id: int, engine: Engine) -> GeoSeries:
    """
    Get the geometry of a trip by shape_id
    
        Parameters
        ----------
            shape_id : int
                The id of the shape
            engine : sqlalchemy engine
            
        Returns
        -------
            GeoSeries
                Geopandas GeoSeries containing the geometry
    """

    sql = """
    SELECT *
    FROM geo_shape_geoms
    WHERE shape_id = :shape_id
    """
    geo_shape_geom = gpd.read_postgis(text(sql), con=engine, geom_col='geom', params={"shape_id": shape_id})
    trip_geom = geo_shape_geom.geom

    return trip_geom


def sql_get_trip_stops(trip_id: int, engine: Engine) -> DataFrame:
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

    stops = pd.read_sql_query(text(sql), con=engine, params={"trip_id": trip_id})

    # stops = gpd.read_postgis(text(sql), geom_col='geom', con=engine, params={"trip_id": trip_id})

    return stops


def sql_get_trip_geom(trip_id: int, engine: Engine, crs: Optional[Any] = None) -> GeoSeries:
    """

    Parameters
    ----------
    trip_id
    engine
    crs
        if given the geometry is converted to this geom

    Returns
    -------

    """
    shape_id = sql_get_shape_id(trip_id, engine)
    trip_geom = sql_get_geometry(shape_id, engine)

    if crs:
        trip_geom = trip_geom.to_crs(crs)
    return trip_geom


def sql_get_osm_for_trip(trip_id, engine, crs, **kwargs):
    """
    Get osm data for a given trip.
    
        Parameters
        ----------
            trip_id : int
                The id of the trip
            engine : sqlalchemy engine
            crs : pyproj crs object or string
                The crs everything is converted to. Can be anything that pyproj crs can read. 

                
        Returns
        -------
            GeoDataFrame
                Geopandas GeoDataframe containing the osm data
    """

    trip_geom = sql_get_trip_geom(trip_id, engine, crs=crs)
    osm_data = sql_get_osm_from_line(trip_geom, engine, **kwargs)
    return osm_data


def sql_get_inclination(trip_id: int, osm_data: GeoDataFrame, elevation_sampler: ElevationSampler, engine: Engine,
                        first_sample_distance: float = 10.0, brunnel_filter_length: float = 10.,
                        interpolated: bool = True, **kwargs) -> Tuple[DataFrame, DataFrame, DataFrame]:
    brunnels = get_osm_prop(osm_data, "brunnel", brunnel_filter_length=brunnel_filter_length)

    trip_geom = sql_get_trip_geom(trip_id, engine, elevation_sampler.dem.crs)

    x_coords, y_coords, distances, elevation = elevation_sampler.elevation_profile(trip_geom,
                                                                                   distance=first_sample_distance,
                                                                                   interpolated=interpolated)

    return process_ele(elevation, distances, brunnels, **kwargs)


def sql_get_timetable(trip_id: int, engine: Engine, min_stop_duration: float = 30.,
                      round_int: bool = True) -> DataFrame:
    # distance from start
    # station name
    # stop time at station in s
    # driving time to next station in s

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

    time_table = pd.read_sql_query(text(sql), con=engine, params={"trip_id": trip_id})

    s15 = pd.Timedelta(min_stop_duration * 0.5, unit="s")

    last_stop = time_table.stop_sequence.iloc[-1]
    first_stop = time_table.stop_sequence.iloc[0]

    # ignore first and last station
    ign_frst_last = (time_table.stop_sequence > first_stop) & (time_table.stop_sequence < last_stop)
    arr_eq_dep = ign_frst_last & (time_table.departure_time == time_table.arrival_time)

    # ANNAHME: Züge halten mindesten 30s
    # if arrival time = departure time, then arrival time -15 and departure time + 15
    time_table.loc[arr_eq_dep, ["arrival_time"]] -= s15
    time_table.loc[arr_eq_dep, ["departure_time"]] += s15

    # stop duration = dperature - arrival
    time_table["stop_duration"] = (time_table.departure_time - time_table.arrival_time).dt.total_seconds()

    # driving time to next station:
    # take arrival time of next station and substract departure time
    driving_time = time_table.arrival_time[1:].dt.total_seconds().values - time_table.departure_time[
                                                                           :-1].dt.total_seconds().values

    driving_time = np.append(driving_time, 0)

    time_table["driving_time"] = driving_time

    if round_int:
        time_table["dist"] = np.rint(time_table["dist"]).astype(int)
        time_table["stop_duration"] = np.rint(time_table["stop_duration"]).astype(int)
        time_table["driving_time"] = np.rint(time_table["driving_time"]).astype(int)

    return time_table[["dist", "stop_name", "stop_duration", "driving_time"]]


def sql_get_trip_title(trip_id: int, engine: Engine) -> str:
    stops = sql_get_trip_stops(trip_id, engine)
    start = stops["stop_name"].iloc[0]
    end = stops["stop_name"].iloc[-1]

    title = clean_name(start) + " - " + clean_name(end)
    return title


def clean_name(name: str) -> str:
    name = name.split()
    name = name[0].split('(')

    return name[0]


def sql_get_trip_data(trip_id: int, engine: Engine, elevation_sampler: ElevationSampler, crs: Any = 25832, **kwargs) \
        -> Tuple[str, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]:
    """
    Returns
    -------
        title, timetable, electrification, maxspeed, inclination
        and elevation if return ele=True
        elevation is a DataFrame with original distances,
    """

    osm_data = sql_get_osm_for_trip(trip_id, engine, crs)

    timetable = sql_get_timetable(trip_id, engine)

    title = str(trip_id) + " " + sql_get_trip_title(trip_id, engine)

    electrification = get_osm_prop(osm_data, "electrified")

    max_speed = get_osm_prop(osm_data, "maxspeed")

    ele_pipeline, ele_smoothed, inclination = sql_get_inclination(trip_id, osm_data,
                                                                  elevation_sampler,
                                                                  engine, **kwargs)

    return title, timetable, electrification, max_speed, inclination, ele_pipeline, ele_smoothed


def sql_get_osm(trip_id, engine, crs, get_osm_buffer=0.0001, filter_buffer=2, intersection_buffer=2,
                filter_difference_length=1):
    """
    Get osm data for a given trip.

        Parameters
        ----------
            trip_id : int
                The id of the trip
            engine : sqlalchemy engine
            crs : pyproj crs object or string
                The crs everything is converted to. Can be anything that pyproj crs can read.
            get_osm_buffer : float, default = 0.0001
                Buffer used to get all osm data arround the trip that lies in the buffer
            filter_buffer: float, default = 2
                Buffer that filters the osm data (start and endpoint of osm geometry have to lie in buffer)
            intersection_buffer : float, default = 2
                Buffer used arround geometries to get differece and intersection
            filter_difference_length : float, default = 1
                Filter segments smaller than this length from difference of osm and trip

        Returns
        -------
            GeoDataFrame
                Geopandas GeoDataframe containing the osm data
    """

    shape_id = sql_get_shape_id(trip_id, engine)
    osm_data = sql_get_osm_raw_by_shape_id(shape_id, engine, get_osm_buffer)

    trip_geom = sql_get_geometry(shape_id, engine)

    # convert the reference system so that they match
    trip_geom = trip_geom.to_crs(crs)
    osm_data = osm_data.to_crs(crs)

    osm_data['start_point'] = osm_data.apply(lambda r: Point(r['geom'].coords[0]), axis=1)
    osm_data['end_point'] = osm_data.apply(lambda r: Point(r['geom'].coords[-1]), axis=1)

    buffered_trip = trip_geom.buffer(filter_buffer)

    trip_contains_start = osm_data.apply(lambda r: buffered_trip.contains(r['start_point']), axis=1).iloc[:, 0]
    trip_contains_end = osm_data.apply(lambda r: buffered_trip.contains(r['end_point']), axis=1).iloc[:, 0]

    # correct osm data: where start and endpoint are in trip buffer
    correct_osm = osm_data[trip_contains_start & trip_contains_end]

    # create buffer around correct geoms and join to single goem for difference
    correct_geoms = correct_osm["geom"].buffer(intersection_buffer, cap_style=2)
    correct_geom = unary_union(correct_geoms)

    # get the difference of the single geom with the trip geom to detect the missing segments
    missing_mask = trip_geom.iloc[0].difference(correct_geom)
    missing_mask_pd = gpd.GeoDataFrame({'geometry': missing_mask}, crs=crs)

    # there are a bunch of tiny segments
    missing_mask_pd = missing_mask_pd[missing_mask_pd.length > filter_difference_length]

    # make intersection with osm data to get missing values
    # for the intersection to work, we need polygons --> make buffer arround linestring
    missing_mask_pd["geometry"] = missing_mask_pd.buffer(intersection_buffer, cap_style=1)

    # save old linestrings to convert polys back to linestring later
    osm_data["geom_old"] = osm_data["geom"]

    # also make osm geom buffered for intersection
    osm_data["geom"] = osm_data.buffer(intersection_buffer, cap_style=1)

    # get the intersection with osm data
    missing_osm = gpd.overlay(missing_mask_pd, osm_data, how='intersection')

    # convert back to linestring
    missing_osm["geom"] = missing_osm["geom_old"]
    missing_osm = missing_osm.drop(['geom_old'], axis=1)
    missing_osm = missing_osm.set_geometry('geom')
    missing_osm = missing_osm.drop(['geometry'], axis=1)

    # merge missing and correct osm data
    final_osm = pd.concat([missing_osm, correct_osm])

    final_osm = filter_overlapping_osm(final_osm, trip_geom)

    return final_osm


def sql_get_osm_raw_by_shape_id(shape_id, engine, intersect_buffer=0.0001):
    sql = """
    SELECT osm_railways.*
    FROM geo_shape_geoms
    LEFT JOIN osm_railways ON ST_intersects(geo_shape_geoms.geom,st_buffer(osm_railways.geom,:intersect_buffer))
    WHERE geo_shape_geoms.shape_id = :shape_id
    AND osm_railways.status = 'active'
    """
    osm_data = gpd.read_postgis(text(sql), geom_col='geom', con=engine,
                                params={"shape_id": shape_id, "intersect_buffer": intersect_buffer})

    return osm_data


def sql_get_osm_raw(trip_id, engine, intersect_buffer=0.0001):
    shape_id = sql_get_shape_id(trip_id, engine)
    osm_data = sql_get_osm_raw_by_shape_id(shape_id, engine, intersect_buffer)
    return osm_data
