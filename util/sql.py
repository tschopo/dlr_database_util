"""
Functions that interface with the dlr "liniendatenbank" database
"""

from typing import Any, Tuple, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from ElevationSampler import DEM
from geopandas import GeoSeries, GeoDataFrame
from pandas import DataFrame
from sqlalchemy import text
from sqlalchemy.engine import Engine

from .osm import get_osm_prop
from .osm import sql_get_osm_from_line
from .tpt import process_ele


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


def sql_get_inclination(trip_id: int, osm_data: GeoDataFrame, elevation_sampler: DEM, engine: Engine,
                        first_sample_distance: float = 10.0, brunnel_filter_length: float = 10.,
                        interpolated: bool = True, **kwargs) -> Tuple[DataFrame, DataFrame, DataFrame]:
    brunnels = get_osm_prop(osm_data, "brunnel", brunnel_filter_length=brunnel_filter_length)

    trip_geom = sql_get_trip_geom(trip_id, engine, elevation_sampler.dem.crs)

    elevation_profile = elevation_sampler.elevation_profile(trip_geom,
                                                            distance=first_sample_distance,
                                                            interpolated=interpolated)

    elevation = elevation_profile.get_elevations()
    distances = elevation_profile.get_distances()

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


def sql_get_trip_data(trip_id: int, engine: Engine, elevation_sampler: DEM, crs: Any = 25832, **kwargs) \
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
