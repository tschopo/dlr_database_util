from ElevationSampler import *
from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry import LineString
from shapely.ops import unary_union
import xlsxwriter
import re


def sql_get_shape_id(trip_id, engine):
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
    
    sql = """\
    SELECT geo_shape_geoms.shape_id 
    FROM geo_trips, geo_shape_geoms
    WHERE geo_trips.shape_id = geo_shape_geoms.shape_id
    AND geo_trips.trip_id = :trip_id
    """
    shape_id = int(pd.read_sql_query(text(sql), con=engine, params={"trip_id":trip_id}).iloc[0][0])

    return shape_id

def sql_get_geometry(shape_id, engine):
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
                Geopandas Geoseries containing the geometry
    """
    
    sql = """
    SELECT *
    FROM geo_shape_geoms
    WHERE shape_id = :shape_id
    """
    geo_shape_geom = gpd.read_postgis(text(sql), con=engine, geom_col='geom', params={"shape_id":shape_id})
    trip_geom = geo_shape_geom.geom
    
    return trip_geom

def sql_get_trip_geom(trip_id, engine, crs=None):
    
    
    shape_id = sql_get_shape_id(trip_id, engine)
    trip_geom = sql_get_geometry(shape_id, engine)
    
    if crs:
        trip_geom = trip_geom.to_crs(crs)
    return trip_geom
    
def sql_get_osm_raw_by_shape_id(shape_id, engine, intersect_buffer = 0.0001):

    sql = """
    select osm_railways.*
    from geo_shape_geoms
    left join osm_railways on ST_intersects(geo_shape_geoms.geom,st_buffer(osm_railways.geom,:intersect_buffer))
    where geo_shape_geoms.shape_id = :shape_id
    and osm_railways.status = 'active'
    """
    osm_data = gpd.read_postgis(text(sql), geom_col='geom', con=engine,params={"shape_id":shape_id, "intersect_buffer":intersect_buffer})
    
    return osm_data

def sql_get_osm_raw(trip_id, engine, intersect_buffer = 0.0001):

    shape_id = sql_get_shape_id(trip_id, engine)
    osm_data = sql_get_osm_raw_by_shape_id(shape_id, engine, intersect_buffer)
    return osm_data
    
def sql_get_osm(trip_id, engine, crs, get_osm_buffer = 0.0001, filter_buffer = 2, intersection_buffer = 2, filter_difference_length = 1):
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

    osm_data['start_point'] = osm_data.apply(lambda r:  Point(r['geom'].coords[0]),axis=1)
    osm_data['end_point'] = osm_data.apply(lambda r:  Point(r['geom'].coords[-1]),axis=1)

    buffered_trip = trip_geom.buffer(filter_buffer)

    trip_contains_start = osm_data.apply(lambda r: buffered_trip.contains(r['start_point']),axis=1).iloc[:,0]
    trip_contains_end = osm_data.apply(lambda r: buffered_trip.contains(r['end_point']),axis=1).iloc[:,0]

    # correct osm data: where start and endpoint are in trip buffer
    correct_osm = osm_data[trip_contains_start & trip_contains_end]

    # create buffer arround correct geoms and join to single goem for difference
    correct_geoms = correct_osm["geom"].buffer(intersection_buffer, cap_style=2)
    correct_geom = unary_union(correct_geoms)

    # get the difference of the single geom with the trip geom to detect the missing segments
    missing_mask = trip_geom.iloc[0].difference(correct_geom)
    missing_mask_pd = gpd.GeoDataFrame({'geometry':missing_mask}, crs = crs)
    missing_mask_pd = missing_mask_pd[missing_mask_pd.length > filter_difference_length] # there are a bunch of tiny segments

    # make intersection with osm data to get missing values
    # for the intersection to work, we need polygons --> make buffer arround linestring
    missing_mask_pd["geometry"] = missing_mask_pd.buffer(intersection_buffer, cap_style=1)

    # save old linestrings to convert polys back to linestring later
    osm_data["geom_old"] =  osm_data["geom"]

    # also make osm geom buffered for intersection
    osm_data["geom"] =  osm_data.buffer(intersection_buffer, cap_style=1)

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
