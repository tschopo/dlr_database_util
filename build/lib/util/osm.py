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

def filter_overlapping_osm(osm_data, trip_geom):
    """
    Calculates overlapping segments along the trip geometry. Removes overlapping if they dont have status=active, or they are a service track.
    Adds start_point, end_point, end_point_distance, end_point_distance columns.
    Sorts the data after start_point.
    Aligns the geoms so they all point in same direction.
    """

    # calculate distances
    osm_data['start_point'] = osm_data.apply(lambda r:  Point(r['geom'].coords[0]),axis=1)
    osm_data['end_point'] = osm_data.apply(lambda r:  Point(r['geom'].coords[-1]),axis=1)

    osm_data['start_point_distance'] = osm_data.apply(lambda r: trip_geom.project(r['start_point']),axis=1)
    osm_data['end_point_distance'] = osm_data.apply(lambda r: trip_geom.project(r['end_point']),axis=1)


    # make linestrings all same direction
    osm_data["geom"] = osm_data.apply(lambda row: LineString(list(row['geom'].coords)[::-1]) if row['start_point_distance'] > row['end_point_distance'] else row['geom'], axis=1)

    # take into account maxspeed forward / backward
    # flip if wrong dir
    t = osm_data.apply(lambda row: row['maxspeed_backward'] if (row['start_point_distance'] > row['end_point_distance']) else row['maxspeed_forward'], axis=1)
    osm_data["maxspeed_backward"] = osm_data.apply(lambda row: row['maxspeed_forward'] if (row['start_point_distance'] > row['end_point_distance']) else row['maxspeed_backward'], axis=1)
    osm_data["maxspeed_forward"] = t
     	 	 	
    # recalculate distances and sort
    osm_data['start_point'] = osm_data.apply(lambda r:  Point(r['geom'].coords[0]),axis=1)
    osm_data['end_point'] = osm_data.apply(lambda r:  Point(r['geom'].coords[-1]),axis=1)

    # sort the osm geoms along the trip
    osm_data['start_point_distance'] = osm_data.apply(lambda r: trip_geom.project(r['start_point']),axis=1)
    osm_data['end_point_distance'] = osm_data.apply(lambda r: trip_geom.project(r['end_point']),axis=1)

    osm_data.sort_values(['start_point_distance','end_point_distance'], inplace=True)
    osm_data.reset_index(drop=True, inplace=True)

    # get overlapping segments
    # important: data must be sorted sfter start_distance
    # def get_redundant_segments(osm_data) returns subset of osm_data where doubled
    overlapping = []

    for i, row in osm_data.iterrows():
        overlapping_idx = osm_data[i+1:].index[row["end_point_distance"] > osm_data[i+1:]["start_point_distance"]].tolist()
        overlapping += overlapping_idx

        if len(overlapping_idx) > 0:
            overlapping += [i]
    overlapping = list(set(overlapping))

    # remove from overlapping where status not active
    osm_data = osm_data.drop(osm_data.iloc[overlapping][osm_data.iloc[overlapping]["status"]!="active"].index)
    # remove overlapping service tracks 
    osm_data = osm_data.drop(osm_data.iloc[overlapping][osm_data.iloc[overlapping]["service"].notnull()].index)
    
    return osm_data
    
def get_osm_prop(osm_data, prop, brunnel_filter_length = 10, round_int = True):
    """
    Get dataframe of start_dist, end_dists, property value, for a given osm property. Merges adjacent sections with same value.
    
    Parameters
    ----------
        
        osm_data : GeoDataFrame
        prop : str
            The OSM property. Accepted Values are "brunnel", "electrified" or "maxspeed"
            
    Returns
    -------
    
        DataFrame
            DataFrame with propertie values, start_dist and end_dist. If brunnel then also length. 
    """
    
    osm_prop_data = osm_data.copy()
    
    if prop == "brunnel":
        filter_bool = (osm_prop_data.bridge=='yes') | (osm_prop_data.tunnel == 'yes')
        osm_prop_data["brunnel"] = np.where(filter_bool, "yes", "no")
        #osm_prop_data = osm_prop_data[filter_bool]
    elif prop == "maxspeed":  
        # if maxspeed not specified take maxspeed forward
        osm_prop_data["maxspeed"] = np.where(np.isnan(osm_prop_data["maxspeed"]), osm_prop_data["maxspeed_forward"], osm_prop_data["maxspeed"])
        # filter nans
        osm_prop_data = osm_prop_data[~np.isnan(osm_prop_data.maxspeed)]
    elif prop == "electrified":
        # filter unknown
        osm_prop_data = osm_prop_data[osm_prop_data.electrified != "unknown"]
    

    # TODO How to handle nan values. If None, then Nans/unknown are ignored. If value, then Nans are replaced with the value.
    # TODO: if all NAN
    # bei maxspeed am anfang 60 anehmen wenn erster wert fehlt?
    
    # create new dataframe with values
    # start_distance, end_distance, prop_value

    # go through dataframe
    # check if the value changes.
    # if it changes then save old_start, prev_row.end_point_distance, old_value

    old_val = osm_prop_data.iloc[0][prop]
    
    
    # always start at 0?
    old_start = 0#osm_prop_data.iloc[0]["start_point_distance"]

    start_dists = []
    end_dists = []
    prop_vals = []

    prev_row = osm_prop_data.iloc[0]
    for index, row in osm_prop_data.iterrows():
        if row[prop] != old_val:
            start_dists.append(old_start)
            end_dists.append(prev_row["end_point_distance"])
            prop_vals.append(old_val)

            old_val = row[prop]
            old_start = row["start_point_distance"]

        prev_row = row.copy()

    start_dists.append(old_start)
    end_dists.append(prev_row["end_point_distance"])
    prop_vals.append(old_val)    

    data = {prop : prop_vals, "start_dist" : start_dists, "end_dist" : end_dists}
    props = pd.DataFrame.from_dict(data)
    

    if prop == "brunnel":
        props = props[props.brunnel == "yes"]

        props["length"] = props["end_dist"] - props["start_dist"]
        props = props[props.length > brunnel_filter_length]
        
        #props["brunnel"] = np.where(props.bridge == "yes", "bridge", "tunnel")
    elif prop == "electrified":
        props["electrified"] = np.where(props.electrified == "yes", 1, 0)
        props["electrified"] = props["electrified"].astype(int)
    
    if round_int:
        props["start_dist"] = np.rint(props["start_dist"]).astype(int)
        props["end_dist"] = np.rint(props["end_dist"]).astype(int)
        
        if prop == "maxspeed":
            props["maxspeed"] = np.rint(props["maxspeed"]).astype(int)
        
    return props
