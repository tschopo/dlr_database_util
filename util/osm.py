"""
Functions that interface with the OSM rail database as defined by the import script
"""
import os
import re
import subprocess
import urllib.request
from shutil import which
from typing import Optional, Union, Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
from geopandas import GeoDataFrame, GeoSeries
from pandas import DataFrame
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.ops import transform, unary_union
from sqlalchemy import text
from sqlalchemy.engine import Engine


def sql_get_srid(engine: Engine, schema: str = "public", table: str = "osm_railways", geom_column: str = "geom") -> str:
    sql = """
        SELECT Find_SRID(:schema, :table, :geom_column);
        """

    srid_df = pd.read_sql_query(text(sql), con=engine,
                                params={"schema": schema, "table": table, "geom_column": geom_column})

    return srid_df.iloc[0][0]


def sql_get_osm_from_line(linestring: Union[LineString, GeoSeries], engine: Engine, crs: Optional[Any] = None,
                          table_srid: Optional[int] = None, get_osm_buffer: float = 0.0001,
                          filter_difference_length: float = 1., schema: str = "public",
                          table: str = "osm_railways", geom_column: str = "geom", ) -> GeoDataFrame:
    """
    Get osm data for a LineString.

    Parameters
    ----------
    engine : SqlAlchemy Engine
    linestring : shapely LineString or GeoSeries
    crs : shapely crs string or object
        the crs of the LineString, if linestring is a LineString. If linestring is a GeoSeries the CRS of the GeoSeries
        is taken
    table_srid : int, optional
        The srid of the osm table. If not provided, retrieves the srid from the database
    get_osm_buffer : float, default = 0.0001
                Buffer used to get all osm data around the trip that lies in the buffer
    filter_difference_length : float, default = 1
        Filter segments smaller than this length from difference of osm and trip
    schema
    table
    geom_column

    Returns
    -------
        geopandas GeoDataFrame
    """

    # convert the linestring to the same crs as GeoDataframe
    if table_srid is None:
        table_srid = int(sql_get_srid(engine, schema=schema, table=table, geom_column=geom_column))

    crs_to = pyproj.CRS(table_srid)

    if crs is not None:
        crs = pyproj.CRS(crs)

    if isinstance(linestring, GeoSeries):
        crs = linestring.crs
        linestring_pd = linestring
        linestring = linestring.iloc[0]
    else:
        linestring_pd = gpd.GeoSeries(linestring, name="geom", crs=crs)

    if crs is None:
        raise Exception("parameter linestring must be GeoSeries or crs parameter must be given")

    project = pyproj.Transformer.from_crs(crs, crs_to, always_xy=True).transform
    linestring = transform(project, linestring)

    # no parameterized input for column names
    geom_column = re.sub('[^A-Za-z0-9_]+', '', geom_column)
    table = re.sub('[^A-Za-z0-9_]+', '', table)
    schema = re.sub('[^A-Za-z0-9_]+', '', schema)

    sql = """
        SELECT *
        FROM {schema}.{table}
        WHERE ST_intersects(ST_GeomFromText(:linestring,:srid),st_buffer({geom_column},:intersect_buffer))
        """.format(schema=schema, table=table, geom_column=geom_column)

    with engine.connect() as connection:
        osm_data = gpd.read_postgis(text(sql), geom_col='geom', con=connection,
                                    params={"linestring": linestring.wkt, "srid": table_srid,
                                            "intersect_buffer": get_osm_buffer})

    osm_data[["way_id",
              "maxspeed",
              "maxspeed_forward",
              "maxspeed_backward",
              'gauge',
              'voltage',
              'frequency']] = osm_data[["way_id",
                                        "maxspeed",
                                        "maxspeed_forward",
                                        "maxspeed_backward",
                                        'gauge',
                                        'voltage',
                                        'frequency']].fillna(np.nan)

    # set the dtypes
    new_dtypes = {"way_id": np.int64,
                  "type": str,
                  "status": str,
                  "electrified": str,
                  "electrification": str,
                  "maxspeed": np.float64,
                  "maxspeed_forward": np.float64,
                  "maxspeed_backward": np.float64,
                  # "tunnel": str,
                  # "bridge": str,
                  # "bridge_type": str,
                  # 'tunnel_type': str,
                  # 'embankment': str,
                  # 'cutting': str,
                  # 'ref': 'Int64',
                  # 'gauge': np.float64,
                  'traffic_mode': str,  # passenger / mixed / freight
                  'service': str,
                  'usage': str,  # branch / main
                  # 'voltage': np.float64,
                  # 'frequency': np.float64,
                  }

    osm_data = osm_data.astype(new_dtypes)

    # convert the reference system so that they match
    trip_geom = linestring_pd
    osm_data = osm_data.to_crs(crs)

    assert trip_geom.crs.equals(osm_data.crs)

    osm_data['start_point'] = osm_data.apply(lambda r: Point(r['geom'].coords[0]), axis=1)
    osm_data['end_point'] = osm_data.apply(lambda r: Point(r['geom'].coords[-1]), axis=1)

    final_osm = combine_osm_pipeline(osm_data, trip_geom, filter_difference_length=filter_difference_length)

    final_osm = final_osm.drop_duplicates(subset=["way_id"])

    final_osm = finalize_osm(final_osm, trip_geom)

    return final_osm


def get_trip_where_no_osm(trip_geom: GeoSeries, osm_data: GeoDataFrame, buffer_size: float = 8.,
                          filter_difference_length: float = 1.) -> GeoSeries:
    """
    Get the segments of trip_geom where there is no osm data.
    For a given trip and incomplete OSM data (meaning that the intersection of osm data and trip does not return the
    whole trip), returns the segments of the trip where there is no OSM data.

    Parameters
    ----------
    trip_geom
    osm_data
    buffer_size
    filter_difference_length
        filter osm segments smaller than this length

    Returns
    -------

    """

    # create buffer around correct geoms and join to single geom
    correct_geoms = osm_data["geom"].buffer(buffer_size, cap_style=2)
    correct_geom = unary_union(correct_geoms)

    # get the difference of the single geom with the trip geom to detect the missing segments
    missing_mask = trip_geom.iloc[0].difference(correct_geom)

    if missing_mask.geom_type == 'LineString':
        missing_mask = [missing_mask]
    missing_trip_segments = gpd.GeoDataFrame({'geometry': missing_mask}, crs=osm_data.crs)

    # there are a bunch of tiny segments
    missing_trip_segments = missing_trip_segments[missing_trip_segments.length > filter_difference_length]

    return missing_trip_segments


def get_osm_in_buffer(trip_geom: Union[GeoSeries, GeoDataFrame], osm_data: GeoDataFrame, buffer_size: float,
                      method: str = "strict") \
        -> GeoDataFrame:
    """
    Returns OSM data that lies in the buffer_size

    Parameters
    ----------
    trip_geom
    osm_data
    buffer_size
    method
        'strict', 'both' or 'either'
        strict: the whole geometry needs to be in buffer
        both: start and end point of geometry need to be in buffer
        either: start or end point need to be in buffer

    Returns
    -------

    """

    buffered_trip = trip_geom.buffer(buffer_size)
    osm_data = osm_data.copy()

    if method == "strict":
        trip_contains_geom = osm_data.apply(lambda r: np.any(buffered_trip.contains(r['geom'])), axis=1)
    else:
        trip_contains_start = osm_data.apply(lambda r: np.any(buffered_trip.contains(r['start_point'])), axis=1)
        trip_contains_end = osm_data.apply(lambda r: np.any(buffered_trip.contains(r['end_point'])), axis=1)
        if method == "both":
            trip_contains_geom = trip_contains_start & trip_contains_end
        elif method == "either":
            trip_contains_geom = trip_contains_start | trip_contains_end
        else:
            raise Exception("Method must be 'strict', 'both' or 'either'!")

    return osm_data[trip_contains_geom]


def combine_osm_pipeline(osm_data: GeoDataFrame, trip_geom: GeoSeries,
                         filter_difference_length: float = 1., search_buffer: float = 8.) -> GeoDataFrame:
    osm_buf_1 = get_osm_in_buffer(trip_geom, osm_data, 1, method="strict")
    osm_buf_2 = get_osm_in_buffer(trip_geom, osm_data, 2, method="strict")
    osm_buf_3 = get_osm_in_buffer(trip_geom, osm_data, 3, method="strict")
    osm_buf_4 = get_osm_in_buffer(trip_geom, osm_data, 4, method="both")
    osm_buf_5 = get_osm_in_buffer(trip_geom, osm_data, 2, method="either")  # for the endings
    osm_buf_6 = get_osm_in_buffer(trip_geom, osm_data, 5, method="either")

    osm_pyramid = [osm_buf_2, osm_buf_3, osm_buf_4, osm_buf_5, osm_buf_6]
    final_osm = osm_buf_1

    for osm_buf in osm_pyramid:
        missing_segments = get_trip_where_no_osm(trip_geom, final_osm, search_buffer,
                                                 filter_difference_length=filter_difference_length)
        missing_osm = get_osm_in_buffer(missing_segments, osm_buf, search_buffer, method="either")
        final_osm = pd.concat([missing_osm, final_osm])

    return final_osm


def calc_distances(osm_data: GeoDataFrame, trip_geom: GeoSeries, geom_col: str = "geom") -> GeoDataFrame:
    """
    Calculates the Distance of each feature in osm_data along the trip.
    Adds start_point, end_point, start_point_distance, end_point_distance columns

    Parameters
    ----------
    osm_data
    trip_geom
    geom_col

    Returns
    -------

    """
    osm_data = osm_data.copy()
    osm_data['start_point'] = osm_data.apply(lambda r: Point(r[geom_col].coords[0]), axis=1)
    osm_data['end_point'] = osm_data.apply(lambda r: Point(r[geom_col].coords[-1]), axis=1)

    osm_data['start_point_distance'] = osm_data.apply(lambda r: trip_geom.project(r['start_point']), axis=1)
    osm_data['end_point_distance'] = osm_data.apply(lambda r: trip_geom.project(r['end_point']), axis=1)

    return osm_data


def finalize_osm(osm_data: GeoDataFrame, trip_geom, filter_inactive: bool = False):
    """
    Calculates overlapping segments along the trip geometry. Removes overlapping if they dont have status=active,
    or they are a service track.
    Adds start_point, end_point, end_point_distance, end_point_distance columns.
    Sorts the data after start_point.
    Aligns the geoms so they all point in same direction.

    Parameters
    ----------

    osm_data
    trip_geom
    filter_inactive
        if True filters overlapping segments with status != active and service tracks
        if False then no overlapping segments are discarded.

    """

    # calculate distances
    osm_data = calc_distances(osm_data, trip_geom)

    # make linestrings all same direction
    osm_data["geom"] = osm_data.apply(
        lambda r: LineString(list(r['geom'].coords)[::-1]) if r['start_point_distance'] > r[
            'end_point_distance'] else r['geom'], axis=1)

    # take into account maxspeed forward / backward
    # flip if wrong dir
    t = osm_data.apply(
        lambda r: r['maxspeed_backward'] if (r['start_point_distance'] > r['end_point_distance']) else r[
            'maxspeed_forward'], axis=1)
    osm_data["maxspeed_backward"] = osm_data.apply(
        lambda r: r['maxspeed_forward'] if (r['start_point_distance'] > r['end_point_distance']) else r[
            'maxspeed_backward'], axis=1)
    osm_data["maxspeed_forward"] = t

    # recalculate distances and sort
    osm_data = calc_distances(osm_data, trip_geom)

    osm_data.sort_values(['start_point_distance', 'end_point_distance'], inplace=True)
    osm_data.reset_index(drop=True, inplace=True)

    # problem when the trip_geom has overlapping segments (e.g. at "Kopfbahnhöfe")
    # identify overlapping segments. in the overlapping segments duplicate the osm data with flipped geometry
    # get the segments where trip_geom crosses itself
    linestrings = []
    for a, b in zip(trip_geom.iloc[0].coords, trip_geom.iloc[0].coords[1:]):
        linestrings.append(LineString([a, b]))

    l = []
    for l1, l2, l3 in zip(linestrings, linestrings[1:], linestrings[2:]):
        if not LineString(l1.coords[:] + l2.coords[:] + l3.coords[:]).is_simple:
            l.append(l1)
            l.append(l2)
            l.append(l3)
    segments = gpd.GeoDataFrame({"geometry": l})

    # get the osm data in these segments
    doubled_osm = get_osm_in_buffer(segments, osm_data, 5, method="either")

    for index, row in doubled_osm.iterrows():
        row["geom"] = LineString(row["geom"].coords[::-1])

        t_spd = row["start_point_distance"]
        t_sp = row["start_point"]
        row["start_point_distance"] = row["end_point_distance"]
        row["start_point"] = row["end_point"]
        row["end_point_distance"] = t_spd
        row["end_point"] = t_sp

        # add to osm with flipped geometries
        osm_data.append(row)

    # make sure shorter segments come before longer segments if same start point
    osm_data.sort_values(['start_point_distance', 'end_point_distance'], inplace=True)
    osm_data.reset_index(drop=True, inplace=True)

    # get overlapping segments
    # important: data must be sorted after start_distance
    # step 1: assign plausible values to maxspeed, electrified, brunnels for all overlapping segments
    overlapping = []
    for i, row in osm_data.iterrows():

        # find all overlapping segments for this segment
        overlapping_idx = osm_data[i + 1:].index[
            row["end_point_distance"] > osm_data[i + 1:]["start_point_distance"]].tolist()

        # add this segment to overlapping
        if len(overlapping_idx) > 0:
            overlapping_idx += [i]

            overlapping += overlapping_idx

            # set maxspeed to highest value
            maxspeeds = osm_data.iloc[overlapping_idx]["maxspeed"].values
            if not np.isnan(maxspeeds).all():
                osm_data.iloc[i, osm_data.columns.get_loc("maxspeed")] = np.nanmax(maxspeeds)

            # set electrified to yes if there is a yes
            electrified = osm_data.iloc[overlapping_idx]["electrified"].values
            if "yes" in electrified:
                osm_data.iloc[i, osm_data.columns.get_loc("electrified")] = "yes"

            # set tunnel to yes if there is a yes
            tunnels = osm_data.iloc[overlapping_idx]["tunnel"].values
            if "yes" in tunnels:
                osm_data.iloc[i, osm_data.columns.get_loc("tunnel")] = "yes"

            # set bridge to yes if there is a yes
            bridges = osm_data.iloc[overlapping_idx]["bridge"].values
            if "yes" in bridges:
                osm_data.iloc[i, osm_data.columns.get_loc("bridge")] = "yes"

    # step 2:
    # set the endpoint to the next startpoint for overlapping segments
    drop_idx = []
    for i in range(osm_data.shape[0]-1):
        # if overlaps
        if osm_data.at[i, "end_point_distance"] > osm_data.at[i+1, "start_point_distance"]:

            # adjust endpoint
            osm_data.at[i, "end_point_distance"] = osm_data.at[i + 1, "start_point_distance"]

            # it can happen that the segment has 0 length --> drop
            if osm_data.at[i, "start_point_distance"] >= osm_data.at[i, "end_point_distance"]:
                drop_idx.append(i)

    osm_data.drop(drop_idx, inplace=True)

    # from overlapping filter sections that are not active or are service tracks
    # remove isin(overlapping) and ((status != active) or (service != 'None))
    if filter_inactive:
        keep = (~osm_data.index.isin(overlapping)) | ((osm_data.status == "active") & (osm_data.service == "None"))

        osm_data = osm_data[keep]

    osm_data.reset_index(drop=True, inplace=True)

    # this way there are tracks lost. add final step: compute missing segments
    #  get all osm in missing segment
    #  create new osm row where geom is missing segment, and values are computed from the osm segments
    #  e.g. max(maxspeeds), tunnel = yes if any yes, bridge = yes if any yes, electrified = yes if any yes ...

    return osm_data


def get_osm_prop(osm_data: GeoDataFrame, prop: str, brunnel_filter_length: float = 10., round_int: bool = True,
                 train_length: Optional[float] = 150., maxspeed_if_all_null: float = 120.,
                 maxspeed_null_segment_length=1000., maxspeed_min_if_null: float = 60.,
                 trip_length: Optional[float] = None, harmonize_end_dists: bool = True,
                 set_unknown_electrified_to_no: bool = True, tpt: bool = True):
    """
    Get dataframe of start_dist, end_dists, property value, for a given osm property. Merges adjacent sections with same
    value.

    Parameters
    ----------

        osm_data : GeoDataFrame
            Columns start_point_distance, end_point_distance, prop [brunnel, electrified, maxspeed]
            brunnel: 'yes'|'no', electrified: 'yes'|'no'|'unknown', maxspeed: int|np.nan

            unknown and nan values are discarded.

            For electrified and maxspeed the last end_point_distance must be the trip length.

        prop : str
            The OSM property. Accepted Values are "brunnel", "electrified" or "maxspeed"
        brunnel_filter_length
            ignores brunnels that are smaller than this length
        round_int : bool
            return ints not floats for start_dist, end_dist, maxspeed
        train_length : float or None
            The minimum length of maxspeed spikes. Should be larger than train length.
        maxspeed_if_all_null : float
            The maxspead that is set if there are 0 maxspeeds present
        maxspeed_null_segment_length : float
            For Segments of nan values longer than this, the maxspeed is set to the median of the trip maxspeed.
        maxspeed_min_if_null : float
            For nan segments: if median of trip is below this value, set the nan segment to maxspeed_if_all_null
        trip_length : float or None
            Set the last end_dist to trip_length, so that align with trip, only for maxspeed an electrified
        harmonize_end_dists : bool
            if true, sets the end_dists to the start dist of next row, so that there are no gaps, only for maxspeed and
            electrified
        set_unknown_electrified_to_no : bool
            if false, then unknown electrified are ignored, meaning that always the previous known value is used until
            next known value. if true then unknown segments are set to "no"
        tpt : bool
            set true if using the outputs for tpt simulator. fixes tpt bugs

    Returns
    -------
    
        DataFrame
            DataFrame with property values, start_dist and end_dist. If brunnel then also length.
    """

    osm_prop_data = osm_data.copy()

    if prop == "brunnel":
        filter_bool = (osm_prop_data.bridge == 'yes') | (osm_prop_data.tunnel == 'yes')
        osm_prop_data["brunnel"] = np.where(filter_bool, "yes", "no")
        # osm_prop_data = osm_prop_data[filter_bool]
    elif prop == "maxspeed":
        if 'maxspeed_forward' in osm_prop_data.columns:

            # take max of maxspeed forward and maxspeed
            # if maxspeed not specified take maxspeed forward
            osm_prop_data[prop] = np.fmax(osm_prop_data["maxspeed_forward"].values, osm_prop_data["maxspeed"].values)

            # if maxspeed and maxspeed forward not specified take maxspeed backward
            if 'maxspeed_backward' in osm_prop_data.columns:
                osm_prop_data[prop] = np.where(np.isnan(osm_prop_data[prop]),
                                               osm_prop_data["maxspeed_backward"],
                                               osm_prop_data[prop])

        # if all nans, set maxspeed 100
        if osm_prop_data[prop].isnull().all():
            osm_prop_data[prop] = maxspeed_if_all_null

        # check for long segments with nans. set long segments to median maxspeed of trip
        # - calculate nan segment lengths
        # - create a list of segments where a segment has start, end, and list of indexes of elements that belong to the
        # segment
        # - for all indexes that are in segments that are longer than maxspeed_null_segment_length, set to median speed

        class Segment:
            start = None
            end = None
            length = None
            members = []

        segments = []

        current_segment = None
        segment_open = False
        segment_end_candidate = None
        for index, row in osm_prop_data.iterrows():

            # if nan
            if pd.isnull(row[prop]):

                # save the start_dist if new segment
                if not segment_open:
                    current_segment = Segment()
                    current_segment.start = row["start_point_distance"]

                current_segment.members.append(index)
                segment_end_candidate = row["end_point_distance"]
                segment_open = True

            else:
                # add end value
                if segment_open:
                    current_segment.end = segment_end_candidate
                    current_segment.length = current_segment.end - current_segment.start
                    segments.append(current_segment)

                # close the segment
                segment_open = False

        # close segment if still open
        if segment_open:
            current_segment.end = segment_end_candidate
            current_segment.length = current_segment.end - current_segment.start
            segments.append(current_segment)

        # go through segments. if a segment is long then set all maxspeeds to median of trip
        median_maxspeed = spatial_median(osm_prop_data)

        # if the median is very low, set it to 100
        if median_maxspeed < maxspeed_min_if_null:
            median_maxspeed = maxspeed_if_all_null

        for segment in segments:
            if segment.length > maxspeed_null_segment_length:
                osm_prop_data.loc[segment.members, prop] = median_maxspeed

        # filter nans
        osm_prop_data = osm_prop_data[~np.isnan(osm_prop_data.maxspeed)]

    elif prop == "electrified":

        # if all unknown set to not electrified
        if (osm_prop_data[prop] == "unknown").all():
            osm_prop_data[prop] = "no"

        # filter unknown
        if set_unknown_electrified_to_no:
            osm_prop_data[prop] = np.where(osm_data.electrified == "unknown", "no", osm_data.electrified)
        else:
            osm_prop_data = osm_prop_data[osm_prop_data.electrified != "unknown"]

    # reset index because removed elements
    osm_prop_data.reset_index(drop=True, inplace=True)

    # create new dataframe with values
    # start_distance, end_distance, prop_value

    # Idea: go through dataframe
    # check if the value changes.
    # if it changes then save old_start, prev_row.end_point_distance, old_value

    old_val = osm_prop_data.iloc[0][prop]

    # always start at 0
    old_start = 0

    start_dists = []
    end_dists = []
    prop_vals = []

    prev_row = osm_prop_data.iloc[0]
    for index, row in osm_prop_data.iterrows():

        # We look if the new val changes, if it changes we add the previous value. This way we know the end distance.
        if row[prop] != old_val:
            start_dists.append(old_start)
            end_dists.append(prev_row["end_point_distance"])
            prop_vals.append(old_val)

            # save the values for the next segment
            old_val = row[prop]
            old_start = row["start_point_distance"]

        prev_row = row.copy()

    start_dists.append(old_start)
    end_dists.append(prev_row["end_point_distance"])
    prop_vals.append(old_val)

    data = {prop: prop_vals, "start_dist": start_dists, "end_dist": end_dists}
    props = pd.DataFrame.from_dict(data)

    # set end dist to trip length so that dists are aligned with trip length
    if prop == "electrified" or prop == "maxspeed":
        if trip_length is not None:
            props.iloc[-1, props.columns.get_loc('end_dist')] = trip_length
        if harmonize_end_dists:
            props.iloc[0, props.columns.get_loc('start_dist')] = 0
            props.iloc[0:-1, props.columns.get_loc('end_dist')] = props.iloc[1:,
                                                                             props.columns.get_loc('start_dist')].values

    if prop == "brunnel":
        props = props[props.brunnel == "yes"]

        props["length"] = props["end_dist"] - props["start_dist"]
        props = props[props.length > brunnel_filter_length]

        # props["brunnel"] = np.where(props.bridge == "yes", "bridge", "tunnel")
    elif prop == "electrified":

        props["electrified"] = np.where(props.electrified == "yes", 1, 0)
        props["electrified"] = props["electrified"].astype(int)

    elif prop == "maxspeed":

        if train_length is not None:

            # filter maxspeed segments that are small and are "spikes" e.g. that are higher than their neighbours.

            # removing the spikes can lead to more spikes, so we have to repeat until there are no more spikes
            changed = True
            while changed:

                changed = False
                drop_idx = []
                for i in range(1, props.shape[0] - 1):

                    # segment length is calculated only from the start dists
                    segment_length = props.iloc[i + 1]["start_dist"] - props.iloc[i]["start_dist"]

                    if segment_length <= train_length \
                            and props.iloc[i - 1][prop] < props.iloc[i][prop] > props.iloc[i + 1][prop]:
                        # remove the row and extend the previous segment
                        drop_idx.append(i)
                        props.iloc[i - 1]["end_dist"] += segment_length
                        changed = True

                props.drop(drop_idx, inplace=True)
                props.reset_index(drop=True, inplace=True)

            # TPT bug: If maxspeed reduction after short distance from start, it crashes. --> Remove the short segment
            while tpt and \
                    props.shape[0] > 1 and \
                    props.iloc[1]["start_dist"] <= train_length and \
                    props.iloc[1]["maxspeed"] <= props.iloc[0]["maxspeed"]:

                props.iloc[1, props.columns.get_loc("start_dist")] = 0
                props.drop([0], inplace=True)
                props.reset_index(drop=True, inplace=True)

            # TPT bug: If maxspeed raise in short distance from end, it crashes. --> Remove the short segment
            # use larger buffer since distances are not exact
            while tpt and \
                    props.shape[0] > 1 and \
                    props.iloc[-1]["end_dist"] - props.iloc[-1]["start_dist"] <= 500 and \
                    props.iloc[-1]["maxspeed"] >= props.iloc[-2]["maxspeed"]:
                props.iloc[-2, props.columns.get_loc("end_dist")] = props.iloc[-1]["end_dist"]
                props.drop([props.shape[0] - 1], inplace=True)
                props.reset_index(drop=True, inplace=True)

    if round_int:

        if prop == "brunnel":
            # for brunnels we have to prevent that enddist = next start_dist
            props["start_dist"] = np.ceil(props["start_dist"]).astype(int)
            props["end_dist"] = np.floor(props["end_dist"]).astype(int)
        else:
            props["start_dist"] = np.rint(props["start_dist"]).astype(int)
            props["end_dist"] = np.rint(props["end_dist"]).astype(int)

        if prop == "maxspeed":
            props["maxspeed"] = np.rint(props["maxspeed"]).astype(int)

    props.reset_index(drop=True, inplace=True)
    return props


def spatial_median(osm_data, prop="maxspeed"):
    """
    calculate the weighted median. repeat each value by length.

    Parameters
    ----------

    Returns
    -------

    """

    osm_data = osm_data.copy()

    osm_data["length"] = osm_data.length
    vals = []
    for index, row in osm_data.iterrows():
        vals = vals + ([row[prop]] * int(row["length"]))

    vals = np.array(vals)
    median = np.nanmedian(vals)

    return median


def osm_railways_to_psql(geofabrik_pbf_folder: str, geofabrik_pbf: str, database="liniendatenbank", user="postgres", password=None, osmium_filter = True):
    """
    Warning not tested

    Parameters
    ----------

    Returns
    -------

    """

    # filter geofabrik germany osm data to only include railway data, to speedup import
    # test if osmium is installed
    if which('osmium') is not None and osmium_filter:
        os.system('osmium tags-filter -o ' + geofabrik_pbf_folder + '/filtered.osm.pbf '
                  + geofabrik_pbf +
                  ' nwr/railway nwr/disused:railway '
                  'nwr/abandoned:railway nwr/razed:railway nwr/construction:railway nwr/proposed:railway '
                  'nwr/planned:railway n/public_transport=stop_position nwr/public_transport=platform r/route=train '
                  'r/route=tram r/route=light_rail')
        geofabrik_pbf = 'filtered.osm.pbf'

    # osm2pgsql -d liniendatenbank -U postgres -W -O flex -S railways.lua data/germany-railway.osm.pbf
    lua_path = os.path.dirname(os.path.abspath(__file__)) + '/railways.lua'
    geofabrik_pbf_path = geofabrik_pbf_folder + '/' + geofabrik_pbf
    proc = subprocess.Popen(['osm2pgsql', '-d', database, '-U', user, '-W', '-O', 'flex', '-S', lua_path,
                             geofabrik_pbf_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    proc.stdin.write(password+'\n')
    proc.stdin.flush()

    o, e = proc.communicate()

    print('Output: ' + o)
    print('Error: ' + e)
    print('code: ' + str(proc.returncode))

    return


def get_elevation_at_dist(elevation: DataFrame, dist: float) -> float:
    """

    Parameters
    ----------
    elevation
        dataframe with distance and elevation columns
    dist

    Returns
    -------

    """
    return np.interp(dist, elevation.distance, elevation.elevation)


def get_maxspeed_at_dist(maxspeed: DataFrame, dist: float) -> Union[float, int]:
    """

    Parameters
    ----------
    maxspeed
        dataframe with start_dist, end_dist and maxspeed columns
    dist

    Returns
    -------

    """

    if dist > maxspeed.end_dist.iloc[-1]:
        return maxspeed["maxspeed"].iloc[-1]
    return maxspeed[(maxspeed.start_dist <= dist) & (maxspeed.end_dist >= dist)]["maxspeed"].iloc[0]


def get_electrified_at_dist(electrified: DataFrame, dist: float) -> int:
    """

    Parameters
    ----------
    electrified
        dataframe with start_dist, end_dist and electrified columns
    dist

    Returns
    -------

    """
    if dist > electrified.end_dist.iloc[-1]:
        return electrified["electrified"].iloc[-1]
    return electrified[(electrified.start_dist <= dist) & (electrified.end_dist >= dist)]["electrified"].iloc[0]


def resample_prop(prop_df: DataFrame, distances: np.ndarray, prop: str) -> DataFrame:
    """

    Parameters
    ----------
    prop_df
        either elevation dataframe with "elevation" and "distance" or maxspeed dataframe with "maxspeed", "start_dist",
        "end_dist" or electrified dataframe with "electrified", "start_dist", "end_dist"
    distances
    prop
        either "elevation", "maxpeed" or "electrified"

    Returns
    -------

    """
    resampled_props = []
    for distance in distances:
        resampled_prop = None
        if prop == "maxspeed":
            resampled_prop = get_maxspeed_at_dist(prop_df, distance)
        elif prop == "electrified":
            resampled_prop = get_electrified_at_dist(prop_df, distance)
        elif prop == "elevation":
            resampled_prop = get_elevation_at_dist(prop_df, distance)

        assert resampled_prop is not None

        resampled_props.append(resampled_prop)

    data = {"distance": distances, prop: resampled_props}
    resampled_props = pd.DataFrame(data)

    return resampled_props
