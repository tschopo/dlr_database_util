"""
Functions that interface with the OSM rail database as defined by the import script
"""
import os
import re
import subprocess
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
from shapely.ops import unary_union
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
                          get_osm_srid: Optional[int] = 25832, get_osm_buffer: float = 10.,
                          filter_difference_length: float = 1., schema: str = "public",
                          table: str = "osm_railways", geom_column: str = "geom") -> GeoDataFrame:
    """
    Get osm data for a LineString. Raises an Exception if no osm data is found.

    Parameters
    ----------
    engine : SqlAlchemy Engine
    linestring : shapely LineString or GeoSeries
    crs : shapely crs string or object
        the crs of the LineString, if linestring is a LineString. If linestring is a GeoSeries the CRS of the GeoSeries
        is taken
    get_osm_srid : int
        The srid which is used internally to get the osm data (the linestring and the osm geometries are converted to
        this srid in the sql query)
    get_osm_buffer : float, default = 10(m)
                Buffer used to get all osm data around the trip that lies in the buffer
    filter_difference_length : float, default = 1
        Filter segments smaller than this length from difference of osm and trip
    schema
    table
    geom_column

    Returns
    -------
        geopandas GeoDataFrame
        in crs of the linestring
    """

    if crs is not None:
        crs = pyproj.CRS(crs)

    if isinstance(linestring, GeoSeries):
        crs = linestring.crs
        linestring_pd = linestring.copy()
        linestring = linestring.iloc[0]
    else:
        linestring_pd = gpd.GeoSeries(linestring, name="geom", crs=crs)

    if crs is None:
        raise Exception("Parameter 'linestring' must be GeoSeries or 'crs' parameter must be given!")

    # no parameterized input for column names --> escape input
    geom_column = re.sub('[^A-Za-z0-9_]+', '', geom_column)
    table = re.sub('[^A-Za-z0-9_]+', '', table)
    schema = re.sub('[^A-Za-z0-9_]+', '', schema)

    sql = """
    SELECT *
    FROM {schema}.{table}
    WHERE ST_intersects(ST_Transform(ST_GeomFromText(:linestring,:linestring_crs),:srid),
        st_buffer(ST_Transform({geom_column},:srid),:intersect_buffer))
    """.format(schema=schema, table=table, geom_column=geom_column)

    with engine.connect() as connection:
        osm_data = gpd.read_postgis(text(sql), geom_col='geom', con=connection,
                                    params={"linestring": linestring.wkt,
                                            "linestring_crs": crs.to_epsg(),
                                            "srid": get_osm_srid,
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

    if osm_data.shape[0] == 0:
        raise Exception("No OSM data found for the given shape!")

    # convert the reference system so that they match
    trip_geom = linestring_pd
    osm_data = osm_data.to_crs(crs)

    assert trip_geom.crs.equals(osm_data.crs)
    # print(osm_data.crs.to_epsg())
    # print(trip_geom.crs.to_epsg())

    osm_data['start_point'] = osm_data.apply(lambda r: Point(r['geom'].coords[0]), axis=1)
    osm_data['end_point'] = osm_data.apply(lambda r: Point(r['geom'].coords[-1]), axis=1)

    osm_data = combine_osm_pipeline(osm_data, trip_geom, filter_difference_length=filter_difference_length)

    osm_data = osm_data.drop_duplicates(subset=["way_id"])

    osm_data = add_osm_to_geom(osm_data, trip_geom)

    return osm_data


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


def add_osm_to_geom(osm_data: GeoDataFrame, trip_geom: GeoSeries):
    """
    Aligns the geoms so they all point in same direction.
    Adds the osm properties to the trip_geom.
    Adds start_point, end_point, start_point_distance, end_point_distance columns.
    Sorts the data after start_point.

    Parameters
    ----------

    osm_data
    trip_geom

    """

    # calculate distances
    osm_data = calc_distances(osm_data, trip_geom)

    # make linestrings all same direction
    # flip if wrong dir
    osm_data["geom"] = osm_data.apply(
        lambda r: LineString(list(r['geom'].coords)[::-1]) if r['start_point_distance'] > r[
            'end_point_distance'] else r['geom'], axis=1)

    # take into account maxspeed forward / backward
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
    # identify overlapping segments in trip geom.
    # in the overlapping segments duplicate the osm data with flipped geometry
    # get the segments where trip_geom crosses itself
    # this is kind of hacky, would be better to split trip_geom at sharp turns and apply get_osm for each slitted
    # segment.
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

        # flip
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

    # --------------------

    # split trip_geom at every start and endpoint in osm_data
    # for each split trip segment get all osm data that overlaps and assign values from these osm segments

    cut_distances = np.append(osm_data.start_point_distance.values, osm_data.end_point_distance.values)

    linestring = trip_geom.iloc[0]

    # add linestring end to cut_distances and 0
    cut_distances = np.append(cut_distances, [0, linestring.length])

    # remove cut distances that are larger than trip
    cut_distances = cut_distances[cut_distances <= linestring.length]

    # sort
    cut_distances = np.unique(np.sort(cut_distances))

    # cut trip geom into segments
    segments = []
    assert cut_distances[-1] >= linestring.length
    rest = linestring
    for cut_distance in cut_distances[::-1]:
        if rest is not None:
            rest, segment = _cut_line_at_distance(rest, cut_distance)
            if segment is not None:
                segments.append(segment)

    start_distances = cut_distances[:-1]
    end_distances = cut_distances[1:]

    # for each segment get osm data that lies in segment
    assert (len(segments) == len(start_distances) == len(end_distances))

    data = {'way_id': [],
            'type': [],
            'status': [],
            'electrified': [],
            'electrification': [],
            'maxspeed': [],
            'maxspeed_forward': [],
            'maxspeed_backward': [],
            'bridge': [],
            'bridge_type': [],
            'tunnel': [],
            'tunnel_type': [],
            'embankment': [],
            'cutting': [],
            'ref': [],
            'gauge': [],
            'traffic_mode': [],
            'service': [],
            'usage': [],
            'voltage': [],
            'frequency': [],
            'tags': [],
            'geom': [],
            'start_point': [],
            'end_point': [],
            'start_point_distance': [],
            'end_point_distance': [],
            'way_ids': []}

    for i, segment in enumerate(segments):
        # get osm_data that overlaps
        overlapping_osm = osm_data[
            (osm_data.start_point_distance < end_distances[i]) & (osm_data.end_point_distance > start_distances[i])]
        if overlapping_osm.shape[0] == 0:
            continue

        longest_segment_ix = np.argmin(overlapping_osm.length)

        # set maxspeed to highest value
        maxspeeds = overlapping_osm["maxspeed"].values
        if not np.isnan(maxspeeds).all():
            maxspeed = np.nanmax(maxspeeds)
        else:
            maxspeed = np.nan

        # set maxspeed_forward to highest value
        maxspeeds_forward = overlapping_osm["maxspeed_forward"].values
        if not np.isnan(maxspeeds_forward).all():
            maxspeed_forward = np.nanmax(maxspeeds_forward)
        else:
            maxspeed_forward = np.nan

        # set maxspeed_backward to highest value
        maxspeeds_backward = overlapping_osm["maxspeed_backward"].values
        if not np.isnan(maxspeeds_backward).all():
            maxspeed_backward = np.nanmax(maxspeeds_backward)
        else:
            maxspeed_backward = np.nan

        # set electrified to yes if there is a yes
        electrifieds = overlapping_osm["electrified"].values
        if "yes" in electrifieds:
            electrified = "yes"
            electrification = overlapping_osm[overlapping_osm.electrified == "yes"]["electrification"].iloc[0]
            voltage = overlapping_osm[overlapping_osm.electrified == "yes"]["voltage"].iloc[0]
            frequency = overlapping_osm[overlapping_osm.electrified == "yes"]["frequency"].iloc[0]

        elif "no" in electrifieds:
            electrified = "no"
            electrification = "none"
            voltage = None
            frequency = None
        else:
            electrified = "unknown"
            electrification = "unknown"
            voltage = None
            frequency = None

        # set tunnel to yes if there is a yes
        tunnels = overlapping_osm["tunnel"].values
        if "yes" in tunnels:
            tunnel = "yes"
            tunnel_type = overlapping_osm[overlapping_osm.tunnel == "yes"]["tunnel_type"].iloc[0]
        else:
            tunnel = "no"
            tunnel_type = None

        # set bridge to yes if there is a yes
        bridges = overlapping_osm["bridge"].values
        if "yes" in bridges:
            bridge = "yes"
            bridge_type = overlapping_osm[overlapping_osm.bridge == "yes"]["bridge_type"].iloc[0]
        else:
            bridge = "no"
            bridge_type = None

        embankments = overlapping_osm["embankment"].values
        if "yes" in embankments:
            embankment = "yes"
        else:
            embankment = "no"

        cuttings = overlapping_osm["cutting"].values
        if "yes" in cuttings:
            cutting = "yes"
        else:
            cutting = "no"

        types = overlapping_osm["type"].values
        if "rail" in types:
            _type = "rail"
        else:
            _type = overlapping_osm.iloc[longest_segment_ix]['type']

        status = overlapping_osm["status"].values
        if "active" in status:
            status = "active"
        else:
            status = overlapping_osm.iloc[longest_segment_ix]['status']

        usages = overlapping_osm["usage"].values
        if "main" in usages:
            usage = "main"
        elif "branch" in usages:
            usage = "branch"
        elif "tourism" in usages:
            usage = "tourism"
        elif "industrial" in usages:
            usage = "industrial"
        else:
            usage = overlapping_osm.iloc[longest_segment_ix]['usage']

        gauges = overlapping_osm["gauge"].values
        if 1435 in gauges or '1435' in gauges:
            gauge = 1435
        elif 1000 in gauges or '1000' in gauges:
            gauge = 1000
        else:
            gauge = overlapping_osm.iloc[longest_segment_ix]['gauge']

        ref = overlapping_osm.iloc[longest_segment_ix]['ref']
        if ref is None:
            refs = overlapping_osm["ref"].values
            if refs[~pd.isnull(refs)].shape[0] > 0:
                ref = refs[0]

        traffic_mode = overlapping_osm.iloc[longest_segment_ix]['traffic_mode']
        tags = overlapping_osm.iloc[longest_segment_ix]['tags']
        service = overlapping_osm.iloc[longest_segment_ix]['service']
        way_id = overlapping_osm.iloc[longest_segment_ix]['way_id']

        geom = segment
        start_point = Point(segment.coords[0])
        end_point = Point(segment.coords[-1])

        start_point_distance = start_distances[i]
        end_point_distance = end_distances[i]

        # add column way_ids where all way_ids of overlapping
        way_ids = ';'.join([str(s) for s in overlapping_osm.way_id.values])

        data['way_id'].append(way_id)
        data['type'].append(_type)
        data['status'].append(status)
        data['electrified'].append(electrified)
        data['electrification'].append(electrification)
        data['maxspeed'].append(maxspeed)
        data['maxspeed_forward'].append(maxspeed_forward)
        data['maxspeed_backward'].append(maxspeed_backward)
        data['bridge'].append(bridge)
        data['bridge_type'].append(bridge_type)
        data['tunnel'].append(tunnel)
        data['tunnel_type'].append(tunnel_type)
        data['embankment'].append(embankment)
        data['cutting'].append(cutting)
        data['ref'].append(ref)
        data['gauge'].append(gauge)
        data['traffic_mode'].append(traffic_mode)
        data['service'].append(service)
        data['usage'].append(usage)
        data['voltage'].append(voltage)
        data['frequency'].append(frequency)
        data['tags'].append(tags)
        data['geom'].append(geom)
        data['start_point'].append(start_point)
        data['end_point'].append(end_point)
        data['start_point_distance'].append(start_point_distance)
        data['end_point_distance'].append(end_point_distance)
        data['way_ids'].append(way_ids)

    osm_data = gpd.GeoDataFrame(data, geometry='geom', crs=osm_data.crs)

    return osm_data


def get_osm_prop(osm_data: GeoDataFrame, prop: str, brunnel_filter_length: float = 10., round_int: bool = True,
                 train_length: Optional[float] = 150., maxspeed_if_all_null: float = 120.,
                 maxspeed_null_segment_length=1000., maxspeed_null_max_frac: float = 0.5,
                 maxspeed_min_if_null: float = 60.,
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
            The maxspeed that is set if there are 0 maxspeeds present or for if all nullsegments lengths sum up
             to more than maxspeed_null_max_frac * trip_length
        maxspeed_null_segment_length : float
            For Segments of nan values longer than this, the maxspeed is set to the median of the trip maxspeed.
        maxspeed_null_max_frac: float
            between 0 and 1. if the null segment lengths sum up to a length larger than maxspeed_null_max_frac *
            trip_length, then the null segment ist set to maxspeed_if_all_null
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

        total_null_length = 0
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
                    total_null_length += current_segment.length
                    segments.append(current_segment)

                # close the segment
                segment_open = False

        # close segment if still open
        if segment_open:
            current_segment.end = segment_end_candidate
            current_segment.length = current_segment.end - current_segment.start
            total_null_length += current_segment.length
            segments.append(current_segment)

        # go through segments. if a segment is long then set all maxspeeds to median of trip
        median_maxspeed = spatial_median_osm(osm_prop_data)

        # if the median is very low, set it to maxspeed_if_all_null
        if median_maxspeed < maxspeed_min_if_null:
            median_maxspeed = maxspeed_if_all_null

        # get the total length of nans
        # if total length is above maxspeed_if_null_frac*trip_length then set all nans to maxspeed if all null
        set_null_to_default_maxspeed = False
        if trip_length is not None and total_null_length > maxspeed_null_max_frac * trip_length:
            set_null_to_default_maxspeed = True

        for segment in segments:
            if set_null_to_default_maxspeed:
                osm_prop_data.loc[segment.members, prop] = maxspeed_if_all_null
            elif segment.length > maxspeed_null_segment_length:
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
            # calculate missing length
            # if missing length is large, then add segment with default maxspeed.
            # this prevents maxspeed being to low if trip goes outside of germany and last maxspeed is low.
            if prop == "maxspeed" and \
                    (trip_length - props.iloc[-1, props.columns.get_loc('end_dist')]) > maxspeed_null_segment_length:
                new_row = {'maxspeed': maxspeed_if_all_null,
                           'start_dist': props.iloc[-1, props.columns.get_loc('end_dist')],
                           'end_dist': trip_length}
                props = props.append(new_row, ignore_index=True)

            props.iloc[-1, props.columns.get_loc('end_dist')] = trip_length
        if harmonize_end_dists:
            # make sure all start at 0
            props.iloc[0, props.columns.get_loc('start_dist')] = 0

            # make sure all segments are connected by setting the end dist do the start_dist of the next segment
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

            # the rounding can lead to segments of length 0. filter these segments
            props = props[props["start_dist"] != props["end_dist"]]

        if prop == "maxspeed":
            props["maxspeed"] = np.rint(props["maxspeed"]).astype(int)

    props.reset_index(drop=True, inplace=True)
    return props


def spatial_median_osm(osm_data, prop="maxspeed"):
    """
    calculate the weighted median. repeat each value by length.

    Parameters
    ----------

    Returns
    -------

    """

    return spatial_median(osm_data[prop].values, osm_data.length.values)


def spatial_median(vals, lengths):
    """
    calculate the weighted median. repeat each value by length.

    Parameters
    ----------
    vals : List
        The values
    lengths : List
        The lengths

    Returns
    -------

    """

    if len(vals) != len(lengths):
        raise Exception("Input Arrays must be same length")
    mult_vals = []
    for i in range(len(vals)):
        mult_vals = mult_vals + ([vals[i]] * int(lengths[i]))

    mult_vals = np.array(mult_vals)
    median = np.nanmedian(mult_vals)

    return median


def osm_railways_to_psql(geofabrik_pbf_folder: str, geofabrik_pbf: str, database="liniendatenbank", user="postgres",
                         password=None, osmium_filter=True):
    """

    Parameters
    ----------

    Returns
    -------

    """

    # filter geofabrik germany osm data to only include railway data, to speedup import
    # test if osmium is installed
    if which('osmium') is not None and osmium_filter:
        # todo use subprocess here instead os.system
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
                             geofabrik_pbf_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)
    proc.stdin.write(password+'\n')
    proc.stdin.flush()

    e, o = proc.communicate()

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


def _cut_line_at_distance(line, distance):
    # Cuts a line in two at a distance from its starting point
    if distance <= 0.0:
        return None, LineString(line)
    if distance >= line.length:
        return LineString(line), None
    coords = list(line.coords)
    for i, p in enumerate(coords):
        point_distance = line.project(Point(p))
        if point_distance == distance:
            return [
                LineString(coords[:i+1]),
                LineString(coords[i:])]
        if point_distance > distance:
            cp = line.interpolate(distance)
            return LineString(coords[:i] + [(cp.x, cp.y)]), LineString([(cp.x, cp.y)] + coords[i:])
