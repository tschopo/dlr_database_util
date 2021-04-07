"""
Functions that interface with the OSM rail database as defined by the import script
"""
import os
import re
import urllib.request
from shutil import which
from typing import Optional, Union, Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
from ElevationSampler import DEM
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
    osm_data = gpd.read_postgis(text(sql), geom_col='geom', con=engine,
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
                  'gauge': np.float64,
                  'traffic_mode': str,  # passenger / mixed / freight
                  'service': str,
                  'usage': str,  # branch / main
                  'voltage': np.float64,
                  'frequency': np.float64,
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


def finalize_osm(osm_data, trip_geom, filter_inactive: bool = True):
    """
    Calculates overlapping segments along the trip geometry. Removes overlapping if they dont have status=active,
    or they are a service track.
    Adds start_point, end_point, end_point_distance, end_point_distance columns.
    Sorts the data after start_point.
    Aligns the geoms so they all point in same direction.

    Parameters
    ----------

    filter_inactive
        if True filters overlapping segments with status != active and service tracks
        if False then all overlapping segments are discarded. This leads to gaps in the data.

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

    osm_data.sort_values(['start_point_distance', 'end_point_distance'], inplace=True)
    osm_data.reset_index(drop=True, inplace=True)

    # get overlapping segments
    # important: data must be sorted after start_distance
    overlapping = []
    for i, row in osm_data.iterrows():
        overlapping_idx = osm_data[i + 1:].index[
            row["end_point_distance"] > osm_data[i + 1:]["start_point_distance"]].tolist()
        overlapping += overlapping_idx

        if len(overlapping_idx) > 0:
            overlapping += [i]
    overlapping = list(set(overlapping))

    if not filter_inactive:
        return osm_data[~osm_data.index.isin(overlapping)]

    # from overlapping filter sections that are not active or are service tracks
    # remove isin(overlapping) and ((status != active) or (service != 'None))

    keep = (~osm_data.index.isin(overlapping)) | ((osm_data.status == "active") & (osm_data.service == "None"))
    discard = (osm_data.index.isin(overlapping)) & ((osm_data.status != "active") | (osm_data.service != "None"))

    assert np.all(keep == ~discard)

    osm_data = osm_data[keep]

    # TODO this way there are tracks lost. add final step: compute missing segments
    # get all osm in missing segment
    # create new osm row where geom is missing segment, and values are computed from the osm segments
    # e.g. max(maxspeeds), tunnel = yes if any yes, bridge = yes if any yes, electrified = yes if any yes ...

    return osm_data


def get_osm_prop(osm_data: GeoDataFrame, prop: str, brunnel_filter_length: float = 10., round_int: bool = True,
                 min_length: Optional[float] = None):
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
        round_int
            return ints not floats for start_dist, end_dist, maxspeed
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
            # if maxspeed not specified take maxspeed forward
            osm_prop_data["maxspeed"] = np.where(np.isnan(osm_prop_data["maxspeed"]), osm_prop_data["maxspeed_forward"],
                                                 osm_prop_data["maxspeed"])
        # filter nans
        osm_prop_data = osm_prop_data[~np.isnan(osm_prop_data.maxspeed)]

    elif prop == "electrified":
        # filter unknown
        osm_prop_data = osm_prop_data[osm_prop_data.electrified != "unknown"]

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

    if prop == "brunnel":
        props = props[props.brunnel == "yes"]

        props["length"] = props["end_dist"] - props["start_dist"]
        props = props[props.length > brunnel_filter_length]

        # props["brunnel"] = np.where(props.bridge == "yes", "bridge", "tunnel")
    elif prop == "electrified":
        props["electrified"] = np.where(props.electrified == "yes", 1, 0)
        props["electrified"] = props["electrified"].astype(int)

    if round_int:
        props["start_dist"] = np.rint(props["start_dist"]).astype(int)
        props["end_dist"] = np.rint(props["end_dist"]).astype(int)

        if prop == "maxspeed":
            props["maxspeed"] = np.rint(props["maxspeed"]).astype(int)

    props.reset_index(drop=True, inplace=True)
    return props


def osm_railways_to_psql(geofabrik_pbf: str, database="liniendatenbank", user="postgres", password=None):
    """
    Warning not tested

    Parameters
    ----------

    Returns
    -------

    """

    # TODO test this

    # download geofabrik data if geo_fabrik_pbf is a url
    if geofabrik_pbf.startswith('http'):
        urllib.request.urlretrieve(geofabrik_pbf)
        geofabrik_pbf = geofabrik_pbf.split('/')[-1]

    # filter geofabrik germany osm data to only include railway data, to speedup import
    # test if osmium is installed
    if which('osmium') is not None:
        os.system('osmium tags-filter -o filtered.osm.pbf '
                  + geofabrik_pbf +
                  ' nwr/railway nwr/disused:railway '
                  'nwr/abandoned:railway nwr/razed:railway nwr/construction:railway nwr/proposed:railway '
                  'nwr/planned:railway n/public_transport=stop_position nwr/public_transport=platform r/route=train '
                  'r/route=tram r/route=light_rail')
        geofabrik_pbf = 'filtered.osm.pbf'

    # osm2pgsql -d liniendatenbank -U postgres -W -O flex -S railways.lua data/germany-railway.osm.pbf
    os.system('osm2pgsql -d '
              + database +
              '-U '
              + user +
              '-W -O flex -S railways.lua '
              + geofabrik_pbf +
              '< '
              + password)
    return


def plot(osm_data: GeoDataFrame, prop: Optional[str] = None, dem: Optional[DEM] = None):


    return
