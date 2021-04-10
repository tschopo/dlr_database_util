"""
Functions that interface with the OSM rail database as defined by the import script
"""
import os
import re
import urllib.request
from shutil import which
from typing import Optional, Union, Any, List

import altair as alt
import folium
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
                 maxspeed_spikes_min_length: Optional[float] = 1000., maxspeed_if_all_null: float = 100.,
                 maxspeed_null_segment_length=1000.):
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
        maxspeed_spikes_min_length : float or None
            The minimum length of maxspeed spikes. Should be larger than train length.
        maxspeed_if_all_null : float
            The maxspead that is set if there are 0 maxspeeds present
        maxspeed_null_segment_length : float
            For Segments of nan values longer than this, the maxspeed is set to the median of the trip maxspeed.

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
            # if maxspeed and maxspeed forward not specified take maxspeed backward
            osm_prop_data[prop] = np.where(np.isnan(osm_prop_data[prop]), osm_prop_data["maxspeed_forward"],
                                           osm_prop_data[prop])

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

    if prop == "brunnel":
        props = props[props.brunnel == "yes"]

        props["length"] = props["end_dist"] - props["start_dist"]
        props = props[props.length > brunnel_filter_length]

        # props["brunnel"] = np.where(props.bridge == "yes", "bridge", "tunnel")
    elif prop == "electrified":

        # set unknown to not electrified
        props["electrified"] = np.where(props.electrified == "yes", 1, 0)
        props["electrified"] = props["electrified"].astype(int)

    elif prop == "maxspeed":

        if maxspeed_spikes_min_length is not None:

            # filter maxspeed segments that are small and are "spikes" e.g. that are higher than their neighbours.

            # removing the spikes can lead to more spikes, so we have to repeat until there are no more spikes
            changed = True
            while changed:

                changed = False
                drop_idx = []
                for i in range(1, props.shape[0] - 1):

                    # segment length is calculated only from the start dists
                    segment_length = props.iloc[i + 1]["start_dist"] - props.iloc[i]["start_dist"]

                    if segment_length < maxspeed_spikes_min_length \
                            and props.iloc[i - 1][prop] < props.iloc[i][prop] > props.iloc[i + 1][prop]:
                        # remove the row and extend the previous segment
                        drop_idx.append(i)
                        props.iloc[i - 1]["end_dist"] += segment_length
                        changed = True

                props.drop(drop_idx, inplace=True)
                props.reset_index(drop=True, inplace=True)

    if round_int:
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


# TODO put plots in separate file
def plot_osm(osm_data: GeoDataFrame, prop: Optional[str] = None, dem: Optional[DEM] = None):
    """
    Plot osm data on map.

    Parameters
    ----------
    osm_data
    prop
    dem

    Returns
    -------

    """
    osm_data_map = osm_data.to_crs(4326)

    if prop == "maxspeed":

        # pep doesnt like lambdas
        def colormap(x):
            return '#aaa'

        if not osm_data_map['maxspeed'].isna().all():
            colormap = folium.LinearColormap(colors=["red", "orange", "yellow", "green"],
                                             vmin=osm_data_map['maxspeed'].min(),
                                             vmax=osm_data_map['maxspeed'].max()).to_step(
                n=len(osm_data_map["maxspeed"].unique()))
        osm_data_map['maxspeed'] = osm_data_map['maxspeed'].fillna(-9999)
    elif prop == "electrified":

        def colormap(x):
            if x == 0:
                return '#d62728'
            elif x == 1:
                return '#2ca02c'
            else:
                return '#aaa'

        osm_data_map['electrified'] = np.where(osm_data_map['electrified'] == 'yes', 1, osm_data_map['electrified'])
        osm_data_map['electrified'] = np.where(osm_data_map['electrified'] == 'no', 0, osm_data_map['electrified'])
        osm_data_map['electrified'] = np.where((osm_data_map['electrified'] == 1) | (osm_data_map['electrified'] == 0),
                                               osm_data_map['electrified'], -9999)

    osm_json = osm_data_map[
        ["electrified", "maxspeed", "maxspeed_forward", "maxspeed_backward", "bridge", "tunnel", "geom",
         "start_point_distance", "end_point_distance"]].to_json(na='keep')

    m = folium.Map(location=[osm_data_map.total_bounds[[1, 3]].mean(), osm_data_map.total_bounds[[0, 2]].mean()],
                   # tiles='Stamen Terrain',
                   zoom_start=9)

    folium.GeoJson(osm_json, name="geojson",
                   style_function=lambda x: {
                       'color': colormap(x['properties'][prop]) if x['properties'][prop] >= 0 else '#aaa',
                       'weight': 2.5
                   },
                   tooltip=folium.features.GeoJsonTooltip(
                       fields=['maxspeed', 'maxspeed_forward', 'maxspeed_backward', 'electrified',
                               'start_point_distance', 'end_point_distance'],
                       # aliases=['Max Speed', 'Electrified'],
                       labels=True,
                       sticky=True,
                       toLocaleString=True)
                   ).add_to(m)

    return m


def plot_trip_props(maxspeed, electrified, elevation, trip_title, color_monotone=None):
    # ideas: add stations to elevation and maxspeed (points on the line)
    # add timetable plot (like electrified, with station names and color is the duration between the stations?)
    # interactivity: scrolling, zooming, highlighting
    # add geometry, linked circle of cursor position on all plots

    maxspeed_color = None
    electrified_color = None
    not_electrified_color = None
    elevation_color = None
    if color_monotone is not None:
        maxspeed_color = color_monotone
        electrified_color = color_monotone
        not_electrified_color = '#ccc'
        elevation_color = color_monotone

    chart_maxspeed = plot_maxspeeds(maxspeed, color=maxspeed_color)
    chart_electrified = plot_electrified(electrified, electrified_color=electrified_color,
                                         not_electrified_color=not_electrified_color)
    chart_elevation = plot_elevation(elevation, color=elevation_color)

    chart_maxspeed = chart_maxspeed \
        .encode(
        x=alt.X('distance:Q', axis=alt.Axis(labels=False, ticks=False, tickRound=True),
                title='',
                scale=alt.Scale(domain=(0, max(chart_maxspeed.data.distance)), clamp=True, nice=False))) \
        .properties(width=1000, height=100)

    chart_electrified = chart_electrified.encode(x=alt.X('distance:Q', axis=None,
                                                         title='',
                                                         scale=alt.Scale(domain=(0, max(electrified.end_dist)),
                                                                         clamp=True,
                                                                         nice=False))).properties(width=1000, height=6)

    chart_elevation = chart_elevation.properties(width=1000, height=100)

    chart = chart_electrified & chart_maxspeed & chart_elevation
    chart = chart.properties(title=trip_title).configure_title(
        align='center',
        anchor='middle',
        offset=30)

    return chart


def plot_maxspeeds(maxspeed: DataFrame, color=None) -> alt.Chart:
    """

    Parameters
    ----------
    maxspeed : DataFrame
        cols start_dist, maxspeed must be present

    Returns
    -------

    """
    maxspeeds = []
    start_dists = []

    # plot horizontal lines: duplicate the values and add start and end for the x values
    for i in range(maxspeed.shape[0] - 1):
        maxspeeds.append(maxspeed.iloc[i]["maxspeed"])
        maxspeeds.append(maxspeed.iloc[i]["maxspeed"])

        start_dists.append(maxspeed.iloc[i]["start_dist"])
        start_dists.append(maxspeed.iloc[i + 1]["start_dist"])

    # for last value add end_dist as end
    maxspeeds.append(maxspeed.iloc[-1]["maxspeed"])
    start_dists.append(maxspeed.iloc[-1]["start_dist"])
    maxspeeds.append(maxspeed.iloc[-1]["maxspeed"])
    start_dists.append(maxspeed.iloc[-1]["end_dist"])

    maxspeed_chart_data = pd.DataFrame({"maxspeed": maxspeeds, "distance": start_dists})

    # ff7f00
    if color is None:
        color = '#377eb8'
    chart = alt.Chart(maxspeed_chart_data) \
        .mark_line(color=color) \
        .encode(x=alt.X('distance:Q',
                        scale=alt.Scale(
                            domain=(0, max(maxspeed_chart_data.distance)),
                            clamp=True,
                            nice=False),
                        axis=alt.Axis(format="~s")),
                y=alt.Y('maxspeed:Q',
                        scale=alt.Scale(domain=(
                            0, 150)))
                )

    return chart


def plot_elevation(elevation: DataFrame, color: Optional[str] = None) -> alt.Chart:
    """

    Parameters
    ----------
    elevation : DataFrame
        columns 'distance', 'elevation' and 'ele_smoothed' must be present

    Returns
    -------

    """
    if color is None:
        color = '#a65628'

    chart = alt.Chart(elevation) \
                .mark_line(color='#ccc') \
                .encode(
        x=alt.X('distance:Q',
                axis=alt.Axis(format="~s"),
                scale=alt.Scale(
                    domain=(0, max(elevation.distance)),
                    clamp=True,
                    nice=False)),
        y=alt.Y('elevation:Q',
                title='elevation',
                scale=alt.Scale(
                    domain=(max(0, min(elevation.elevation) - 50), max(elevation.elevation) * 0.85)))) \
            + alt.Chart(elevation).mark_line(color=color).encode(x='distance:Q', y='ele_smoothed:Q')

    return chart


def plot_electrified(electrified: DataFrame, electrified_color: Optional[str] = None,
                     not_electrified_color: Optional[str] = None):
    data = {'y': ['electrified'] * electrified.shape[0],
            'electrified': np.where(electrified.electrified.values == 1, 'yes', 'no'),
            'distance': electrified.end_dist - electrified.start_dist, 'start_dist': electrified.start_dist}
    df = pd.DataFrame(data)

    if not_electrified_color is None:
        not_electrified_color = '#ccc'  # '#e41a1c'
    if electrified_color is None:
        electrified_color = '#4daf4a'

    chart = alt.Chart(df).mark_bar().encode(
        y=alt.Y('y:N', axis=alt.Axis(title='', labels=False, ticks=False)),
        x=alt.X('distance:Q',
                axis=alt.Axis(format="~s"),
                scale=alt.Scale(
                    domain=(0, max(electrified.end_dist)),
                    clamp=True,
                    nice=False)),
        color=alt.Color('electrified:N',
                        scale=alt.Scale(
                            domain=['yes', 'no'],
                            range=[electrified_color, not_electrified_color])),
        order=alt.Order(
            # Sort the segments of the bars by this field
            'start_dist',
            sort='ascending'
        )
    )
    return chart
