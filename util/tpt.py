import re
import numpy as np
import pandas as pd
from sqlalchemy import text
from .osm import get_osm_prop
from .sql import sql_get_osm, sql_get_trip_geom
from ElevationSampler import *


def process_ele(elevation, distances, brunnels, first_sample_distance=10, end_sample_distance=100,
                construct_brunnels=False, max_bridge_length=300,
                max_tunnel_length=300, construct_brunnel_thresh=3, adjust_window_size=12, std_thresh=3, sub_factor=3,
                clip=20, smooth_window_size=301, poly_order=3, degrees=False, smooth_after_resampling=True,
                window_size_2=5, poly_order_2=1, mode="nearest", resample_first=True, output_all=False,
                adjust_forest_height=True, drop_last_incl_if_high=True, last_incl_thresh=0, last_incl_dist=100):
    distances_10, elevation_10 = elevation, distances

    if resample_first:
        # we need evenly spaced points for brunnels
        distances_10, elevation_10 = ElevationSampler.resample_ele(elevation, distances, first_sample_distance)

    ele_brunnel = ElevationSampler.interpolate_brunnels(elevation_10, distances_10, brunnels,
                                                        distance_delta=first_sample_distance,
                                                        construct_brunnels=construct_brunnels,
                                                        max_bridge_length=max_bridge_length,
                                                        max_tunnel_length=max_tunnel_length,
                                                        construct_brunnel_thresh=construct_brunnel_thresh)
    ele_adjusted = ele_brunnel
    if adjust_forest_height:
        ele_adjusted = ElevationSampler.adjust_forest_height(ele_brunnel, window_size=adjust_window_size,
                                                             std_thresh=std_thresh, sub_factor=sub_factor, clip=clip)

    ele_smoothed = ElevationSampler.smooth_ele(ele_adjusted, window_size=smooth_window_size, poly_order=poly_order,
                                               mode=mode)

    distances_100, elevation_100 = ElevationSampler.resample_ele(ele_smoothed, distances_10, end_sample_distance)

    if smooth_after_resampling:
        elevation_100 = ElevationSampler.smooth_ele(elevation_100, window_size=window_size_2, poly_order=poly_order_2,
                                                    mode=mode)
    if output_all:
        return distances_10, elevation_10, ele_brunnel, ele_adjusted, ele_smoothed, distances_100, elevation_100

    incl_100 = ElevationSampler.ele_to_incl(elevation_100, distances_100, degrees=degrees)

    # set last incl to 0 in some cases
    if drop_last_incl_if_high \
            and incl_100[-1] > last_incl_thresh \
            and distances_100[-1] - distances_100[-2] < last_incl_dist:
        incl_100[-1] = 0

    distances_100 = distances_100[:-1]

    data = {"start_dist": distances_100, "incl": incl_100}

    return pd.DataFrame.from_dict(data)


def sql_get_inclination(trip_id, osm_data, elevation_sampler, engine, first_sample_distance=10, end_sample_distance=100,
                        brunnel_filter_length=10, interpolated=True, construct_brunnels=True, max_bridge_length=300,
                        max_tunnel_length=300, construct_brunnel_thresh=3, adjust_window_size=12, std_thresh=3,
                        sub_factor=3,
                        clip=20, smooth_window_size=301, poly_order=3, degrees=False, smooth_after_resampling=True,
                        window_size_2=5, poly_order_2=1, mode="nearest", output_all=False):
    brunnels = get_osm_prop(osm_data, "brunnel", brunnel_filter_length=brunnel_filter_length)

    trip_geom = sql_get_trip_geom(trip_id, engine, elevation_sampler.dem.crs)

    x_coords, y_coords, distances, elevation = elevation_sampler.elevation_profile(trip_geom,
                                                                                   distance=first_sample_distance,
                                                                                   interpolated=interpolated)

    return process_ele(elevation, distances, brunnels, first_sample_distance=first_sample_distance,
                       end_sample_distance=end_sample_distance,
                       construct_brunnels=construct_brunnels, max_bridge_length=max_bridge_length,
                       max_tunnel_length=max_tunnel_length, construct_brunnel_thresh=construct_brunnel_thresh,
                       adjust_window_size=adjust_window_size, std_thresh=std_thresh, sub_factor=sub_factor,
                       clip=clip, smooth_window_size=smooth_window_size, poly_order=poly_order, degrees=degrees,
                       smooth_after_resampling=smooth_after_resampling,
                       window_size_2=window_size_2, poly_order_2=poly_order_2, mode=mode, resample_first=False,
                       output_all=output_all)


def sql_get_timetable(trip_id, engine, min_stop_duration=30, round_int=True):
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

    return time_table.trip_headsign.iloc[0], time_table[["dist", "stop_name", "stop_duration", "driving_time"]]


def flip_trip(end_dists, parameter, invert_para=False):
    """
    Calculate the distances and parameter from the other direction.

    Parameters
    ----------
    end_dists : Numpy Array
    parameter : Numpy Array
    invert_para : bool
        Take parameter * -1

    Returns
    -------
    Numpy Array
        The fliped distances and parameter
    """

    # calculate start dists from the end distances
    distances = end_dists[::-1]
    distances = distances[0] - distances
    parameter = parameter[::-1]

    if invert_para:
        parameter = parameter * -1

    return distances, parameter


def calc_end_dists(start_dists, trip_length):
    """

    Parameters
    ----------
    start_dists : 1d numpy array of floats
    trip_length : float

    Returns
    -------
        1d numpy array of floats
            The end distances
    """
    # end dist is always the start dist of next element. last element end dist is trip length
    end_dists = start_dists[1:].copy()
    end_dists = np.append(end_dists, trip_length)

    return end_dists


def create_umlauf(parameter_df, trip_count, parameter_column, start_dists_column="start_dist",
                  end_dists_column="end_dist", flip_first=False, drop_dups=True):
    """
    For the given distances and parameter arrays returns the computed "umlauf" arrays.

    Parameters
    ----------

    parameter_df : pandas dataframe
        with parameter_column, start_dists_column and end_dists_column
    trip_count : int
    parameter_column : str
    start_dists_column : str
    end_dists_column : str
    flip_first : bool
    drop_dups : bool

    Returns
    -------
    pandas DataFrame
        The computed umlauf. Dataframe with start_dists_column and parameter columns.
    """

    invert_para = False

    if parameter_column == "incl":
        invert_para = True
        if end_dists_column not in parameter_df.columns:
            end_dists_column = start_dists_column

    distances = parameter_df[start_dists_column].values

    end_dists = parameter_df[end_dists_column].values
    parameter = parameter_df[parameter_column].values

    trip_length = end_dists[-1]

    if flip_first:
        distances, parameter = flip_trip(end_dists, parameter, invert_para=invert_para)

    res_dists = []
    res_para = []
    start_dist = 0
    for _ in range(trip_count):
        distances = distances + start_dist
        res_dists.append(distances)
        res_para.append(parameter)

        # add trip length
        start_dist += trip_length

        # flip the trip
        distances, parameter = flip_trip(end_dists, parameter, invert_para=invert_para)

    res_dists = np.concatenate(res_dists, axis=None)
    res_para = np.concatenate(res_para, axis=None)

    data = {start_dists_column: res_dists, parameter_column: res_para}
    res = pd.DataFrame(data)

    if drop_dups:
        res = res.drop_duplicates(ignore_index=True)

        drop_idx = []
        for i in range(1, res.shape[0]):
            if (res.iloc[i][parameter_column]) == (res.iloc[i - 1][parameter_column]):
                drop_idx.append(i)
        res = res.drop(drop_idx)
        res = res.reset_index(drop=True)

    return res


def trip_title_to_filename(title):
    special_char_map = {ord('ä'): 'ae', ord('ü'): 'ue', ord('ö'): 'oe', ord('ß'): 'ss', ord('('): '', ord(')'): '',
                        ord(' '): '_'}

    filename = title.lower()
    filename = filename.replace(" - ", " ")
    filename = filename.translate(special_char_map)
    filename = re.sub('[^a-z0-9_]+', '', filename)

    return filename


def write_input_sheet(trip_title, timetable, electrification, maxspeed, inclination, params={}):
    """

    Parameters
    ----------
    trip_title
    timetable
    electrification
    maxspeed
    inclination
    constants : dict
        sheet1 : "Timetable and limits",
        sheet2 : "Gradients and curves",
        sheet3 : "TPT requirements",
        accelerationLimits : 1.2
        decelerationLimits : 0.9
        gravity_acceleration : 9.81
        time_overhead : 0.05
        gradient_interpolation_binary : 0
        effort_in_curve : 8
        resistances_kp : "normal"

    Returns
    -------

    """

    default_params = {
        "sheet1": "Timetable and limits",
        "sheet2": "Gradients and curves",
        "sheet3": "TPT requirements",
        "accelerationLimits": 1.2,
        "decelerationLimits": 0.9,
        "gravity_acceleration": 9.81,
        "time_overhead": 0.05,
        "gradient_interpolation_binary": 0,
        "effort_in_curve": 8,
        "resistances_kp": "normal"
    }

    default_params.update(params)

    filename = trip_title_to_filename(trip_title)

    writer = pd.ExcelWriter(filename + '.xlsx', engine='xlsxwriter')

    # we first need to write a dataframe, to get the sheets later

    # timetable
    timetable.to_excel(writer, sheet_name=default_params['sheet1'], index=False, header=False, startrow=3)

    workbook = writer.book
    worksheet = writer.sheets[default_params['sheet1']]

    # timetable end
    current_row = timetable.shape[0] + 3
    worksheet.write(current_row, 0, "]end")
    current_row += 1

    # 2 empty rows
    current_row += 2

    # title
    worksheet.write('A1', ']title')
    worksheet.write('B1', trip_title)

    # timetable header
    worksheet.write(2, 0, "]timetable")
    worksheet.write(2, 1, "Station Name")
    worksheet.write(2, 2, "Stop time at station [s]")
    worksheet.write(2, 3, "Driving time to next station [s]")

    # electrification header
    worksheet.write(current_row, 0, "]electrification")
    worksheet.write(current_row, 1, "binary")
    current_row += 1

    # electrification
    electrification[["start_dist", "electrified"]].to_excel(writer, sheet_name=default_params['sheet1'], index=False,
                                                            header=False, startrow=current_row)

    current_row += electrification.shape[0]

    # ectrification end
    worksheet.write(current_row, 0, "]end")
    current_row += 1

    # empty row
    current_row += 1

    # maxspeed header
    worksheet.write(current_row, 0, "]speedLimits")
    worksheet.write(current_row, 1, "v [km/h]")
    current_row += 1

    # maxspeed
    maxspeed[["start_dist", "maxspeed"]].to_excel(writer, sheet_name=default_params['sheet1'], index=False,
                                                  header=False,
                                                  startrow=current_row)

    current_row += maxspeed.shape[0]

    # maxspeed end
    worksheet.write(current_row, 0, "]end")
    current_row += 1

    # 2 empty rows
    current_row += 2

    worksheet.write(current_row, 0, "]accelerationLimits")
    worksheet.write(current_row, 1, "a [m/s^2]")
    current_row += 1

    worksheet.write(current_row, 0, 0)
    worksheet.write(current_row, 1, default_params['accelerationLimits'])
    current_row += 1

    worksheet.write(current_row, 0, "]end")
    current_row += 1

    current_row += 1

    worksheet.write(current_row, 0, "]decelerationLimits")
    worksheet.write(current_row, 1, "a [m/s^2]")
    current_row += 1

    worksheet.write(current_row, 0, 0)
    worksheet.write(current_row, 1, default_params['decelerationLimits'])
    current_row += 1

    worksheet.write(current_row, 0, "]end")

    # column width
    worksheet.set_column(0, 3, 25)

    # new worksheet
    inclination.to_excel(writer, sheet_name=default_params['sheet2'], index=False, header=False, startrow=3)

    worksheet = writer.sheets[default_params['sheet2']]

    # title
    worksheet.write('A1', "]gravity_acceleration(m/s2)")
    worksheet.write('B1', default_params['gravity_acceleration'])

    # inclination header
    worksheet.write(2, 0, "]gradients")
    worksheet.write(2, 1, "Gradient [‰]")

    # inclination end
    current_row = inclination.shape[0] + 3
    worksheet.write(current_row, 0, "]end")
    current_row += 1

    # 2 empty rows
    current_row += 2

    worksheet.write(current_row, 0, "]curves")
    worksheet.write(current_row, 1, "Radius [m]")
    current_row += 1

    worksheet.write(current_row, 0, "]end")

    # column width
    worksheet.set_column(0, 3, 25)

    worksheet = workbook.add_worksheet(default_params['sheet3'])

    worksheet.write('A1', "]time_overhead")
    worksheet.write('B1', default_params['time_overhead'])

    worksheet.write('A4', "]gradient_interpolation_binary")
    worksheet.write('B4', default_params['gradient_interpolation_binary'])

    worksheet.write('A6', "]effort_in_curve(N.m/kg)")
    worksheet.write('B6', default_params['effort_in_curve'])

    worksheet.write('A9', "]resistances_kp(m)_name(word)")

    worksheet.write('A10', 0)
    worksheet.write('B10', default_params['resistances_kp'])

    worksheet.write('A11', "]end")

    worksheet.set_column(0, 3, 25)

    writer.save()


def get_trip_data(trip_id, engine, elevation_sampler, crs=25832):
    """
    Returns
    -------
        title, timetable, electrification, maxspeed, inclination
    """

    osm_data = sql_get_osm(trip_id, engine, crs)

    trip_headsign, timetable = sql_get_timetable(trip_id, engine)

    title = str(trip_id) + " " + trip_headsign

    electrification = get_osm_prop(osm_data, "electrified")

    maxspeed = get_osm_prop(osm_data, "maxspeed")

    inclination = sql_get_inclination(trip_id, osm_data, elevation_sampler, engine)

    return title, timetable, electrification, maxspeed, inclination
