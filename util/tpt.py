import re
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
from ElevationSampler import ElevationProfile
from numpy import ndarray
from pandas import DataFrame


def process_ele(elevation: ndarray, distances: ndarray, brunnels: DataFrame, first_sample_distance: float = 10.0,
                end_sample_distance: float = 100., construct_brunnels: bool = True,
                max_brunnel_length: float = 300., construct_brunnel_thresh: float = 5., diff_kernel_dist: int = 10,
                adjust_window_size: int = 12, std_thresh: float = 3., sub_factor: float = 2., clip: float = 20,
                smooth_window_size: int = 501, poly_order: int = 3, degrees: bool = False,
                smooth_after_resampling: bool = True, window_size_2: int = 21, poly_order_2: int = 1,
                mode: str = "nearest", adjust_forest_height: bool = True, adjust_method: str = "minimum",
                minimum_loops: int = 1, double_adjust: bool = True, drop_last_incl_if_high: bool = True,
                last_incl_thresh: float = 10., last_incl_dist: float = 100., min_ele: float = -3, max_ele: float = 2962.) -> Tuple[DataFrame, DataFrame, DataFrame]:

    distances, elevation = np.array(distances), np.array(elevation)

    # filter unrealistic values
    keep = (elevation > min_ele) & (elevation < max_ele)
    elevation = elevation[keep]
    distances = distances[keep]

    elevation_profile = ElevationProfile(distances, elevation)

    # first_resample so that equidistant sample points at first_sample_distance apart
    elevation_10 = elevation_profile.resample(first_sample_distance).get_elevations()
    distances_10 = elevation_profile.get_distances()

    # then interpolate the brunnels
    ele_brunnel = elevation_profile.interpolate_brunnels(brunnels,
                                                         distance_delta=first_sample_distance,
                                                         construct_brunnels=construct_brunnels,
                                                         max_brunnel_length=max_brunnel_length,
                                                         construct_brunnel_thresh=construct_brunnel_thresh,
                                                         diff_kernel_dist=diff_kernel_dist).get_elevations()

    # then adjust the forest height
    ele_adjusted = ele_brunnel
    if adjust_forest_height:
        ele_adjusted = elevation_profile.to_terrain_model(method=adjust_method,
                                                          window_size=adjust_window_size,
                                                          std_thresh=std_thresh, sub_factor=sub_factor, clip=clip,
                                                          minimum_loops=minimum_loops).get_elevations()

    # then adjust the forest height again with method variance
    if double_adjust and adjust_forest_height:
        ele_adjusted = elevation_profile.to_terrain_model(method="variance",
                                                          window_size=adjust_window_size,
                                                          std_thresh=std_thresh, sub_factor=sub_factor,
                                                          clip=clip).get_elevations()

    # then smooth the elevation profile with high polyorder
    ele_smoothed = elevation_profile.smooth(window_size=smooth_window_size, poly_order=poly_order,
                                            mode=mode).get_elevations()

    # then resample the elevation to the end sample distance
    elevation_100 = elevation_profile.resample(end_sample_distance).get_elevations()
    distances_100 = elevation_profile.get_distances()

    # then smooth again with averaging smoothing method
    if smooth_after_resampling:
        elevation_100 = elevation_profile.smooth(window_size=window_size_2, poly_order=poly_order_2,
                                                 mode=mode).get_elevations()

    # then calculate the inclination
    incl_100 = elevation_profile.inclination(degrees=degrees)

    # set last incl to 0 in some cases
    if drop_last_incl_if_high \
            and abs(incl_100[-1]) > last_incl_thresh \
            and distances_100[-1] - distances_100[-2] < last_incl_dist:
        incl_100[-1] = 0

    distances_incl = distances_100[:-1]

    data = {
        "distance": distances_10,
        "elevation": elevation_10,
        "ele_brunnel": ele_brunnel,
        "ele_adjusted": ele_adjusted,
        "ele_smoothed": ele_smoothed
    }
    elevation_pipeline_df: DataFrame = pd.DataFrame.from_dict(data)

    data = {
        "distance": distances_100,
        "elevation": elevation_100
    }
    elevation_result_df: DataFrame = pd.DataFrame.from_dict(data)

    data = {
        "start_dist": distances_incl,
        "incl": incl_100
    }
    inclination_df: DataFrame = pd.DataFrame.from_dict(data)

    return elevation_pipeline_df, elevation_result_df, inclination_df


def flip_trip(start_dists: ndarray, end_dists: ndarray, parameter: ndarray, invert_para: bool = False) \
        -> Tuple[ndarray, ndarray, ndarray]:
    """
    Calculate the distances and parameter from the other direction.

    Parameters
    ----------
    start_dists : Numpy Array
    end_dists : Numpy Array
    parameter : Numpy Array
    invert_para : bool
        Take parameter * -1

    Returns
    -------
    Numpy Array
        The flipped distances and parameter
    """

    trip_length = end_dists[-1]

    # calculate start dists from the end distances
    new_start_dists = end_dists[::-1]
    new_start_dists = trip_length - new_start_dists

    new_end_dists = start_dists[::-1]
    new_end_dists = trip_length - new_end_dists

    parameter = parameter[::-1]

    if invert_para:
        parameter = parameter * -1

    return new_start_dists, new_end_dists, parameter


def calc_end_dists(start_dists: ndarray, trip_length: float) -> ndarray:
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


def create_umlauf(parameter_df: DataFrame, trip_count: int, parameter_column: str,
                  start_dists_column: str = "start_dist", end_dists_column: str = "end_dist", flip_first: bool = False,
                  drop_dups: bool = True):
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

    start_dists = parameter_df[start_dists_column].values
    end_dists = parameter_df[end_dists_column].values
    parameter = parameter_df[parameter_column].values

    trip_length = end_dists[-1]

    if flip_first:
        start_dists, end_dists, parameter = flip_trip(start_dists, end_dists, parameter, invert_para=invert_para)

    res_dists = []
    res_para = []

    start_dist = 0
    for _ in range(trip_count):
        distances = start_dists + start_dist
        res_dists.append(distances)
        res_para.append(parameter)

        # add trip length
        start_dist += trip_length

        # flip the trip
        start_dists, end_dists, parameter = flip_trip(start_dists, end_dists, parameter, invert_para=invert_para)

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


def trip_title_to_filename(title: str) -> str:
    special_char_map = {ord('ä'): 'ae', ord('ü'): 'ue', ord('ö'): 'oe', ord('ß'): 'ss', ord('('): '', ord(')'): '',
                        ord(' '): '_'}

    filename = title.lower()
    filename = filename.replace(" - ", " ")
    filename = filename.translate(special_char_map)
    filename = re.sub('[^a-z0-9_]+', '', filename)

    return filename


def write_input_sheet(trip_title: str, timetable: DataFrame, electrification: DataFrame, max_speed: DataFrame,
                      inclination: DataFrame, params: Optional[Dict] = None, folder: Optional[str] = None) -> int:
    """

    Parameters
    ----------
    trip_title
    timetable
    electrification
    max_speed
    inclination
    params : dict
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

    if params is not None:
        default_params.update(params)

    filename = trip_title_to_filename(trip_title)

    if folder is not None:
        filename = folder + '/' + filename

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

    # electrification end
    worksheet.write(current_row, 0, "]end")
    current_row += 1

    # empty row
    current_row += 1

    # maxspeed header
    worksheet.write(current_row, 0, "]speedLimits")
    worksheet.write(current_row, 1, "v [km/h]")
    current_row += 1

    # maxspeed
    max_speed[["start_dist", "maxspeed"]].to_excel(writer, sheet_name=default_params['sheet1'], index=False,
                                                   header=False,
                                                   startrow=current_row)

    current_row += max_speed.shape[0]

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

    return 0
