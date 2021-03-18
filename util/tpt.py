import re
import numpy as np
import pandas as pd
from sqlalchemy import text
from .osm import get_osm_prop
from .sql import sql_get_osm, sql_get_trip_geom


def get_inclination(trip_id, osm_data, elevation_sampler, engine, first_sample_distance=10, end_sample_distance=100,
                    brunnel_filter_length=10, interpolated=True, adjust_window_size=12, std_thresh=3, sub_factor=3,
                    clip=20, smooth_window_size=301, poly_order=3, degrees=False, smooth_after_resampling=True,
                    window_size_2=5, poly_order_2=1, mode="nearest", output_all=False):
    brunnels = get_osm_prop(osm_data, "brunnel", brunnel_filter_length=brunnel_filter_length)

    trip_geom = sql_get_trip_geom(trip_id, engine, elevation_sampler.dem.crs)

    x_coords, y_coords, distances, elevation = elevation_sampler.elevation_profile(trip_geom,
                                                                                   distance=first_sample_distance,
                                                                                   interpolated=interpolated)

    ele_brunnel = elevation_sampler.interpolate_brunnels(elevation, distances, brunnels, first_sample_distance)

    ele_adjusted = elevation_sampler.adjust_forest_height(ele_brunnel, window_size=adjust_window_size,
                                                          std_thresh=std_thresh, sub_factor=sub_factor, clip=clip)

    ele_smoothed = elevation_sampler.smooth_ele(ele_adjusted, window_size=smooth_window_size, poly_order=poly_order,
                                                mode=mode)

    distances_100, elevation_100 = elevation_sampler.resample_ele(ele_smoothed, distances, end_sample_distance)

    if smooth_after_resampling:
        elevation_100 = elevation_sampler.smooth_ele(elevation_100, window_size=window_size_2, poly_order=poly_order_2,
                                                     mode=mode)

    incl_100 = elevation_sampler.ele_to_incl(elevation_100, distances_100, degrees=degrees)
    distances_100 = distances_100[:-1]

    if output_all:
        return distances, elevation, ele_brunnel, ele_adjusted, ele_smoothed, distances_100, elevation_100

    data = {"start_dist": distances_100, "incl": incl_100}

    return pd.DataFrame.from_dict(data)


def get_timetable(trip_id, engine, min_stop_duration=30, round_int=True):
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


def trip_title_to_filename(title):
    special_char_map = {ord('ä'): 'ae', ord('ü'): 'ue', ord('ö'): 'oe', ord('ß'): 'ss', ord('('): '', ord(')'): '',
                        ord(' '): '_'}

    filename = title.lower()
    filename = filename.replace(" - ", " ")
    filename = filename.translate(special_char_map)
    filename = re.sub('[^a-z0-9_]+', '', filename)

    return filename


def write_input_sheet(trip_title, timetable, electrification, maxspeed, inclination):
    filename = trip_title_to_filename(trip_title)

    writer = pd.ExcelWriter(filename + '.xlsx', engine='xlsxwriter')

    # we first need to write a dataframe, to get the sheets later

    # timetable
    timetable.to_excel(writer, sheet_name='Timetable and limits', index=False, header=False, startrow=3)

    workbook = writer.book
    worksheet = writer.sheets['Timetable and limits']

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
    electrification[["start_dist", "electrified"]].to_excel(writer, sheet_name='Timetable and limits', index=False,
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
    maxspeed[["start_dist", "maxspeed"]].to_excel(writer, sheet_name='Timetable and limits', index=False, header=False,
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
    worksheet.write(current_row, 1, 1.2)
    current_row += 1

    worksheet.write(current_row, 0, "]end")
    current_row += 1

    current_row += 1

    worksheet.write(current_row, 0, "]decelerationLimits")
    worksheet.write(current_row, 1, "a [m/s^2]")
    current_row += 1

    worksheet.write(current_row, 0, 0)
    worksheet.write(current_row, 1, 0.9)
    current_row += 1

    worksheet.write(current_row, 0, "]end")

    # column width
    worksheet.set_column(0, 3, 25)

    # new worksheet
    inclination.to_excel(writer, sheet_name='Gradients and curves', index=False, header=False, startrow=3)

    worksheet = writer.sheets['Gradients and curves']

    # title
    worksheet.write('A1', "]gravity_acceleration(m/s2)")
    worksheet.write('B1', 9.81)

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

    worksheet = workbook.add_worksheet("TPT requirements")

    worksheet.write('A1', "]time_overhead")
    worksheet.write('B1', 0.05)

    worksheet.write('A4', "]gradient_interpolation_binary")
    worksheet.write('B4', 0)

    worksheet.write('A6', "]effort_in_curve(N.m/kg)")
    worksheet.write('B6', 8)

    worksheet.write('A9', "]resistances_kp(m)_name(word)")

    worksheet.write('A10', 0)
    worksheet.write('B10', "normal")

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

    trip_headsign, timetable = get_timetable(trip_id, engine)

    title = str(trip_id) + " " + trip_headsign

    electrification = get_osm_prop(osm_data, "electrified")

    maxspeed = get_osm_prop(osm_data, "maxspeed")

    inclination = get_inclination(trip_id, osm_data, elevation_sampler, engine)

    return title, timetable, electrification, maxspeed, inclination
