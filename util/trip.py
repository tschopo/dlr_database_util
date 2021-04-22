from typing import Optional

import pandas as pd
from folium import Map
import numpy as np
from pandas import DataFrame

from util import sql_get_osm_from_line, get_osm_prop, RailwayDatabase, elevation_pipeline, write_tpt_input_sheet, \
    write_sensor_input_sheet, read_tpt_output_sheet, add_inputs_to_simulation_results, resample_simulation_results, plot_osm, plot_trip_props


class Trip:
    """
    collection of data that belongs to a trip
    """

    def __init__(self, trip_id, railway_db: RailwayDatabase, dem, date="2021-01-01", first_sample_distance: float = 10.0,
                 brunnel_filter_length: float = 10., interpolated: bool = True, ele_kwargs=None,
                 osm_kwargs=None):

        if osm_kwargs is None:
            osm_kwargs = {}
        if ele_kwargs is None:
            ele_kwargs = {}

        self.simulated = False
        self.trip_id = int(trip_id)
        self.crs = dem.crs

        # get shape from database
        shape = railway_db.get_trip_shape(trip_id, dem.crs)

        self.geom = shape["geom"]

        # get osm data from osm table
        self.osm_data = sql_get_osm_from_line(self.geom, railway_db.engine, **osm_kwargs)

        # get timetable data from database
        self.timetable = railway_db.get_trip_timetable(self.trip_id)

        # add time_delta column that starts at 0
        self.timetable['time_delta'] = self.timetable["arrival_time"] - self.timetable.iloc[0]["arrival_time"]

        # convert the timetable arrival and start time to datetime instead of timedelta
        self.timetable["arrival_time"] = np.datetime64(date) + self.timetable.arrival_time
        self.timetable["departure_time"] = np.datetime64(date) + self.timetable.departure_time

        self.warnings = []
        # set dist to 0 for first station
        if self.timetable.iloc[0]["dist"] > 250:
            self.warnings.append("WARNING: First Station far away")
        self.timetable.iloc[0, self.timetable.columns.get_loc("dist")] = 0

        self.start_time = self.timetable.iloc[0]["arrival_time"]
        self.end_time = self.timetable.iloc[-1]["arrival_time"]

        self.start_station = self.timetable["stop_name"].iloc[0]
        self.end_station = self.timetable["stop_name"].iloc[-1]

        self.title = str(trip_id) + "_" + clean_name(self.start_station) + " - " + clean_name(self.end_station)

        self.electrified = get_osm_prop(self.osm_data, "electrified")
        self.maxspeed = get_osm_prop(self.osm_data, "maxspeed")

        # TODO set maxspeed endist correctly by get_osm
        self.length = self.geom.length.iloc[0]
        self.maxspeed.at[self.maxspeed.shape[0]-1, 'end_dist'] = self.length + 1
        self.electrified.at[self.electrified.shape[0] - 1, 'end_dist'] = self.length + 1

        self.brunnels = get_osm_prop(self.osm_data, "brunnel", brunnel_filter_length=brunnel_filter_length)

        elevation_profile = dem.elevation_profile(self.geom, distance=first_sample_distance,
                                                  interpolated=interpolated)

        self.elevation_profile = elevation_pipeline(elevation_profile, self.brunnels, **ele_kwargs)

        self.simulation_results = None

    def get_elevation(self, smoothed=True):
        """ if simulated also returns time column. """

        # TODO: make optional equidistant time ot equidistant distance

        if smoothed:
            data = {"elevation": self.elevation_profile.elevations, "distance": self.elevation_profile.distances}

        else:
            data = {"elevation": self.elevation_profile.elevations_orig, "distance": self.elevation_profile.distances_orig}

        return pd.DataFrame(data)

    def get_velocity(self, delta_t=10):
        if self.simulated:
            velocity = resample_simulation_results(self.simulation_results[["time_delta", "distance", "time", "velocity"]], t=delta_t)
            velocity = velocity[["distance", "time", "velocity"]]
        else:
            velocity = None
        return velocity

    def get_power(self, delta_t=10):
        if self.simulated:
            power = resample_simulation_results(self.simulation_results[["time_delta", "distance", "time", "power"]], t=delta_t)
            power = power[["distance", "time", "power"]]
        else:
            power = None
        return power

    def write_input_sheet(self, simulation="tpt", template_file=None, folder=None, last_incl_thresh=10.):

        if simulation == "tpt" and template_file is None:
            raise Exception("TPT writer needs a template_file")

        # then calculate the inclination
        incl = self.elevation_profile.inclination(degrees=False)

        # set last incl to 0 in some cases
        if last_incl_thresh is not None and abs(incl[-1]) > last_incl_thresh:
            incl[-1] = 0

        distances_incl = self.elevation_profile.distances[:-1]

        data = {
            "start_dist": distances_incl,
            "incl": incl
        }
        inclination: pd.DataFrame = pd.DataFrame.from_dict(data)

        timetable = self.timetable[["dist", "stop_name", "stop_duration", "driving_time"]]

        if simulation == "tpt":
            return write_tpt_input_sheet(template_file, self.title, timetable, self.electrified, self.maxspeed,
                                         inclination, folder=folder)

        elif simulation == "sensor":
            return write_sensor_input_sheet(self.title, self.timetable, self.electrified, self.maxspeed, inclination,
                                            folder=folder)

    def add_simulation_results(self, output_sheet, t=10, train_length=100):
        tpt_df = read_tpt_output_sheet(output_sheet)

        if t is not None:
            tpt_df = resample_simulation_results(tpt_df, t=t)

        tpt_df = add_inputs_to_simulation_results(tpt_df, self.get_elevation(smoothed=True), self.maxspeed,
                                                  self.electrified)

        # add time column
        tpt_df['time'] = tpt_df.time_delta + self.start_time

        self.simulated = True
        self.simulation_results = tpt_df

        # add delay column to timetable
        # find the simulation time at distance of station
        simulated_arrival_time = self.timetable.apply(
            lambda r: find_closest(self.simulation_results, 'distance', r['dist']-(train_length/2))['time'], axis=1)
        delay = simulated_arrival_time - self.timetable.arrival_time
        self.timetable["delay"] = delay


    def plot_map(self, prop=None) -> Map:
        """
        Returns a folium map with the trip.

        Parameters
        ----------
        prop
            The osm parameter to plot. Can be "maxspeed" or "electrified".

        Returns
        -------
            Map
                Folium map

        """

        m = plot_osm(self.osm_data, prop=prop)
        return m

    def summary_chart(self, save=False, filename: Optional[str] = None, folder=None, show_delay=True, **kwargs):
        """

        Parameters
        ----------
        save
        filename
            If none uses trip title as filename. must end with '.png'.

        Returns
        -------

        """

        if not show_delay:
            timetable = self.timetable[['dist', 'stop_name', 'arrival_time']]
        else:
            timetable = self.timetable[['dist', 'stop_name', 'arrival_time', 'delay']]

        chart = plot_trip_props(self.maxspeed, self.electrified, self.get_elevation(smoothed=False),
                                self.get_elevation(smoothed=True), self.title, self.length, velocity=self.get_velocity(),
                                power=self.get_power(), timetable=timetable, **kwargs)

        if save:
            folder = '' if folder is None else folder + '/'
            filename = self.title + '.png' if filename is None else filename
            chart.save(folder + filename)

        return chart

    # def summary
    # summary stats about the trip
    # e_rad, e_break, n_stops, total_travel_time, trip_length


def clean_name(name: str) -> str:
    name = name.split()
    name = name[0].split('(')

    return name[0]


def find_closest(df: DataFrame, col_name: str, value: any):
    """
    returns the row in "df", with the closest value to "value" in column "col_name"

    Parameters
    ----------
    df
    col_name
    value

    Returns
    -------

    """
    index = abs(df[col_name] - value).idxmin()
    return df.loc[index]