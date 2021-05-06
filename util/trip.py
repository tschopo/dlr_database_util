from typing import Optional, List

import pandas as pd
from ElevationSampler import DEM, ElevationProfile
from folium import Map
import numpy as np
from geopandas import GeoSeries
from pandas import DataFrame
from sqlalchemy.engine import Engine

from util import sql_get_osm_from_line, get_osm_prop, RailwayDatabase, elevation_pipeline, write_tpt_input_sheet, \
    write_sensor_input_sheet, read_tpt_output_sheet, add_inputs_to_simulation_results, resample_simulation_results, \
    plot_trip_props, spatial_median


class Trip:
    """
    collection of data that belongs to a trip + functions on the data
    """

    def __init__(self, trip_id, electrified, maxspeed, brunnels, timetable, elevation_profile, trip_geom):
        """

        Parameters
        ----------
        trip_id
        electrified
        maxspeed
        brunnels
        timetable
            DataFrame with Columns: "dist", "stop_name", "stop_duration", "driving_time", "arrival_time",
            "departure_time"
                dist: float, station distance from start
                stop_name: str
                stop_duration: int, in seconds
                driving_time: int, driving time to next station in seconds
                arrival_time: datetime object
                departure_time: datetime object
        elevation_profile
        trip_geom
        """

        self.simulated: bool = False

        self.trip_id: int = int(trip_id)
        self.geom: GeoSeries = trip_geom
        self.timetable = timetable
        self.electrified: DataFrame = electrified
        self.maxspeed: DataFrame = maxspeed
        self.brunnels: DataFrame = brunnels
        self.elevation_profile: ElevationProfile = elevation_profile

        self.warnings: List[str] = []

        # TODO add time_delta column that starts at 0
        # timetable['time_delta'] = timetable["arrival_time"] - timetable.iloc[0]["arrival_time"]

        self.start_time: np.datetime64 = self.timetable.iloc[0]["arrival_time"]
        self.end_time: np.datetime64 = self.timetable.iloc[-1]["arrival_time"]
        # TODO should be timedelta in Seconds? jsut take last element of timetable['time_delta']
        # self.duration = self.end_time - self.start_time

        self.start_station: str = self.timetable["stop_name"].iloc[0]
        self.end_station: str = self.timetable["stop_name"].iloc[-1]

        self.title: str = str(trip_id) + "_" + clean_name(self.start_station) + " - " + clean_name(self.end_station)

        # set the enddists to trip length
        self.length: float = self.geom.length.iloc[0]

        self.maxspeed.at[self.maxspeed.shape[0]-1, 'end_dist'] = np.ceil(self.length)
        self.electrified.at[self.electrified.shape[0] - 1, 'end_dist'] = np.ceil(self.length)

        # also timetable dists don't align perfectly
        # set dist to 0 for first station
        if abs(self.timetable.at[self.timetable.shape[0] - 1, 'dist'] - self.length) > 250:
            self.warnings.append("WARNING: Timetable last station distance doesn't match trip length")
        self.timetable.at[self.timetable.shape[0] - 1, 'dist'] = self.length

        # set dist to 0 for first station
        if self.timetable.iloc[0]["dist"] > 250:
            self.warnings.append("WARNING: First Station far away")
        self.timetable.iloc[0, self.timetable.columns.get_loc("dist")] = 0

        self.simulation_results: Optional[DataFrame] = None

        # then calculate the inclination
        incl = self.elevation_profile.inclination(degrees=False)

        # set last incl to 0 in some cases
        if abs(incl[-1]) > 15:
            incl[-1] = 0
            self.warnings.append("WARNING: Last inclination is above threshold. Setting to 0")

        if np.max(np.abs(incl)) > 30:
            self.warnings.append("WARNING: Inclination above 30 permille")

        if self.elevation_profile.cumulative_ascent() <= 0:
            self.warnings.append("WARNING: Cumulative Climb is 0")

        if self.length < 2000:
            self.warnings.append("WARNING: Trip Length below 2km")

        if np.min(self.timetable.dist - self.timetable.dist.shift(1)) < 100:
            self.warnings.append("WARNING: Stations closer 100m")

        if self.timetable.shape[0] >= 50:
            self.warnings.append("WARNING: More than 50 Stations")

        # weighted maxspeed median must be over 49 kmh
        if spatial_median(self.maxspeed.maxspeed.values,
                          (self.maxspeed.end_dist - self.maxspeed.start_dist).values) < 50:
            self.warnings.append("WARNING: Median Maxspeed below 50")

    def get_stations(self) -> List[str]:
        """

        Returns
        -------
        List[str]
            List of station names in order of arrival

        """
        raise NotImplementedError

    def get_elevation(self, smoothed=True):

        if smoothed:
            data = {"elevation": self.elevation_profile.elevations, "distance": self.elevation_profile.distances}
        else:
            data = {"elevation": self.elevation_profile.elevations_orig,
                    "distance": self.elevation_profile.distances_orig}

        return pd.DataFrame(data)

    def get_velocity(self, delta_t=10):
        if self.simulated:
            velocity = resample_simulation_results(
                self.simulation_results[["time_delta", "distance", "time", "velocity"]], t=delta_t)
            velocity = velocity[["distance", "time", "velocity"]]
        else:
            velocity = None
        return velocity

    def get_power(self, delta_t=10):
        if self.simulated:
            power = resample_simulation_results(
                self.simulation_results[["time_delta", "distance", "time", "power"]], t=delta_t)
            power = power[["distance", "time", "power"]]
        else:
            power = None
        return power

    def write_input_sheet(self, simulation="tpt", template_file=None, folder=None, last_incl_thresh=15.):

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

    def add_simulation_results(self, output_sheet, t=10):
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
        # calculate delay by calculating tpt driving time
        simulated_arrival_time = self.timetable.apply(
            lambda r: find_closest(self.simulation_results, 'distance', r['dist']-10)['time'], axis=1)
        self.timetable["simulated_arrival_time"] = simulated_arrival_time

        simulated_departure_time = self.timetable.apply(
            lambda r: find_closest(self.simulation_results, 'distance', r['dist']+10,
                                   first_occurrence=False)['time'], axis=1)
        self.timetable["simulated_departure_time"] = simulated_departure_time

        simulated_driving_time = simulated_arrival_time.shift(-1) - simulated_departure_time
        delay = simulated_driving_time - pd.to_timedelta(self.timetable['driving_time'], unit='S')
        self.timetable["delay"] = delay.shift(1)

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
        raise NotImplementedError

        # m = plot_osm(self.osm_data, prop=prop)
        # return m

    def summary_chart(self, save=False, filename: Optional[str] = None, folder: str = None, show_delay=True, **kwargs):
        """

        Parameters
        ----------
        save
        filename
            If none uses trip title as filename. must end with '.png'.
        folder
            If should save in sub-folder
        show_delay
            If the simulated delays should be plotted

        Returns
        -------

        """

        if not show_delay or not self.simulated:
            timetable = self.timetable[['dist', 'stop_name', 'arrival_time']]
        else:
            timetable = self.timetable[['dist', 'stop_name', 'arrival_time', 'delay']]

        elevation_orig = self.get_elevation(smoothed=False)
        elevation_smoothed = self.get_elevation(smoothed=True)

        if np.all(elevation_smoothed.elevation == elevation_orig.elevation):
            elevation_orig = None

        chart = plot_trip_props(self.maxspeed, self.electrified, elevation_orig,
                                elevation_smoothed, self.title,
                                self.length, velocity=self.get_velocity(),
                                power=self.get_power(), timetable=timetable, **kwargs)

        if save:
            folder = '' if folder is None else folder + '/'
            filename = self.title + '.png' if filename is None else filename
            chart.save(folder + filename)

        return chart

    def summary(self):
        # print summary stats about the trip, or return as dataframe
        # e_rad_gesamt, e_rad_peak
        # e_break_gesamt, e_break_peak,
        # net_e (rad-break)
        # n_stops, total_travel_time, trip_length,
        # average velocity (when driving), peak velocity
        # median_maxspeed, peak_maxspeed
        # avg/min/max distance between stations,
        # avg/min/max duration between stations,
        # avg/min/max inline,
        # avg/min/max elevation
        # cum_ascent, cum_descent,
        # net elevationgain,

        raise NotImplementedError


class TripGenerator:
    # TODO implement caching so that trip is returned if already generated with same parameters

    def __init__(self, dem: DEM, db_connection: Engine, first_sample_distance: float = 10.0,
                 brunnel_filter_length: float = 10., interpolated: bool = True, ele_pipeline_kwargs=None,
                 get_osm_kwargs=None):
        # TODO better would be to store all parameters and not use kwargs
        # store generation parameters with trip so that reproducible

        self.railway_db = None
        self.engine = db_connection
        self.dem = dem

        if get_osm_kwargs is None:
            get_osm_kwargs = {}
        self.get_osm_kwargs = get_osm_kwargs

        if ele_pipeline_kwargs is None:
            ele_pipeline_kwargs = {}
        self.ele_pipeline_kwargs = ele_pipeline_kwargs

        self.first_sample_distance = first_sample_distance
        self.brunnel_filter_length = brunnel_filter_length
        self.interpolated = interpolated

        self.current_osm_data = None

    def generate_from_railway_db(self, trip_id) -> Trip:

        if self.railway_db is None:
            self.railway_db = RailwayDatabase(self.engine)

        # get shape from database
        trip_geom = self.railway_db.get_trip_shape(trip_id, self.dem.crs)["geom"]

        # get timetable data from database
        timetable = self.railway_db.get_trip_timetable(trip_id)

        # convert the timetable arrival and start time to datetime instead of timedelta
        date = "2021-01-01"
        timetable["arrival_time"] = np.datetime64(date) + timetable.arrival_time
        timetable["departure_time"] = np.datetime64(date) + timetable.departure_time

        # check if generated trip is already stored in database
        if self.railway_db.contains_generated_trip(trip_id):

            # if so get the data from database
            brunnels = self.railway_db.get_trip_brunnels(trip_id)
            electrified = self.railway_db.get_trip_electrified(trip_id)
            maxspeed = self.railway_db.get_trip_maxspeed(trip_id)
            elevation_profile = self.railway_db.get_trip_elevation_profile(trip_id)
            return Trip(trip_id, electrified, maxspeed, brunnels, timetable, elevation_profile, trip_geom)

        # if not, calculate the data
        else:
            return self.generate_from_osm_db(trip_id, trip_geom, timetable)

    def generate_from_simulation_results(self, results_file):
        # problem: how to get trip_geom
        raise NotImplementedError

    def generate_from_osm_db(self, trip_id: int, trip_geom: GeoSeries, timetable: DataFrame):
        osm_data = sql_get_osm_from_line(trip_geom, self.engine, **self.get_osm_kwargs)

        self.current_osm_data = osm_data

        return self.generate(trip_id, trip_geom, osm_data, timetable)

    def generate(self, trip_id, trip_geom, osm_data, timetable):

        trip_length = trip_geom.length.iloc[0]

        electrified = get_osm_prop(osm_data, "electrified", trip_length=trip_length)
        maxspeed = get_osm_prop(osm_data, "maxspeed", trip_length=trip_length)
        brunnels = get_osm_prop(osm_data, "brunnel", brunnel_filter_length=self.brunnel_filter_length)

        elevation_profile = self.dem.elevation_profile(trip_geom, distance=self.first_sample_distance,
                                                       interpolated=self.interpolated)

        elevation_profile = elevation_pipeline(elevation_profile, brunnels, **self.ele_pipeline_kwargs)

        return Trip(trip_id, electrified, maxspeed, brunnels, timetable, elevation_profile, trip_geom)


def clean_name(name: str) -> str:
    # TODO to simple because "bad schussenried" becomes just "bad"
    name = name.split()
    name = name[0].split('(')

    return name[0]


def find_closest(df: DataFrame, col_name: str, value: any, first_occurrence=True):
    """
    returns the row in "df", with the closest value to "value" in column "col_name"

    Parameters
    ----------
    df
    col_name
    value
    first_occurrence
        If True takes the first occurance of closest value, if False takes the last occurance of closest value.

    Returns
    -------

    """

    if first_occurrence:
        index = np.abs(df[col_name] - value).idxmin()
    else:
        index = np.abs(df[col_name] - value)[::-1].idxmin()
    return df.loc[index]
