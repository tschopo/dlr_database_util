import pandas as pd

from util import sql_get_osm_from_line, get_osm_prop, RailwayDatabase, elevation_pipeline, write_tpt_input_sheet, \
    write_sensor_input_sheet, read_tpt_output_sheet, get_maxspeed_at_dist, get_elevation_at_dist, \
    get_electrified_at_dist, add_inputs_to_simulation_results, resample_simulation_results


class Trip:
    """
    collection of data that belongs to a trip
    """

    def __init__(self, trip_id, railway_db: RailwayDatabase, dem, first_sample_distance: float = 10.0,
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

        self.trip_geom = shape["geom"]

        # get osm data from osm table
        self.osm_data = sql_get_osm_from_line(self.trip_geom, railway_db.engine, **osm_kwargs)

        # get timetable data from database
        self.timetable = railway_db.get_trip_timetable(self.trip_id)

        start = self.timetable["stop_name"].iloc[0]
        end = self.timetable["stop_name"].iloc[-1]

        self.title = clean_name(start) + " - " + clean_name(end)

        self.electrified = get_osm_prop(self.osm_data, "electrified")
        self.maxspeed = get_osm_prop(self.osm_data, "maxspeed")
        self.brunnels = get_osm_prop(self.osm_data, "brunnel", brunnel_filter_length=brunnel_filter_length)

        elevation_profile = dem.elevation_profile(self.trip_geom, distance=first_sample_distance,
                                                  interpolated=interpolated)

        self.elevation_profile = elevation_pipeline(elevation_profile, self.brunnels, **ele_kwargs)

        self.simulation_results = None

    def get_elevation(self, smoothed=True):
        if smoothed:
            data = {"elevation": self.elevation_profile.elevations, "distance": self.elevation_profile.distances}
        else:
            data = {"elevation": self.elevation_profile.elevations_orig, "distance": self.elevation_profile.distances_orig}

        return pd.DataFrame(data)

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

        timetable = self.timetable[["dist", "stop_name", "stop_duration", "driving_time", "arrival_time",
                                    "departure_time"]]

        if simulation == "tpt":
            return write_tpt_input_sheet(template_file, self.title, timetable, self.electrified, self.maxspeed,
                                         inclination, folder=folder)

        elif simulation == "sensor":
            return write_sensor_input_sheet(self.title, self.timetable, self.electrified, self.maxspeed, inclination,
                                            folder=folder)

    def add_simulation_results(self, output_sheet):
        tpt_df = read_tpt_output_sheet(output_sheet)
        tpt_df = add_inputs_to_simulation_results(tpt_df, self.get_elevation(smoothed=True), self.maxspeed,
                                                  self.electrified)

        tpt_df = resample_simulation_results(tpt_df, s=10)

        self.timetable
        self.simulated = True
        self.simulation_results = tpt_df.copy()


def clean_name(name: str) -> str:
    name = name.split()
    name = name[0].split('(')

    return name[0]
