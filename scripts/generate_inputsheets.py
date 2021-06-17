from ElevationSampler import DEM
from sqlalchemy import create_engine, text
import pandas as pd
import util
import multiprocessing as mp

from scripts.database_config import engine_config

if __name__ == '__main__':

    dem_file = "../../../data/dem/de_region.tiff"
    output_folder = "../../../simulation/input_sheets/all_trip_candidates"
    chart_folder = "../../../simulation/charts/all_trip_candidates"
    n_threads = 8

    engine = create_engine(engine_config)
    railway_db = util.RailwayDatabase(engine)
    dem = DEM(dem_file)

    trip_generator = util.TripGenerator(dem, engine)

    sql = """
        select distinct same_geom_and_stops_candidate
        from diesel_nahverkehr
        order by same_geom_and_stops_candidate
        """

    candidate_trip_ids = pd.read_sql_query(text(sql), con=engine)

    candidate_trip_ids = candidate_trip_ids.same_geom_and_stops_candidate.unique()
    candidate_trip_ids.sort()

    error_ids = []

    def exception_wrapper(trip_id):
        print(trip_id)
        try:
            trip = trip_generator.generate_from_railway_db(trip_id)
            # print the warnings of the trip
            for warning in trip.warnings:
                print(str(trip_id) + ": " + warning)

            trip.write_input_sheet(template_file='../util/tpt_input_template.xlsx',
                                   folder=output_folder)

            trip.summary_chart(save=True, folder=chart_folder)

        except Exception as err:
            # use std out as shared queue for logging errors
            print(str(trip_id) + ": ERROR")
            print(err)

    pool = mp.Pool(n_threads)
    res = pool.map(exception_wrapper, candidate_trip_ids)

    print('Process finished.')
    print('# errors: ' + str(len(error_ids)))
    print('------------------')

    print('Start recalculate errors.')
    # try regeneration of error_ids, if error due database connectivity
    for trip_id in error_ids:
        exception_wrapper(trip_id)
    print('finished.')


