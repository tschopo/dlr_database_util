from ElevationSampler import DEM
from sqlalchemy import create_engine, text
import pandas as pd
import util
import multiprocessing as mp

from scripts.database_config import engine_config

if __name__ == '__main__':

    n_threads = 8


    engine = create_engine(engine_config)
    railway_db = util.RailwayDatabase(engine)

    # get all same_geom_and_stops_candidates
    sql = """
        select distinct same_geom_candidate 
        from ldb_trip_candidates;
        """

    same_geom_candidates = pd.read_sql_query(text(sql), con=engine)
    same_geom_candidates = same_geom_candidates.sort_values(by='same_geom_candidate',
                                                            ignore_index=True).same_geom_candidate.values

    def exception_wrapper(trip_id):
        trip_id = int(trip_id)
        print(trip_id)
        try:
            # get shape from database
            shape = railway_db.get_trip_shape(trip_id, crs=25832)

            trip_geom = shape["geom"]
            osm_data = util.sql_get_osm_from_line(trip_geom, engine)
            osm_data['same_geom_candidate'] = trip_id

            with engine.connect() as con:
                osm_data[['same_geom_candidate', 'way_id']].drop_duplicates(ignore_index=True).to_sql(
                    'gtfs_trips_osm_railways', con, if_exists='append', index=False)

        except Exception as err:
            print("ERROR: " + str(trip_id))
            print(err)

    pool = mp.Pool(n_threads)
    res = pool.map(exception_wrapper, same_geom_candidates)

    print('Process finished.')
    print('------------------')
