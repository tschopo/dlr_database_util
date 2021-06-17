from ElevationSampler import DEM
from sqlalchemy import create_engine, text
import pandas as pd
import util
import multiprocessing as mp

from scripts.database_config import engine_config

if __name__ == '__main__':

    dem_file = "/media/data/Documents/dlr/datenbank/data/dem/de_region.tiff"
    n_threads = 2

    engine = create_engine(engine_config)
    railway_db = util.RailwayDatabase(engine)
    dem = DEM(dem_file)

    error_ids = []

    sql = """
        SELECT *
        FROM ldb_trip_candidates
        """
    candidate_trip_ids = pd.read_sql_query(text(sql), con=engine)

    candidate_trip_ids = candidate_trip_ids.same_geom_candidate.unique()
    candidate_trip_ids.sort()

    def exception_wrapper(trip_id):
        print(trip_id)
        try:
            brunnels = railway_db.get_trip_brunnels(trip_id)
            railway_db.save_trip_elevation_table(int(trip_id), dem, brunnels, trip_id_is_candidate_trip_id=True,
                                                 replace=True)
        except Exception as err:
            print("ERROR: " + str(trip_id))
            print(err)

    pool = mp.Pool(n_threads)
    res = pool.map(exception_wrapper, candidate_trip_ids)

    print('Process finished.')
    print('# errors: ' + str(len(error_ids)))
    print('------------------')

    print('Start recalculate errors.')
    # try regeneration of error_ids, if error due database connectivity
    for trip_id in []:#[264008,20527,9,1615538,174332,362635,663762,45521,122447,4036,901432,764553,136441,525163,14712,455049,67341,528688,14710,142296,439815,182520,974977,36286,295299,2002,478365]:
        exception_wrapper(trip_id)
    print('finished.')
