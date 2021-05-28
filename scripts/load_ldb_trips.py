from sqlalchemy import create_engine, text
import pandas as pd
import util
import multiprocessing as mp

from scripts.database_config import engine_config

if __name__ == '__main__':

    n_threads = 14

    engine = create_engine(engine_config)
    railway_db = util.RailwayDatabase(engine)

    sql = """
        SELECT *
        FROM ldb_trip_candidates
        """

    candidate_trip_ids = pd.read_sql_query(text(sql), con=engine)
    candidate_trip_ids = candidate_trip_ids.same_geom_candidate.unique()

    error_ids = []

    def exception_wrapper(trip_id):
        print(trip_id)
        try:
            e = railway_db.save_trip_osm_tables(int(trip_id), trip_id_is_candidate_trip_id=True, replace=True,
                                                max_trip_length=None)
            if e != 0:
                print("LONGTRIP: " + str(e))
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
    for trip_id in []:  # [264008,222025,123715,49812,246108,20527,554473,45908,98281,113905,851074,602620,200753,9344,25399,1588,166351,183539,81768,54449,469678,146260,235167,9,90623,229777,264505,104942,1615538,568868,11976,956703,174332,362635,434121,384670,663762,238525,153982,35617,1121862,256968,37583,45521,516942,1332125,448844,49397,122447,242244,300296,1000847,1108318,164289,31775,1496,201610,259363,12032,355341,4036,223001,86406,222185,52304,1456,134522,335535,130932,1509682,901432,72679,119995,175820,230826,74327,1048130,92731,47380,16593,22175,1600089,91757,391597,764553,136441,258216,167629,47792,1426205,279059,29348,7897,25542,24949,7602,49201,873503,547037,355550,99634,36013,1323598,46714,1677681,8171,199873,231226,62529,189367,525163,580137,61087,31819,702112,455049,4591,53180,537331,67143,53090,130568,163203,67341,996813,62939,280142,38268,73135,201582,5432,459897,666343,570351,72403,164384,528688,436624,943507,14710,233145,1570702,142296,439815,182520,87762,159271,267956,26170,974977,266101,36286,704494,348194,113322,916891,221421,20699,69347,586419,257820,120252,65485,295299,93876,14723,97002,46477,571310,1774665,12947,2002,386858,751544,478365]:
        exception_wrapper(trip_id)
    print('finished.')


