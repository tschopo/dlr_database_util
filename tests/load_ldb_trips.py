from sqlalchemy import create_engine, text
import pandas as pd
import util
import multiprocessing as mp

if __name__ == '__main__':

    engine = create_engine('postgresql://dlr:dlr4thewin@localhost/liniendatenbank')
    railway_db = util.RailwayDatabase(engine)

    sql = """
        SELECT *
        FROM ldb_trip_candidates
        """
    candidate_trip_ids = pd.read_sql_query(text(sql), con=engine)

    candidate_trip_ids = candidate_trip_ids.same_geom_candidate.unique()

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

    pool = mp.Pool(16)

    res = pool.map(exception_wrapper, candidate_trip_ids)
