from sqlalchemy import create_engine, text
import pandas as pd
import util
import multiprocessing as mp

from scripts.database_config import engine_config

if __name__ == '__main__':

    n_threads = 1

    engine = create_engine(engine_config)
    railway_db = util.RailwayDatabase(engine)

    # get all same_geom_and_stops_candidates of rb/re that traverse sbahn
    sql = """
        -- rb/re finden die über reine sbahn fahren
        SELECT distinct same_geom_and_stops_candidate
          FROM
          (
            SELECT * FROM ed_bahnstrecke
            WHERE bahnkategorie = ARRAY[1104]
          ) sbahnstrecken, gtfs_trips_ax_bahnstrecke, geo_trips, geo_routes
        where sbahnstrecken.t_id = gtfs_trips_ax_bahnstrecke.t_id
        and gtfs_trips_ax_bahnstrecke.same_geom_and_stops_candidate = geo_trips.trip_id
        and geo_trips.route_id = geo_routes.route_id
          AND (route_long_name LIKE 'RE%'
          OR route_long_name LIKE 'RB%')
        """

    same_geom_and_stops_candidates = pd.read_sql_query(text(sql), con=engine)
    same_geom_and_stops_candidates = same_geom_and_stops_candidates.sort_values(by='same_geom_and_stops_candidate',
                                                                                ignore_index=True).same_geom_and_stops_candidate.values

    def exception_wrapper(trip_id):
        print(trip_id)
        try:
            same_geom_and_stops_candidate = int(trip_id)

            sql = """
                SELECT * from gtfs_get_ax_route_sbahn_exception_wrapper(:same_geom_and_stops_candidate);
            """

            route = pd.read_sql_query(text(sql), con=engine,
                                      params={'same_geom_and_stops_candidate': same_geom_and_stops_candidate})
            with engine.connect() as con:
                route.to_sql('gtfs_trips_ax_bahnstrecke', con, if_exists='append', index=False)

        except Exception as err:
            print("ERROR: " + str(trip_id))
            print(err)

    pool = mp.Pool(n_threads)
    res = pool.map(exception_wrapper, same_geom_and_stops_candidates)

    print('Process finished.')
    print('------------------')
