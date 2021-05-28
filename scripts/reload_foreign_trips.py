"""
Recalculate elevations and osm props for trips outside germany. User other DEM and set_unknown_electrified_to_no=False
"""
import multiprocessing as mp
import pandas as pd
import numpy as np
from ElevationSampler import DEM
from sqlalchemy import create_engine, text

import util
from scripts.database_config import engine_config


def exception_wrapper(trip_id):
    print(trip_id)
    try:
        railway_db.save_trip_osm_tables(int(trip_id), trip_id_is_candidate_trip_id=True, replace=True,
                                        max_trip_length=None, set_unknown_electrified_to_no=False)

    except Exception as err:
        print("ERROR: (OSM)" + str(trip_id))
        print(err)

    try:
        brunnels = railway_db.get_trip_brunnels(trip_id)
        railway_db.save_trip_elevation_table(int(trip_id), dem, brunnels, trip_id_is_candidate_trip_id=True,
                                             replace=True)
    except Exception as err:
        print("ERROR: (ELEVATION)" + str(trip_id))
        print(err)


if __name__ == '__main__':
    engine = create_engine(engine_config)
    railway_db = util.RailwayDatabase(engine)
    n_threads = 8
    dem = DEM("/media/data/Documents/dlr/datenbank/data/dem/de_region.tiff")

    # get all trips that have a part outside of germany
    sql = """\
    select distinct same_geom_candidate 
    from (
        select same_geom_candidate, st_contains((
            select st_buffer(geometry, 0.01) 
            from administrative_boundaries 
            where name = 'Deutschland' limit 1), 
            geo_shape_geoms.geom) as in_germany
        from (select distinct same_geom_candidate from ldb_trip_candidates) t
    left join geo_trips on same_geom_candidate = geo_trips.trip_id
    left join geo_shape_geoms on geo_trips.shape_id = geo_shape_geoms.shape_id)t
    where not in_germany
    """

    candidate_trip_ids = pd.read_sql_query(text(sql), con=engine)
    candidate_trip_ids = candidate_trip_ids.same_geom_candidate.unique()

    candidate_trip_ids = [11766, 11976, 12032, 12192, 12250, 12581, 12754, 12947, 13372,
     13608, 13647, 13812, 13918, 13950, 19658, 19991, 20525, 20527,
     20692, 20699, 20808, 61915, 62529, 62939, 63379, 76565, 77028,
     77140, 78210, 79075, 80102, 1962065, 1530600, 1953353, 1914600]

    pool = mp.Pool(n_threads)
    res = pool.map(exception_wrapper, candidate_trip_ids)
