from sqlalchemy import create_engine, text
import pandas as pd
import util
from datetime import datetime, timedelta


from scripts.database_config import engine_config

import sys


def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()


if __name__ == '__main__':

    # geogtfs start: 02.10.2020
    # geogts end: 12.12.2020
    # KW41 05.10 - 11.10 x
    # KW42 12.10. - 18.10
    # KW43 19.10. - 25.10 x
    # KW44 26.10. - 01.11 x
    # KW45 02.11. - 08.11 x

    gtfs_start = 20201002
    gtfs_end = 20201212

    engine = create_engine(engine_config)
    railway_db = util.RailwayDatabase(engine)

    sql = """
        select distinct trip_id
        from geo_trips
        order by trip_id
        """

    trip_ids = pd.read_sql_query(text(sql), con=engine)
    trip_ids = trip_ids.trip_id.values

    kw_data = []
    trip_id_data = []
    n_fahrten_data = []

    error_ids = []

    kw = 46
    period_start = 20201109  # gtfs_start
    while period_start < gtfs_end:
        print('starting KW' + str(kw))

        for i, trip_id in enumerate(trip_ids):
            try:
                n_fahrten = railway_db.calculate_n_fahrten_in_period(int(trip_id), period_start)
                kw_data.append(kw)
                trip_id_data.append(trip_id)
                n_fahrten_data.append(n_fahrten)
                progress(i, trip_ids.shape[0])
            except Exception as err:
                print('ERROR')
                print(err)
                print(trip_id)
                error_ids.append(trip_id)

        next_date = datetime.strptime(str(period_start), '%Y%m%d') + timedelta(days=7)
        period_start = int(next_date.strftime('%Y%m%d'))
        print('finished. next:' + next_date.strftime('%Y%m%d'))
        kw += 1

    data = {'trip_id': trip_id_data, 'kw': kw_data, 'n_trips': n_fahrten_data}
    df = pd.DataFrame(data)

    with engine.connect() as con:
        df.to_sql('calc_n_trips_per_period', con, if_exists='append', index=False)

