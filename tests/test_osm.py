import util
from sqlalchemy import create_engine
from ElevationSampler import DEM, ElevationProfile

if __name__ == '__main__':
    engine = create_engine('postgresql://dlr:dlr4thewin@localhost/liniendatenbank')
    dem = DEM("../../../data/dem/de.tif")
    railway_db = util.RailwayDatabase(engine)
    trip_id = 1075937
    trip = util.Trip(trip_id, railway_db, dem)
    trip.add_simulation_results('../../../code/TPT_inputs/results_3/1075937_bad_tuerkheim/1075937_resultsPerSeconds.csv')