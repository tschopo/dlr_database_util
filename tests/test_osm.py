import util
from sqlalchemy import create_engine
from ElevationSampler import DEM, ElevationProfile

if __name__ == '__main__':
    engine = create_engine('postgresql://dlr:dlr4thewin@localhost/liniendatenbank')
    dem = DEM("../../../data/dem/de.tif")
    trip_id = 585090

    trip_generator = util.TripGenerator(dem, engine)
    trip = trip_generator.generate_from_railway_db(trip_id)
    trip.add_simulation_results('../../../code/TPT_inputs/results_3/1075937_bad_tuerkheim/1075937_resultsPerSeconds.csv')