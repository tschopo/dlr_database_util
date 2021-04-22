import util
from sqlalchemy import create_engine
from ElevationSampler import DEM, ElevationProfile

if __name__ == '__main__':
    engine = create_engine('postgresql://dlr:dlr4thewin@localhost/liniendatenbank')
    dem = DEM("../../../data/dem/de.tif")
    railway_db = util.RailwayDatabase(engine)
    trip_id = 1485062
    trip = util.Trip(trip_id, railway_db, dem)
