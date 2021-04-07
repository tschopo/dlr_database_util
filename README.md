# Util Functions for the DLR Database

## Install

```sh
python3 -m pip install -e .
```
## Dependencies

### Python modules
```
shapely
sqlalchemy
pandas
geopandas
xlsxwriter
numpy
folium

https://github.com/tschopo/elevation_profile
```

### External programs
- [osmium](https://osmcode.org/osmium-tool/manual.html) (not required, only for linux / mac)
- [osm2pgsql](https://osm2pgsql.org/) (for loading osm data into postgres database)
