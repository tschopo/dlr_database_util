from setuptools import setup

setup(name='util',
      version='0.2',
      description='Util functions for the DLR database',
      url='#',
      author='Johannes Polster',
      author_email='johannes.polster@posteo.de',
      license='',
      packages=['util'],
      install_requires=[
          'sqlalchemy',
          'geopandas',
          'pandas',
          'numpy',
          'pyproj',
          'shapely',
          'xlsxwriter',
          'folium',
          'altair'
          # 'ElevationSampler @ https://github.com/tschopo/elevation_profile/blob/main/dist/ElevationSampler-1.0.tar.gz'
      ],
      zip_safe=True)
