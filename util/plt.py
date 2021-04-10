"""
Functions that interface with the OSM rail database as defined by the import script
"""

from typing import Optional

import altair as alt
import folium
import numpy as np
import pandas as pd
from ElevationSampler import DEM
from geopandas import GeoDataFrame
from pandas import DataFrame


def plot_osm(osm_data: GeoDataFrame, prop: Optional[str] = None):
    """
    Plot osm data on map.

    Parameters
    ----------
    osm_data
    prop
    dem

    Returns
    -------

    """
    osm_data_map = osm_data.to_crs(4326)

    if prop == "maxspeed":

        # pep doesnt like lambdas
        def colormap(x):
            return '#aaa'

        if not osm_data_map['maxspeed'].isna().all():
            colormap = folium.LinearColormap(colors=["red", "orange", "yellow", "green"],
                                             vmin=osm_data_map['maxspeed'].min(),
                                             vmax=osm_data_map['maxspeed'].max()).to_step(
                n=len(osm_data_map["maxspeed"].unique()))
        osm_data_map['maxspeed'] = osm_data_map['maxspeed'].fillna(-9999)
    elif prop == "electrified":

        def colormap(x):
            if x == 0:
                return '#d62728'
            elif x == 1:
                return '#2ca02c'
            else:
                return '#aaa'

        osm_data_map['electrified'] = np.where(osm_data_map['electrified'] == 'yes', 1, osm_data_map['electrified'])
        osm_data_map['electrified'] = np.where(osm_data_map['electrified'] == 'no', 0, osm_data_map['electrified'])
        osm_data_map['electrified'] = np.where((osm_data_map['electrified'] == 1) | (osm_data_map['electrified'] == 0),
                                               osm_data_map['electrified'], -9999)

    osm_json = osm_data_map[
        ["electrified", "maxspeed", "maxspeed_forward", "maxspeed_backward", "bridge", "tunnel", "geom",
         "start_point_distance", "end_point_distance"]].to_json(na='keep')

    m = folium.Map(location=[osm_data_map.total_bounds[[1, 3]].mean(), osm_data_map.total_bounds[[0, 2]].mean()],
                   # tiles='Stamen Terrain',
                   zoom_start=9)

    folium.GeoJson(osm_json, name="geojson",
                   style_function=lambda x: {
                       'color': colormap(x['properties'][prop]) if x['properties'][prop] >= 0 else '#aaa',
                       'weight': 2.5
                   },
                   tooltip=folium.features.GeoJsonTooltip(
                       fields=['maxspeed', 'maxspeed_forward', 'maxspeed_backward', 'electrified',
                               'start_point_distance', 'end_point_distance'],
                       # aliases=['Max Speed', 'Electrified'],
                       labels=True,
                       sticky=True,
                       toLocaleString=True)
                   ).add_to(m)

    return m


def plot_trip_props(maxspeed, electrified, elevation_background, elevation_smoothed, trip_title, color_monotone=None,
                    elevation_plot_height=100):
    # ideas: add stations to elevation and maxspeed (points on the line)
    # add timetable plot (like electrified, with station names and color is the duration between the stations?)
    # interactivity: scrolling, zooming, highlighting
    # add geometry, linked circle of cursor position on all plots

    maxspeed_color = None
    electrified_color = None
    not_electrified_color = None
    elevation_color = None
    if color_monotone is not None:
        maxspeed_color = color_monotone
        electrified_color = color_monotone
        not_electrified_color = '#ccc'
        elevation_color = color_monotone

    chart_maxspeed = plot_maxspeeds(maxspeed, color=maxspeed_color)
    chart_electrified = plot_electrified(electrified, electrified_color=electrified_color,
                                         not_electrified_color=not_electrified_color)
    chart_elevation = plot_elevation(elevation_background, elevation_smoothed, color=elevation_color)

    chart_maxspeed = chart_maxspeed \
        .encode(
        x=alt.X('distance:Q', axis=alt.Axis(labels=False, ticks=False, tickRound=True),
                title='',
                scale=alt.Scale(domain=(0, max(chart_maxspeed.data.distance)), clamp=True, nice=False))) \
        .properties(width=1000, height=100)

    chart_electrified = chart_electrified.encode(x=alt.X('distance:Q', axis=None,
                                                         title='',
                                                         scale=alt.Scale(domain=(0, max(electrified.end_dist)),
                                                                         clamp=True,
                                                                         nice=False))).properties(width=1000, height=6)

    chart_elevation = chart_elevation.properties(width=1000, height=elevation_plot_height)

    chart = chart_electrified & chart_maxspeed & chart_elevation
    chart = chart.properties(title=trip_title).configure_title(
        align='center',
        anchor='middle',
        offset=30)

    return chart


def plot_maxspeeds(maxspeed: DataFrame, color=None) -> alt.Chart:
    """

    Parameters
    ----------
    maxspeed : DataFrame
        cols start_dist, maxspeed must be present

    Returns
    -------

    """
    maxspeeds = []
    start_dists = []

    # plot horizontal lines: duplicate the values and add start and end for the x values
    for i in range(maxspeed.shape[0] - 1):
        maxspeeds.append(maxspeed.iloc[i]["maxspeed"])
        maxspeeds.append(maxspeed.iloc[i]["maxspeed"])

        start_dists.append(maxspeed.iloc[i]["start_dist"])
        start_dists.append(maxspeed.iloc[i + 1]["start_dist"])

    # for last value add end_dist as end
    maxspeeds.append(maxspeed.iloc[-1]["maxspeed"])
    start_dists.append(maxspeed.iloc[-1]["start_dist"])
    maxspeeds.append(maxspeed.iloc[-1]["maxspeed"])
    start_dists.append(maxspeed.iloc[-1]["end_dist"])

    maxspeed_chart_data = pd.DataFrame({"maxspeed": maxspeeds, "distance": start_dists})

    # ff7f00
    if color is None:
        color = '#377eb8'
    chart = alt.Chart(maxspeed_chart_data) \
        .mark_line(color=color) \
        .encode(x=alt.X('distance:Q',
                        scale=alt.Scale(
                            domain=(0, max(maxspeed_chart_data.distance)),
                            clamp=True,
                            nice=False),
                        axis=alt.Axis(format="~s")),
                y=alt.Y('maxspeed:Q',
                        scale=alt.Scale(domain=(
                            0, 150)))
                )

    return chart


def plot_elevation(elevation_background: DataFrame, elevation_smoothed: DataFrame,
                   color: Optional[str] = None) -> alt.Chart:
    """

    Parameters
    ----------
    elevation_background : DataFrame
        columns 'distance', 'elevation'

    elevation_smoothed : DataFrame
        columns 'distance', 'elevation'

    Returns
    -------

    """
    if color is None:
        color = '#a65628'

    min_ele = max(0, min(elevation_background.elevation) - 50)
    max_ele = max(elevation_background.elevation)
    max_ele = max_ele - ((max_ele - min_ele) * 0.2)

    chart = alt.Chart(elevation_background) \
                .mark_line(color='#ccc') \
                .encode(
        x=alt.X('distance:Q',
                axis=alt.Axis(format="~s"),
                scale=alt.Scale(
                    domain=(0, max(elevation_background.distance)),
                    clamp=True,
                    nice=False)),
        y=alt.Y('elevation:Q',
                title='elevation',
                scale=alt.Scale(
                    domain=(min_ele, max_ele)))) \
            + alt.Chart(elevation_smoothed).mark_line(color=color).encode(x='distance:Q', y='elevation:Q')

    return chart


def plot_electrified(electrified: DataFrame, electrified_color: Optional[str] = None,
                     not_electrified_color: Optional[str] = None):
    data = {'y': ['electrified'] * electrified.shape[0],
            'electrified': np.where(electrified.electrified.values == 1, 'yes', 'no'),
            'distance': electrified.end_dist - electrified.start_dist, 'start_dist': electrified.start_dist}
    df = pd.DataFrame(data)

    if not_electrified_color is None:
        not_electrified_color = '#ccc'  # '#e41a1c'
    if electrified_color is None:
        electrified_color = '#4daf4a'

    chart = alt.Chart(df).mark_bar().encode(
        y=alt.Y('y:N', axis=alt.Axis(title='', labels=False, ticks=False)),
        x=alt.X('distance:Q',
                axis=alt.Axis(format="~s"),
                scale=alt.Scale(
                    domain=(0, max(electrified.end_dist)),
                    clamp=True,
                    nice=False)),
        color=alt.Color('electrified:N',
                        scale=alt.Scale(
                            domain=['yes', 'no'],
                            range=[electrified_color, not_electrified_color])),
        order=alt.Order(
            # Sort the segments of the bars by this field
            'start_dist',
            sort='ascending'
        )
    )
    return chart
