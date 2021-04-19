"""
Functions for plotting trip data
"""

from typing import Optional

import altair as alt
import folium
import numpy as np
import pandas as pd
from folium import Map
from geopandas import GeoDataFrame
from pandas import DataFrame


def plot_osm(osm_data: GeoDataFrame, prop: Optional[str] = None) -> Map:
    """
    Plot osm data on map.

    Parameters
    ----------
    osm_data
    prop

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
    else:
        def colormap(x):
            return '#d62728'
        prop = "maxspeed"

    osm_data_map["brunnel"] = np.where((osm_data_map['bridge'] == 'yes') | (osm_data_map['tunnel'] == 'yes'),
                                       'yes', 'no')

    osm_json = osm_data_map[
        ["electrified", "maxspeed", "maxspeed_forward", "maxspeed_backward", "brunnel", "geom",
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
                       fields=['maxspeed', 'maxspeed_forward', 'maxspeed_backward', 'electrified', 'brunnel',
                               'start_point_distance', 'end_point_distance'],
                       # aliases=['Max Speed', 'Electrified'],
                       labels=True,
                       sticky=True,
                       toLocaleString=True)
                   ).add_to(m)

    return m


def plot_trip_props(maxspeed, electrified, elevation_background, elevation_smoothed, trip_title,
                    velocity: Optional[DataFrame] = None, power: Optional[DataFrame] = None,
                    timetable: Optional[DataFrame] = None, x_time: bool = False, color_monotone=None,
                    elevation_plot_height=100, elevation_overshoot=0.15, interactive: bool = True):
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

    if interactive:
        elevation_overshoot = 0

    chart_maxspeed = plot_maxspeeds(maxspeed, velocity=velocity, x_time=x_time, color=maxspeed_color)
    chart_electrified = plot_electrified(electrified, electrified_color=electrified_color,
                                         not_electrified_color=not_electrified_color)
    chart_elevation = plot_elevation(elevation_background, elevation_smoothed, color=elevation_color,
                                     elevation_overshoot=elevation_overshoot)
    chart_power = None
    if power is not None:
        chart_power = plot_power(power)


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

    if interactive:
        zoom = alt.selection_interval(bind='scales')
        chart_elevation = chart_elevation.add_selection(zoom)
        chart_electrified = chart_electrified.add_selection(zoom)
        chart_maxspeed = chart_maxspeed.add_selection(zoom)
        
        if chart_power is not None:
            chart_power = chart_power.add_selection(zoom)

    if chart_power is None:
        chart = chart_electrified & chart_maxspeed & chart_elevation
    else:
        chart = chart_electrified & chart_maxspeed & chart_elevation & chart_power

    chart = chart.properties(title=trip_title).configure_title(
        align='center',
        anchor='middle',
        offset=30)

    return chart


def plot_maxspeeds(maxspeed: DataFrame, velocity: Optional[DataFrame] = None, color=None, x_time: bool = False,
                   dist_time_mapping: Optional[DataFrame] = None) -> alt.Chart:
    """

    Parameters
    ----------
    maxspeed : DataFrame
        cols start_dist, maxspeed must be present
    color

    Returns
    -------

    """

    maxspeeds = []
    start_dists = []
    cat = []

    # plot horizontal lines: duplicate the values and add start and end for the x values
    c = 0
    for i in range(maxspeed.shape[0] - 1):
        maxspeeds.append(maxspeed.iloc[i]["maxspeed"])
        maxspeeds.append(maxspeed.iloc[i]["maxspeed"])

        start_dists.append(maxspeed.iloc[i]["start_dist"])
        start_dists.append(maxspeed.iloc[i + 1]["start_dist"])
        cat.append(c)
        cat.append(c)
        c += 1

    # for last value add end_dist as end
    maxspeeds.append(maxspeed.iloc[-1]["maxspeed"])
    start_dists.append(maxspeed.iloc[-1]["start_dist"])
    maxspeeds.append(maxspeed.iloc[-1]["maxspeed"])
    start_dists.append(maxspeed.iloc[-1]["end_dist"])
    cat.append(c)
    cat.append(c)
    maxspeed_chart_data = pd.DataFrame({"maxspeed": maxspeeds, "distance": start_dists, "counter": cat})

    if x_time:
        # TODO
        pass

    # separate the velocity lines from the maxspeed lines
    if velocity is not None:
        maxspeed_chart_data["maxspeed"] = maxspeed_chart_data["maxspeed"] + 1
        velocity.loc[:, "velocity"] = velocity["velocity"] - 1

        # TODO
        #  make that the x-values in maxspeed and velocity align
        #  if distance, resample velocity distance so that equidistant
        #  snap maxspeed x-values to next velocity x-value

    if color is None:
        color = '#4c78a8'  # '#377eb8'
        fill_color = '#9ecae1'
    else:
        fill_color = color

    maxspeed_caps_chart = alt.Chart(maxspeed_chart_data).mark_line(opacity=0.5).encode(
        x=alt.X('distance:Q', scale=alt.Scale(nice=False), axis=alt.Axis(format="~s")),
        y='maxspeed',
        color=alt.Color("counter", scale=alt.Scale(domain=[0], range=[color]), legend=None)
    )

    maxspeed_fill_chart = alt.Chart(maxspeed_chart_data).mark_area(
        fill=fill_color,  # "#c6dbef", #"lightgray",
        opacity=0.5,
        interpolate='step',
        line=False
    ).encode(
        x=alt.X('distance:Q', scale=alt.Scale(nice=False), axis=alt.Axis(format="~s")),
        y='maxspeed',
        #  y2='velocity'
    )

    if velocity is not None:
        velocity_chart = alt.Chart(velocity).mark_area(line={'color':'#4c78a8'}, color="#4c78a8", opacity=0.2).encode(
            y=alt.Y('velocity:Q', scale=alt.Scale(domain=(0, max(maxspeed_chart_data.maxspeed)+2), nice=False)),
            x=alt.X('distance:Q', scale=alt.Scale(nice=False), axis=alt.Axis(format="~s")))

        return maxspeed_fill_chart + maxspeed_caps_chart + velocity_chart

    return maxspeed_fill_chart + maxspeed_caps_chart


def plot_elevation(elevation_background: DataFrame, elevation_smoothed: DataFrame,
                   color: Optional[str] = None, elevation_overshoot: float = 0.15) -> alt.Chart:
    """

    Parameters
    ----------
    elevation_background : DataFrame
        columns 'distance', 'elevation'
    elevation_smoothed : DataFrame
        columns 'distance', 'elevation'
    color

    Returns
    -------

    """
    if color is None:
        color = '#a65628'

    if elevation_background is not None:
        min_ele = max(0, min(elevation_background.elevation) - 50)
        max_ele = max(elevation_background.elevation)
        max_ele = max_ele - ((max_ele - min_ele) * elevation_overshoot)
    else:
        min_ele = max(0, min(elevation_smoothed.elevation) - 50)
        max_ele = max(elevation_smoothed.elevation)
        max_ele = max_ele - ((max_ele - min_ele) * elevation_overshoot)

    if elevation_background is not None:
        chart = alt.Chart(elevation_background) \
                    .mark_line(color='#ccc') \
                    .encode(
            x=alt.X('distance:Q',
                    axis=alt.Axis(format="~s"),
                    scale=alt.Scale(
                        domain=(0, max(elevation_background.distance)),
                        nice=False)),
            y=alt.Y('elevation:Q',
                    title='elevation',
                    scale=alt.Scale(
                        domain=(min_ele, max_ele)))) \
                + alt.Chart(elevation_smoothed).mark_line(color=color).encode(x='distance:Q', y='elevation:Q')
    else:
        chart = alt.Chart(elevation_smoothed) \
            .mark_line(color=color) \
            .encode(x=alt.X('distance:Q',
                            axis=alt.Axis(format="~s"),
                            scale=alt.Scale(
                                domain=(0, max(
                                    elevation_smoothed.distance)),
                                clamp=True,
                                nice=False)),
                    y=alt.Y('elevation:Q',
                            title='elevation',
                            scale=alt.Scale(
                                domain=(min_ele, max_ele))))
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


def plot_power(power: DataFrame, pos_color='#9ecae1', neg_color='#d62728'):
    chart_power = alt.Chart(power).transform_calculate(
        negative='datum.power > 0'
    ).mark_area(opacity=0.5).encode(
        x=alt.X('distance:Q',
                scale=alt.Scale(
                    domain=(0, max(power.distance)),
                    clamp=True,
                    nice=False),
                axis=alt.Axis(format="~s")),
        y=alt.Y('power', impute={'value': 0},
                scale=alt.Scale(nice=False, domain=[-15000000,10000000], clamp=False)),
        color=alt.Color('negative:N', legend=None, scale=alt.Scale(domain=[True, False], range=[pos_color, neg_color]))
    )
    return chart_power
