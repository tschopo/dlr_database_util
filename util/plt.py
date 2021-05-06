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


def plot_trip_props(maxspeed, electrified, elevation_background, elevation_smoothed, trip_title, trip_length,
                    velocity: Optional[DataFrame] = None, power: Optional[DataFrame] = None,
                    timetable: Optional[DataFrame] = None, x_time: bool = False, color_monotone=None,
                    elevation_plot_height=100, elevation_overshoot=0.15, interactive: bool = False):
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

    chart_maxspeed = plot_maxspeeds(maxspeed, velocity=velocity, x_time=x_time, color=maxspeed_color, hide_x=True,
                                    x_top=True)
    chart_electrified = plot_electrified(electrified, trip_length=trip_length, electrified_color=electrified_color,
                                         not_electrified_color=not_electrified_color, timetable=timetable)

    # add dummy plot due to vega bug
    chart_electrified = (
                alt.Chart(timetable[['dist']]).mark_point(opacity=0).encode(x=alt.X('dist:Q', axis=None)).properties(
                    height=1) + chart_electrified)

    hide_x = False if power is None else True
    chart_elevation = plot_elevation(elevation_background, elevation_smoothed, trip_length=trip_length,
                                     color=elevation_color,
                                     elevation_overshoot=elevation_overshoot, hide_x=hide_x)
    chart_power = None
    if power is not None:
        hide_x = not hide_x
        chart_power = plot_power(power, hide_x=hide_x).properties(width=1000, height=200)

    chart_maxspeed = chart_maxspeed \
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

    chart = chart_electrified & chart_maxspeed & chart_elevation

    chart = chart.properties(title=trip_title).configure_title(
        align='center',
        anchor='middle',
        offset=30)

    if power is not None:
        chart = chart & chart_power

    return chart


def plot_maxspeeds(maxspeed: DataFrame, velocity: Optional[DataFrame] = None, color=None, x_time: bool = False,
                   dist_time_mapping: Optional[DataFrame] = None, hide_x=False, x_top=False) -> alt.Chart:
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
        maxspeed_chart_data.loc[:, "maxspeed"] = maxspeed_chart_data["maxspeed"] + 1
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

    maxspeed_caps_chart = alt.Chart(maxspeed_chart_data).mark_line(opacity=0.75).encode(
        x=alt.X('distance:Q', scale=alt.Scale(nice=False), axis=alt.Axis(format="~s")),
        y='maxspeed',
        color=alt.Color("counter", scale=alt.Scale(domain=[0], range=[color]), legend=None)
    )

    if x_top:
        x_pos = 'top'
    else:
        x_pos = 'bottom'

    if hide_x:
        x = alt.X('distance:Q', axis=alt.Axis(labels=False, ticks=False, tickRound=True),
                  title='',
                  scale=alt.Scale(domain=(0, max(maxspeed.end_dist)), clamp=True, nice=False))
    else:
        x = alt.X('distance:Q', scale=alt.Scale(nice=False), axis=alt.Axis(format="~s", orient=x_pos),
                  title='distance (m)')

    maxspeed_fill_chart = alt.Chart(maxspeed_chart_data).mark_area(
        fill=color,  # "#c6dbef", #"lightgray",
        opacity=0.2,
        interpolate='step',
        line=False
    ).encode(
        x=x,
        y='maxspeed',
        #  y2='velocity'
    )

    if velocity is not None:
        velocity_chart = alt.Chart(velocity).mark_area(line={'color': color, 'opacity': 0.75, 'strokeWidth': 2},
                                                       color=fill_color, opacity=0.5).encode(
            y=alt.Y('velocity:Q', scale=alt.Scale(domain=(0, max(maxspeed_chart_data.maxspeed) + 2), nice=False),
                    axis=alt.Axis(title='v_max, velocity (km/h)')),
            x=x)

        return maxspeed_fill_chart + maxspeed_caps_chart + velocity_chart

    return maxspeed_fill_chart + maxspeed_caps_chart


def plot_elevation(elevation_background: DataFrame, elevation_smoothed: DataFrame, trip_length,
                   color: Optional[str] = None, elevation_overshoot: float = 0.15, hide_x=False) -> alt.Chart:
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

    if hide_x:
        x = alt.X('distance:Q', axis=alt.Axis(labels=False, ticks=False, tickRound=True),
                  title='',
                  scale=alt.Scale(
                      domain=(0, trip_length),
                      clamp=True,
                      nice=False))
    else:
        x = alt.X('distance:Q',
                  axis=alt.Axis(format="~s", title='distance (m)'),
                  scale=alt.Scale(
                      domain=(0, trip_length),
                      nice=False))

    if elevation_background is not None:
        # alt.Chart(elevation_smoothed).mark_area(color=color, opacity=0.2).encode(x='distance:Q', y='elevation:Q') +
        chart = alt.Chart(elevation_background) \
                    .mark_line(color='#ccc') \
                    .encode(
            x=x,
            y=alt.Y('elevation:Q',
                    title='elevation',
                    scale=alt.Scale(
                        domain=(min_ele, max_ele)),
                    axis=alt.Axis(title='elevation (m)')
                    )) \
                + alt.Chart(elevation_smoothed).mark_line(color=color, strokeWidth=2).encode(x='distance:Q',
                                                                                             y='elevation:Q')
    else:
        chart = alt.Chart(elevation_smoothed) \
            .mark_line(color=color) \
            .encode(x=x,
                    y=alt.Y('elevation:Q',
                            title='elevation (m)',
                            scale=alt.Scale(
                                domain=(min_ele, max_ele))))
    return chart


def plot_electrified(electrified: DataFrame, trip_length, electrified_color: Optional[str] = None,
                     not_electrified_color: Optional[str] = None, hide_x=False, timetable: Optional[DataFrame] = None):
    data = {'y': ['electrified'] * electrified.shape[0],
            'electrified': np.where(electrified.electrified.values == 1, 'yes', 'no'),
            'distance': electrified.end_dist - electrified.start_dist, 'start_dist': electrified.start_dist}
    df = pd.DataFrame(data)

    if not_electrified_color is None:
        not_electrified_color = '#ccc'  # '#e41a1c'
    if electrified_color is None:
        electrified_color = '#4daf4a'

    if timetable is not None:
        hide_x = True

    if hide_x:
        x = alt.X('distance:Q',
                  axis=None,
                  scale=alt.Scale(
                      domain=(0, trip_length),
                      clamp=True,
                      nice=False))
    else:
        x = alt.X('distance:Q',
                  axis=alt.Axis(format="~s"),
                  scale=alt.Scale(
                      domain=(0, trip_length),
                      clamp=True,
                      nice=False))

    electrified_chart = alt.Chart(df).mark_bar(yOffset=-2).encode(
        y=alt.Y('y:N', axis=alt.Axis(title='', labels=False, ticks=False)),
        x=x,
        color=alt.Color('electrified:N',
                        scale=alt.Scale(
                            domain=['yes', 'no'],
                            range=[electrified_color, not_electrified_color]),
                        legend=alt.Legend(orient="bottom-right", direction='horizontal', offset=80)
                        ),
        order=alt.Order(
            # Sort the segments of the bars by this field
            'start_dist',
            sort='ascending'
        )
    ).properties(height=4)

    if timetable is not None:
        delay = False
        if 'delay' in timetable.columns:
            cols = ['dist', 'stop_name', 'arrival_time', 'delay']
            delay = True
        else:
            cols = ['dist', 'stop_name', 'arrival_time']

        timetable_chart_data = timetable.copy()[cols]

        if delay:
            timetable_chart_data['delay'] = timetable.delay.dt.total_seconds()
            labels = []
            for delay in timetable_chart_data['delay'].values:
                if delay > 0:
                    labels.append('+' + str(int(delay)) + 's')
                else:
                    labels.append(' ')
            labels = np.array(labels)
            timetable_chart_data['delay_labels'] = labels

        # get dist to last station
        timetable_chart_data["time_label_pos"] = [1 if x % 2 == 1 else 2 for x in timetable.index]
        timetable_chart_data["station_point_pos"] = [0 for x in timetable.index]
        timetable_chart_data["dist_next"] = (timetable_chart_data.shift(-1)["dist"] - timetable_chart_data[
            "dist"]) / trip_length
        timetable_chart_data["time_label_pos"] = [2 if x < 0.05 else 1 for x in timetable_chart_data.dist_next]

        prev_pos = timetable_chart_data.at[0, "time_label_pos"]
        for i in range(1, timetable_chart_data.shape[0]):
            if prev_pos == 2 == timetable_chart_data.at[i, "time_label_pos"]:
                timetable_chart_data.at[i, "time_label_pos"] = 1 if prev_pos == 2 else 2
            prev_pos = timetable_chart_data.at[i, "time_label_pos"]

        timetable_chart_data["time_label_pos"] = 3 - timetable_chart_data["time_label_pos"]

        station_points = alt.Chart(timetable_chart_data).mark_point(color='#333', filled=True, yOffset=-15).encode(
            x='dist:Q', y=alt.Y('station_point_pos', scale=alt.Scale(domain=[0], range=[0]), axis=None))
        station_names = alt.Chart(timetable_chart_data).mark_text(opacity=0.8, align='left', angle=315, dx=15,
                                                                  dy=-10).encode(
            x=alt.X('dist:Q', scale=alt.Scale(nice=False), axis=alt.Axis(format="~s")),
            text=alt.Text('stop_name:N')).properties(width=1000)
        station_times = alt.Chart(timetable_chart_data).mark_text(opacity=0.8, align='center', angle=0, dy=12).encode(
            x=alt.X('dist:Q', scale=alt.Scale(nice=False), axis=alt.Axis(format="~s")),
            y=alt.Y('time_label_pos', axis=None, scale=alt.Scale(nice=False)),
            text=alt.Text('arrival_time:T', timeUnit='hoursminutes')).properties(width=1000, height=20)

        if delay:
            station_delays = alt.Chart(timetable_chart_data).mark_text(color='#d62728', align='center', angle=0,
                                                                       dy=25).encode(
                x=alt.X('dist:Q', scale=alt.Scale(nice=False), axis=alt.Axis(format="~s")),
                y=alt.Y('time_label_pos', axis=None, scale=alt.Scale(nice=False)),
                text=alt.Text('delay_labels:N'),
                color=alt.Color('delay:Q',
                                scale=alt.Scale(scheme='redyellowgreen', reverse=True, domain=[0, 120], clamp=True),
                                legend=None,
                                )
            ).properties(width=1000, height=20)

            electrified_chart = electrified_chart + station_points + station_names + station_times + station_delays
        else:
            electrified_chart = electrified_chart + station_points + station_names + station_times

        electrified_chart = electrified_chart + station_points + station_names + station_times

    return electrified_chart


def plot_power(power: DataFrame, pos_color='#9ecae1', neg_color='#d62728', hide_x=False):
    if hide_x:
        x = alt.X('distance:Q', axis=alt.Axis(labels=False, ticks=False, tickRound=True),
                  title='',
                  scale=alt.Scale(domain=(0, max(power.distance)), clamp=True, nice=False))
    else:
        x = alt.X('distance:Q', scale=alt.Scale(nice=False), axis=alt.Axis(format="~s", title='distance (m)'))

    # TODO for nice display: calculate the intersection with x=0 and add points at the intersection so that no overlapp
    #  between negative chart and positive chart

    chart_power = alt.Chart(power).transform_calculate(
        negative='datum.power > 0'
    ).mark_area(opacity=0.75).encode(
        x=x,
        y=alt.Y('power:Q', impute={'value': 0},
                scale=alt.Scale(nice=False, domain=[power.power.min() * 0.75, power.power.max()], clamp=False),
                axis=alt.Axis(format="~s", title='power (W)')),
        color=alt.Color('negative:N', legend=None, scale=alt.Scale(domain=[False, True], range=[neg_color, pos_color]))
    )
    return chart_power
