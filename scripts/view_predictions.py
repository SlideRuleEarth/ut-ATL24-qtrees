import argparse
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def main(args):

    # Show args
    if args.verbose:
        print('input_filename:', args.input_filename)

    st.set_page_config(
            page_title=args.input_filename,
            page_icon=":earth_americas:")

    df = pd.read_csv(args.input_filename, engine="python")

    if args.verbose:
        print(df)

    minx = np.amin(df.along_track_dist)
    maxx = np.amax(df.along_track_dist)
    totalx = maxx - minx
    total_segments = int(totalx)//args.segment_size + 1
    st.text(f'{args.input_filename}')
    st.text(f'min {minx}m, max {maxx}m, len {totalx}m')
    st.text(f'segment_size {args.segment_size} \
              total segments {total_segments}')
    segment_numbers = range(0, total_segments)
    segment_number = st.selectbox('Segment', segment_numbers)

    # Get start and end photon indexes
    x1 = minx + segment_number*args.segment_size
    x2 = minx + (segment_number + 1)*args.segment_size
    st.text(f'start x {x1}, end x {x2}')
    df2 = df[(df.along_track_dist > x1) & (df.along_track_dist < x2)]
    aspect_ratio = st.slider('Aspect ratio (0=auto)', 0, 100, 10)

    # Use discrete colormap
    df2 = df2.astype({"manual_label": int})
    df2 = df2.astype({"manual_label": str})
    df2 = df2.astype({"prediction": int})
    df2 = df2.astype({"prediction": str})
    cdm = {'0': 'darkgray',
           '1': 'black',
           '2': '#8B4513',
           '3': 'green',
           '4': 'green',
           '5': 'green',
           '6': 'gray',
           '7': 'darkgray',
           '9': 'cyan',
           '10': 'purple',
           '11': 'lightgray',
           '13': 'yellow',
           '14': 'yellow',
           '15': 'gray',
           '17': 'gray',
           '18': 'gray',
           '20': 'black',
           '21': 'black',
           '22': 'black',
           '23': 'black',
           '24': 'black',
           '40': 'magenta',
           '41': 'cyan',
           '45': 'blue'}
    label_names = {'0': 'other',
                   '1': 'unclassified',
                   '2': 'ground',
                   '3': 'low-veg',
                   '4': 'med-veg',
                   '5': 'high-veg',
                   '6': 'building',
                   '7': 'noise',
                   '9': 'water',
                   '10': 'rail',
                   '11': 'road',
                   '13': 'wire',
                   '14': 'wire',
                   '15': 'tower',
                   '17': 'reserved',
                   '18': 'user-defined',
                   '20': 'user-defined',
                   '21': 'user-defined',
                   '22': 'user-defined',
                   '23': 'user-defined',
                   '24': 'user-defined',
                   '40': 'bathymetry',
                   '41': 'sea surface',
                   '45': 'water column'}

    fig = px.scatter(df2, x="along_track_dist", y="geoid_corrected_h",
                     color='manual_label',
                     color_discrete_map=cdm,
                     hover_data=["along_track_dist", "geoid_corrected_h"],
                     labels={'along_track_dist': 'Along-track distance (m)',
                             'geoid_corrected_h': 'Elevation (m)'})
    fig.for_each_trace(lambda t: t.update(name=label_names[t.name]))
    if aspect_ratio != 0:
        fig.update_yaxes(scaleanchor="x", scaleratio=aspect_ratio)
    fig.update_layout(template='simple_white')
    fig.update_scenes(aspectmode='data')
    fig.update_traces(marker=dict(opacity=1.0, line=dict(width=0)))
    fig.update_traces(marker_size=3)
    fig.update_layout(legend_title_text='Classification')
    fig.update_layout(legend_traceorder='normal')
    st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(df2, x="along_track_dist", y="geoid_corrected_h",
                     color='prediction',
                     color_discrete_map=cdm,
                     hover_data=["along_track_dist", "geoid_corrected_h"],
                     labels={'along_track_dist': 'Along-track distance (m)',
                             'geoid_corrected_h': 'Elevation (m)'})
    fig.for_each_trace(lambda t: t.update(name=label_names[t.name]))
    fig.add_scatter(x=df2["along_track_dist"], y=df2["sea_surface_h"],
                    mode="markers",
                    marker=dict(size=10, color="green"),
                    name='surface est.')
    fig.add_scatter(x=df2["along_track_dist"], y=df2["bathy_h"],
                    mode="markers",
                    marker=dict(size=10, color="blue"),
                    name='bathy est.')

    if aspect_ratio != 0:
        fig.update_yaxes(scaleanchor="x", scaleratio=aspect_ratio)
    fig.update_layout(template='simple_white')
    fig.update_scenes(aspectmode='data')
    fig.update_traces(marker=dict(opacity=1.0, line=dict(width=0)))
    fig.update_traces(marker_size=3)
    fig.update_layout(legend_title_text='Classification')
    fig.update_layout(legend_traceorder='normal')
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='viewer',
        description='Interactive photon viewer')
    parser.add_argument(
        'input_filename',
        help="Input filename specification")
    parser.add_argument(
        '-s', '--segment_size', type=int, default=10000,
        help='Size in meters of one segment')
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help="Show verbose output")
    args = parser.parse_args()

    main(args)
