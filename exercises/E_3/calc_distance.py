import sys
from pykalman import KalmanFilter
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np


def output_gpx(points, output_filename):
    """
    Output a GPX file with latitude and longitude from the points DataFrame.
    """
    from xml.dom.minidom import getDOMImplementation

    def append_trkpt(pt, trkseg, doc):
        trkpt = doc.createElement('trkpt')
        trkpt.setAttribute('lat', '%.8f' % (pt['lat']))
        trkpt.setAttribute('lon', '%.8f' % (pt['lon']))
        trkseg.appendChild(trkpt)

    doc = getDOMImplementation().createDocument(None, 'gpx', None)
    trk = doc.createElement('trk')
    doc.documentElement.appendChild(trk)
    trkseg = doc.createElement('trkseg')
    trk.appendChild(trkseg)

    points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)

    with open(output_filename, 'w') as fh:
        doc.writexml(fh, indent=' ')


def distance_helper(data):
    lat1 = data.lat
    lat2 = data.lat2
    lon1 = data.lon
    lon2 = data.lon2

    r = 6371
    t1 = np.deg2rad(lat2 - lat1)
    t2 = np.deg2rad(lon2 - lon1)
    h1 = (1 - np.cos(t1))/2
    h2 = (1 - np.cos(t2))/2
    inner = np.sqrt(h1 + (np.cos(np.deg2rad(lat1))*np.cos(np.deg2rad(lat2))*h2))
    return 2 * r * np.arcsin(inner) * 1000


def distance(data):
    return np.sum(data.apply(distance_helper, axis=1))


def smooth(data):
    kalman_data = data[['lat', 'lon']]
    initial_state = kalman_data.iloc[0]
    observation_covariance = np.diag([.18, .18]) ** 2
    transition = [[1, 0], [0, 1]]
    transition_covariance = np.diag([0.1, 0.1]) ** 2
    kf = KalmanFilter(initial_state_mean=initial_state,
                      initial_state_covariance=observation_covariance,
                      observation_covariance=observation_covariance,
                      transition_covariance=transition_covariance,
                      transition_matrices=transition)
    kalman_smoothed, _ = kf.smooth(kalman_data)

    kalman_df = pd.DataFrame(data=kalman_smoothed[:], columns=['lat', 'lon'])
    kalman_df['lat2'] = kalman_df['lat'].shift(-1)
    kalman_df['lon2'] = kalman_df['lon'].shift(-1)
    df = kalman_df[:-1]
    return df


def get_data(f1):
    tree1 = ET.parse(f1)
    root1 = tree1.getroot()
    dfcols = ["lat", "lon"]
    df = pd.DataFrame(columns=dfcols)

    for child in root1.iter(tag='{http://www.topografix.com/GPX/1/0}trkpt'):
        lat = child.attrib.get("lat")
        lon = child.attrib.get("lon")
        df = df.append(pd.to_numeric(pd.Series([lat, lon], index=dfcols)), ignore_index=True)

    df['lat2'] = df['lat'].shift(-1)
    df['lon2'] = df['lon'].shift(-1)
    df = df[:-1]
    return df


def main():
    points = get_data(sys.argv[1])
    print('Unfiltered distance: %0.2f' % (distance(points),))

    smoothed_points = smooth(points)
    print('Filtered distance: %0.2f' % (distance(smoothed_points),))
    output_gpx(smoothed_points, 'out.gpx')


if __name__ == '__main__':
    main()
