import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def distance(city, stations):
    lat = city.latitude
    lon = city.longitude

    lat_array = stations.latitude.values
    lon_array = stations.longitude.values

    r = 6371
    t1 = np.deg2rad(lat_array - lat)
    t2 = np.deg2rad(lon_array - lon)
    h1 = (1 - np.cos(t1))/2
    h2 = (1 - np.cos(t2))/2
    inner = np.sqrt(h1 + (np.cos(np.deg2rad(lat))*np.cos(np.deg2rad(lat_array))*h2))
    values = 2 * r * np.arcsin(inner) * 1000

    return np.argmin(values)


def best_tmax(city, stations):
    ind = distance(city, stations)
    return stations.iloc[ind].avg_tmax



#stations_file = sys.argv[1]
#city_data = sys.argv[2]
#f3 = sys.argv[3]
stations_file = "stations.json.gz"
city_data = "city_data.csv"
f3 = "city_data.csv"

stations = pd.read_json(stations_file, lines=True)
stations.avg_tmax = stations.avg_tmax.div(10)

cities = pd.read_csv(city_data, sep=',')
cities = cities.dropna()
cities.area = cities.area.mul(0.000001)
cities = cities.query('area < 10000')
cities['density'] = cities.population/cities.area
cities['best_tmax'] = 0

cities.best_tmax = cities.apply(best_tmax, stations=stations, axis=1)

plt.scatter(cities.best_tmax, cities.density)
plt.title("Temperature vs Population Density")
plt.xlabel("Avg Max Temperature (\u00b0C)")
plt.ylabel("Population Density (people/km\u00b2)")
plt.show()

