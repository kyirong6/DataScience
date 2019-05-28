import sys
import pandas as pd
import numpy as np
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess


def to_timestamp(x):
    return x.timestamp()


# ----------------LOESS SMOOTHING------------------
#f1 = sys.argv[1]
f1 = "sysinfo.csv"
cpu_data = pd.read_csv(f1, parse_dates=['timestamp'])
cpu_data['timestamp'] = cpu_data['timestamp'].apply(to_timestamp)

plt.figure(figsize=(12, 4))
plt.plot(cpu_data['timestamp'], cpu_data['temperature'], 'b.', alpha=0.5)

loess_smoothed = lowess(cpu_data.temperature, cpu_data.timestamp, .085)
plt.plot(cpu_data['timestamp'], loess_smoothed[:, 1], 'r-', label="LOESS-smoothed line")


# ------------------KALMAN FILTERING---------------------
kalman_data = cpu_data[['temperature', 'cpu_percent', 'sys_load_1']]
print(kalman_data)
kalman_sd = kalman_data.std()

initial_state = kalman_data.iloc[0]
observation_covariance = np.diag([kalman_sd[0], kalman_sd[1], kalman_sd[2]]) ** 2
transition_covariance = np.diag([0.1, 0.1, 0.1]) ** 2
transition = [[1, -1, .7], [0, .6, .03], [0, 1.3, .8]]

kf = KalmanFilter(initial_state_mean=initial_state,
                  initial_state_covariance=observation_covariance,
                  observation_covariance=observation_covariance,
                  transition_covariance=transition_covariance,
                  transition_matrices=transition)
kalman_smoothed, _ = kf.smooth(kalman_data)
plt.plot(cpu_data['timestamp'], kalman_smoothed[:, 0], 'g-', label="Kalman-smoothed line")
plt.legend()
#plt.savefig('cpu.svg')
plt.show()
