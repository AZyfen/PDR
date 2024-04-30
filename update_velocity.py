import pandas as pd
import numpy as np
from scipy.signal import find_peaks, gaussian, convolve
import matplotlib.pyplot as plt
import time

class StepCounter:
    def __init__(self, filename):
        self.filename = filename
        self.last_timestamp = None
        self.total_distance = 0
        self.total_time = 0
        self.total_steps = 0
        self.time_stamps = []
        self.estimated_speeds = []

    def load_data(self):
        try:
            data = pd.read_csv(self.filename)
            # Convert timestamps from nanoseconds to seconds
            data['time'] = pd.to_numeric(data['time'], errors='coerce')
            data['time'] /= 1e9
            data['l2_norm'] = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2)
            
            # Only process new data
            if self.last_timestamp is not None:
                data = data[data['time'] > self.last_timestamp]
                
            self.last_timestamp = data['time'].max()
            return data
        except FileNotFoundError:
            print("File not found. Skipping this interval.")
            return pd.DataFrame()

    def smooth_data(self, data, window_size=10, use_gaussian=True):
        if use_gaussian:
            gaussian_window = gaussian(window_size, std=2)
            gaussian_window /= np.sum(gaussian_window)
            return convolve(data, gaussian_window, mode='same')
        else:
            return data.rolling(window=window_size, min_periods=1, center=True).mean()

    def update(self):
        data = self.load_data()
        if data.empty:
            return

        smoothed_data = self.smooth_data(data['l2_norm'])
        self.detect_and_update(data, smoothed_data)

    def detect_and_update(self, data, smoothed_data):
        distance_threshold = int(100 * 0.5)
        height_threshold = 1
        
        peaks, _ = find_peaks(smoothed_data, distance=distance_threshold, height=height_threshold)
        steps_detected = len(peaks)
        
        if steps_detected > 0:
            new_time = data['time'].iloc[peaks].max() - data['time'].iloc[peaks].min()
            step_length = 0.60  # meters
            new_distance = steps_detected * step_length
            self.total_distance += new_distance
            self.total_time += new_time
            self.total_steps += steps_detected
            self.time_stamps.append(time.time())
            self.estimated_speeds.append(self.get_average_velocity())

    def get_average_velocity(self):
        if self.total_time > 0:
            return self.total_distance / self.total_time
        else:
            return 0

    def run(self):
        plt.ion()
        fig, ax = plt.subplots()
        try:
            while True:
                self.update()
                ax.clear()
                ax.plot(self.time_stamps, self.estimated_speeds, label='Estimated Speed (m/s)')
                ax.axhline(y=0.6, color='r', linestyle='-', label='Actual Speed (0.6 m/s)')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Speed (m/s)')
                ax.legend()
                plt.draw()
                plt.pause(0.01)
                time.sleep(5)  # Update every 10 seconds
        except KeyboardInterrupt:
            print("Stopping step counter.")
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    step_counter = StepCounter('accelerometer.csv')
    step_counter.run()
