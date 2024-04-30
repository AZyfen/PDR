import pandas as pd
import numpy as np
from scipy.signal import find_peaks, gaussian, convolve
import matplotlib.pyplot as plt

#利用累加的步数来求平均速度

def load_data(filename):
    data = pd.read_csv(filename)
    data['l2_norm'] = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2)
    return data

def process_data(data, start, end):
    sub_data = data[(data['seconds_elapsed'] >= start) & (data['seconds_elapsed'] < end)]
    if len(sub_data) == 0:
        return 0, []
    
    window_size = 10
    gaussian_window = gaussian(window_size, std=2)  #标准差为2的高斯窗口
    gaussian_window /= np.sum(gaussian_window)  #归一化，不会改变数据的总幅度
    smoothed_data = convolve(sub_data['l2_norm'], gaussian_window, mode='same') #数据平滑
    
    step_length = 0.345
    distance_threshold = int(100 * 0.5)
    height_threshold = 0.65
    peaks, _ = find_peaks(smoothed_data, distance=distance_threshold, height=height_threshold)
    
    total_steps = len(peaks)    # 总步数
    total_distance = total_steps * step_length  # 总距离
    average_velocity = total_distance / (end - start) if (end - start) > 0 else 0   # 平均速度

    peak_times = sub_data.iloc[peaks]['seconds_elapsed'].values if len(peaks) > 0 else []   #如果检测到峰值，就返回峰值时间点
    # print(peak_times)

    return average_velocity, total_steps, peak_times

def main():
    filename = 'Accelerometer.csv'
    data = load_data(filename)
    total_time = data['seconds_elapsed'].max()
    current_time = 0
    interval = 1
    
    # 定义全局变量
    window_size = 10
    gaussian_std = 2
    step_length = 0.345
    distance_threshold = int(100 * 0.5)
    height_threshold = 0.65

    velocities = []
    times = []
    all_peak_times = []  # 初始化峰值时间列表

    cumulative_steps = 0
    average_velocities = []
    step_counts = []  # 用于存储每个时间段的步数

    for current_time in np.arange(0, total_time, interval):
        end_time = current_time + interval
        _, steps, _ = process_data(data, current_time, end_time)
        cumulative_steps += steps
        step_counts.append(steps)  # 存储每个时间段的步数
        # 使用累积步数和总时间来计算平均速度
        if current_time > 0:
            average_velocity = (cumulative_steps * step_length) / current_time
        else:
            average_velocity = 0
        average_velocities.append(average_velocity)
    
    # 应用移动平均来平滑速度曲线
    smoothed_velocities = pd.Series(average_velocities).rolling(window=3, min_periods=1, center=True).mean()
    # 计算全数据集的平滑数据
    full_smoothed_data = convolve(data['l2_norm'], gaussian(window_size, std=gaussian_std) / gaussian(window_size, std=gaussian_std).sum(), mode='same')

     # 在平滑后的数据上重新检测峰值
    peaks, _ = find_peaks(full_smoothed_data, distance=distance_threshold, height=height_threshold)
    all_peak_times = data['seconds_elapsed'].iloc[peaks].values  # 使用全局峰值索引获取峰值时间点

    # 绘制速度曲线
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(0, total_time, interval), average_velocities, label='Average Velocity', alpha=0.5)
    plt.plot(np.arange(0, total_time, interval), smoothed_velocities, label='Smoothed Average Velocity', color='orange', linewidth=2)
    plt.axhline(y=0.33, color='r', linestyle='--', label='Groundtruth Velocity')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.legend()
    plt.title('Velocity Over Time')
    # plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(data['seconds_elapsed'], data['l2_norm'], label='Original L2 Norm', alpha=0.5)
    plt.plot(data['seconds_elapsed'], full_smoothed_data, label='Smoothed Data', linewidth=2)
    plt.scatter(all_peak_times, full_smoothed_data[peaks], marker='x', color='red', label='Detected Peaks')
   
    plt.title('Step Detection in Accelerometer Data')
    plt.xlabel('Time (s)')
    plt.ylabel('L2 Norm of Acceleration')
    plt.grid(True)
    plt.legend()
    

    
    plt.show()

    # 打印总步数
    # print(f"Total number of steps detected: {cumulative_steps}")
    print(f"Total number of steps detected: {len(all_peak_times)}")

if __name__ == "__main__":
    main()