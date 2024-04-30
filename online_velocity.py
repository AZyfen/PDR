from flask import Flask, request,render_template
from flask_socketio import SocketIO, emit
import json
import numpy as np
from scipy.signal import find_peaks, gaussian, convolve
from datetime import datetime
from threading import Lock

app = Flask(__name__)
socketio = SocketIO(app)

# 初始化全局变量
data_buffer = []  # 用于存储加速度数据的缓冲区
buffer_size = 100  # 设定缓冲区大小为100个数据点
step_length = 0.345  # 步长，单位米
lock = Lock()

def process_data():
    global data_buffer
    # 将缓冲区中的数据转换为 Numpy 数组
    buffer_array = np.array(data_buffer)
    # 应用高斯平滑
    window_size = 10
    gaussian_window = gaussian(window_size, std=2)
    gaussian_window /= np.sum(gaussian_window)
    smoothed_data = convolve(buffer_array[:, 2], gaussian_window, mode='same')
    
    # 峰值检测
    distance_threshold = int(100 * 0.5)
    height_threshold = 0.65
    peaks, _ = find_peaks(smoothed_data, distance=distance_threshold, height=height_threshold)
    total_steps = len(peaks)
    
    # 计算平均速度
    if len(data_buffer) > 1:
        total_time = buffer_array[-1, 3] - buffer_array[0, 3]  # 计算总时间
        total_distance = total_steps * step_length
        average_velocity = total_distance / total_time if total_time > 0 else 0
        socketio.emit('data', {'steps': total_steps, 'velocity': average_velocity})
        data_buffer = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/data", methods=["POST"])
def receive_data():
    global data_buffer
    try:
        data = json.loads(request.data)
        if 'payload' not in data:
            raise KeyError("Missing 'payload' in data")
        
        for item in data['payload',[]]:
            sensor_name = item.get('name')
            sensor_values = item.get('values', {})
            x = sensor_values.get('x')
            y = sensor_values.get('y')
            z = sensor_values.get('z')
            # timestamp = item.get('time')
            timestamp = item.get('time') / 1e9  # 确保时间戳是以秒为单位
            
            if None in [x, y, z, timestamp]:
                raise ValueError("Missing necessary sensor data")
            
            # 将数据添加到缓冲区
            with lock:
                data_buffer.append([x, y, z, timestamp])
                if len(data_buffer) >= buffer_size:
                    # 处理数据
                    total_steps, average_velocity = process_data()
                    print(f"Processed {len(data_buffer)} entries, Total steps: {total_steps}, Average velocity: {average_velocity} m/s")
                    data_buffer = []  # 清空缓冲区以便新的数据收集
                    
            # 可以在这里打印每次添加的数据，以便调试
            print(f"Sensor: {sensor_name}, Time: {datetime.fromtimestamp(timestamp)}, Values: x={x}, y={y}, z={z}")
            
    except json.JSONDecodeError:
        return "Invalid JSON", 400
    except KeyError as e:
        return f"Missing key: {e}", 400
    except ValueError as e:
        return str(e), 400
    except Exception as e:
        return f"Unhandled error: {e}", 500

    return "Success", 200

if __name__ == "__main__":
    socketio.run(app, port=8000, host='0.0.0.0')
