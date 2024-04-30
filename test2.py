from flask import Flask, request
import json
from datetime import datetime

server = Flask(__name__)

@server.route("/data", methods=["POST"])
def data():
    if request.method == "POST":
        print(f'Received data: {request.data}')
        # quit()
        try:
            data = json.loads(request.data)
            for d in data['payload']:  # 假设每个传感器数据都在payload列表中
                sensor_name = d.get("name")  # 获取传感器名称
                ts = datetime.fromtimestamp(d["time"] / 1000000000)  # 时间戳转换
                sensor_values = d.get("values", {})  # 获取传感器的值
                
                # 打印传感器名称、时间戳和所有值
                print(f'Sensor: {sensor_name}, Time: {ts}, Values: {sensor_values}')
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return "Invalid JSON", 400
        except (KeyError, TypeError) as e:
            print(f"Error processing data: {e}")
            return "Data processing error", 500
    return "Success", 200

if __name__ == "__main__":
    server.run(port=8000, host="0.0.0.0")
