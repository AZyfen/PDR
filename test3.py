from flask import Flask, request
import json
from datetime import datetime
import pandas as pd
from duplicate import main as duplicate_main

server = Flask(__name__)
interval=1
# count=0
start_time = None  # 记录第一个接收到的时间戳
last_ts_relative = None  # 记录上一次接收到的时间戳相对于第一个时间戳的时间差
df = pd.DataFrame(columns=['time', 'x', 'y','z'])   #初始化空的Dataframe

@server.route("/data", methods=["POST"])    #当此路由接收到 POST 请求时，data() 函数将被调用
def data():
    global count,df,start_time,last_ts_relative

    if request.method == "POST":
        # print(f'Received data: {request.data}')
        try:
            data = json.loads(request.data)
            for d in data['payload']:  # 假设每个传感器数据都在payload列表中
                sensor_name = d.get("name")  # 获取传感器名称
                if(sensor_name == "accelerometer"): 
                    global df
                    # count+=1
                    ts = int(d["time"] / 1000000000)  # 时间戳转换
                    sensor_values = d.get("values", {})  # 获取传感器的值
                
                    if start_time is None:
                        start_time = ts
                    ts_relative = ts - start_time  # 相对于第一个时间戳的时间差
                    

                    # 打印传感器名称、时间戳和所有值
                    # print(f'Sensor: {sensor_name}, Time: {ts}, Values: {sensor_values}')
                    # print('---------------------------')
                    mydate={'time':ts_relative,'x':sensor_values['x'],'y':sensor_values['y'],'z':sensor_values['z']}
                    # print(mydate)
                    
                    # a=pd.DataFrame(mydate,index=[0])
                        #将字典数据转换成DataFrame，键对应为列名
                    # df=df.append(pd.DataFrame(mydate,index=[0]),ignore_index=True)   #新添加的数据追加到DataFrame中
                    df = pd.concat([df, pd.DataFrame([mydate])], ignore_index=True)
                    
                    if((ts_relative%interval==0) and last_ts_relative!=None and (ts_relative-last_ts_relative!=0)):#当前的时间差是否是设定的时间间隔的倍数，确保
                        # TODO:发送数据
                        # print('---------------------')
                        # print(df)
                        duplicate_main(df)  # 调用函数处理数据
                        
                        # df.to_csv(f'data_{count//interval}.csv', index=False)  # 保存数据到CSV文件
                        # print('XXXXXXXXXXXXXXXXXXXXXXXXX')
                        # print(f'data_{count//interval}.csv')
                        df=pd.DataFrame(columns=['time', 'x', 'y','z']) #清空df，准备接收新的数据周期
                    last_ts_relative = ts_relative
        
                    
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return "Invalid JSON", 400
        except (KeyError, TypeError) as e:
            print(f"Error processing data: {e}")
            return "Data processing error", 500
    return "Success", 200

if __name__ == "__main__":
    server.run(port=8000, host="0.0.0.0")
