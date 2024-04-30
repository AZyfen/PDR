#主要是从传感器数据中提取和利用步长信息，进行校准并可视化步行数据
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import plotlyHelper
from scipy.signal import find_peaks
import itertools
#定义一个列表，包含要处理的传感器类型                                           
sensor_types = ['Accelerometer', 'Gyroscope', 'Gravity', 'Barometer', 'Orientation']

#定义一个step类,这个类包含了步长、幅度、真实大小、所有者、原始信号和增益
class Step:
    def __init__(self, owner, length, mag, true_size=None):
        self.length = length
        self.magnitude = mag
        self.true_size = true_size
        self.approx_size = None
        self.owner = owner
        self.raw_signal = None
        self.Gk = None
        if self.true_size is not None:
            self.calc_gain()

    #计算增益的方法
    #增益是用来补偿传感器信号的不准确性，校正步长估计
    def calc_gain(self):            
        self.Gk = self.true_size / (np.cbrt(self.magnitude / self.length))

#接受一个列表的列表并将其展平成一个单一的列表，作用：合并多个列表为一个列表
def flatten_list(list_of_lists):
    return list(itertools.chain(*list_of_lists))


# ------------------------------------------------------ Loading utils ---------------------------------------------
#作用：加载传感器数据并进行预处理，包括读取数据、排序、处理时间戳、去除数据的开始和结束部分等操作，以便后续的分析和处理
#从csv文件中加载传感器数据，并对加速度计数据的偏差校准，并计算采样频率，返回值是一个字典，包含每个传感器的数据和采样频率
def load_session(dirname, remove_ends_seconds=0, biases=False):
    #初始化两个字典，分别用于存储传感器数据和采样频率
    print('开始加载数据,调用load_session函数')
    sensor_data = {}
    fs = {}
    #对于每个传感器类型，构建csv文件路径，然后读取csv文件，最后对数据按照time列进行排序；
    #如果传感器数据是Accelerometer，如果提供了偏差值，则对每个轴的数据减去相应的偏差值，计算每个样本的加速度向量的L2范数（即欧几里得范数,数组元素平方和的平方根）
    for sensor in sensor_types:
        try:
            filename = f'data/{dirname}/{sensor}.csv'
            df = pd.read_csv(filename, sep=',')
            #print(f'df的内容是',df.head())
            df.sort_values(by='time', inplace=True)

            if sensor == 'Accelerometer':
                if biases is not False:
                    for axis, value in biases.items():
                        #df[axis] = df[axis] - value[0]
                        df[axis] = df[axis] - value
                df['l2_norm'] = np.linalg.norm(df[['x', 'y', 'z']].values, axis=1)  #计算每个样本点的加速度向量的L2范数

            df['str_time'] = pd.to_datetime(df['time'], unit='ns')
            #上面是将时间戳转换为pandas的datetime对象，将处理后的数据和采样频率存储到sensor_date和fs字典中

            sensor_data[sensor] = df
            #print(f'{sensor}的数据是',sensor_data[sensor].head())
            #quit()
            
            fs[sensor] = np.round(np.mean(1e9 / df['time'].diff())) #diff()计算相邻时间差，计算每秒的平均采样
            #print(f'{sensor}的采样频率是{fs[sensor]}')

            #如果设置该参数，则移除数据开始和结束部分
            if remove_ends_seconds:
                sensor_data[sensor] = sensor_data[sensor].iloc[
                                      round(remove_ends_seconds * fs[sensor]):-round(remove_ends_seconds * fs[sensor])]
            #iloc方法是根据指定位置范围内来进行选取数据
        except IOError:
            print("No such sensor for this session")

    return sensor_data, fs


# ------------------------------------------------------ Plotting utils ---------------------------------------------
#作用：绘制单个传感器数据的图表，并以直观的方式展示传感器数据的趋势，同时标记出峰值和阈值
#用于绘制单个传感器的数据和步行数据的图表，遍历列，跳过时间戳和最后一列，然后添加轨迹显示传感器数据，如果提供了峰值索引，绘制峰值标记
def plot_single_sensor(single_sensor_data, type, fig, col, row, fs=0, peaks=None, peakTH=2):
    for metric in single_sensor_data.columns[1:-1]:
        #print(f'绘制{type}的{metric}数据')
        #添加传感器数据的scatter图，用于展示传感器数据的变化趋势
        fig.add_trace(go.Scatter(x=list(single_sensor_data['str_time']),
                                 y=list(single_sensor_data[metric]),
                                 name=f'{type} ({metric})', mode='lines'),
                      col=col,
                      row=row)
    #添加峰值的scatter图，用于标记检测到的峰值，以红色的交叉点表示
    if peaks is not None:
        fig.add_trace(go.Scatter(x=list(single_sensor_data['str_time'][peaks]),
                                 y=list(single_sensor_data['l2_norm'][peaks]), mode='markers',
                                 marker=dict(size=8, color='red', symbol='cross'),
                                 name='Detected Peaks'),
                      col=col,
                      row=row)
        #添加轨迹，显示峰值阈值水平线，阈值为peakTH，以蓝色的线表示
        # fig.add_trace(go.Scatter(x=[single_sensor_data['str_time'].head(1), single_sensor_data['str_time'].tail(1)],
        fig.add_trace(
            go.Scatter(x=list(single_sensor_data['str_time'].head(1).append(single_sensor_data['str_time'].tail(1))),
                       y=[peakTH, peakTH],
                       mode='lines', line=dict(color="RoyalBlue", width=0.5), fillcolor="LightSkyBlue",
                       name='TH: ' + str(peakTH)),
            row=row, col=col)

    fig.layout.annotations[row - 1]['text'] = f'Plot of {type} sensor sampling @ {fs} Hz'
    #更新子图的X轴和Y轴的样子
    fig.update_xaxes(title='time', tickfont_size=6, **plotlyHelper.axisStyle, row=row, col=1)
    fig.update_yaxes(title='Signal', tickfont_size=6, **plotlyHelper.axisStyle, row=row, col=1)

#绘制多个传感器的数据
def plot_sensors(sensor_data, fs, peaks=None, title='', peakTH=2):
    fig = make_subplots(rows=len(sensor_data), cols=1, horizontal_spacing=0.055,
                        subplot_titles=['d' for _ in range(len(sensor_data))])
    #创建一个子图，行数等于传感器数据的数量，遍历传感器数据字典，为每个传感器创建一个子图，绘制单个传感器数据
    row = 1
    for sensor, data in sensor_data.items():
        plot_single_sensor(data, sensor, fig, 1, row, fs=fs[sensor],
                           peaks=peaks if sensor == 'Accelerometer' else None,
                           peakTH=peakTH)
        row += 1 #更新行索引，为下一个传感器数据做准备

    #针对图表中的注释，注释内容左对齐
    for i in fig['layout']['annotations']:
        i['font']['size'] = 14
        i['xanchor'] = 'left'

    fig.update_layout(title_text=f'<b>Mobile phone sensor readings: {title}</b>', showlegend=True,
                      **plotlyHelper.layoutStyle)

    fig.show()

#绘制一系列步行数据，接受step对象的列表
def plot_steps(list_of_steps):
    '''
    Plot all steps on-top of each other (just to get a sense of pattern_
    :param list_of_steps: list with Step objects
    :return: plots the steps aligned on top of each other
    '''
    fig = go.Figure()
    #遍历步行数据列表，为图表添加一个新的轨迹，显示步行数据
    for step in list_of_steps:
        fig.add_trace(go.Scatter(x=np.arange(len(step.raw_signal)), y=step.raw_signal,
                                 mode='lines', line=dict(color="RoyalBlue", width=0.5), fillcolor="LightSkyBlue"))

    fig.show()


# -------------------------------------------------- Processing utils -------------------------------------------------
#校准步长传感器数据，根据加速度向量、步数、移动距离计算增益，并返回校准后的步长数据，有助于在后续步长估计中得到更准确的结果
def calibrate(acc_vec, n_steps, distance, method='kim'):
    if method == 'kim':
        sk_true = distance / n_steps         # 真实步长
        mean_acc = np.cbrt(acc_vec)         # 计算加速度的三次方根  
        gk = sk_true / mean_acc[0]          # 计算增益，真实步长与平均加速度的比值
        return gk
    else:
        print('method not supported')
        return False

#从加速度计数据中提取和计算步行数据并生成步行对象的列表，th是时间窗口阈值，每个步伐的时间窗口大小
def collect_steps(data, fs, indices, owner, true_size=None, th=0.58):
    steps = []
    #遍历索引列表indices，识别和收集每一步的步长数据
    for i, indice in enumerate(indices[:-1]):
        indx_start = max(int(indice - 0.5 * th * fs), 0)    #计算步行数据中每个步伐的时间窗口的起始索引。这样设计的目的是确保时间窗口围绕着步伐中心，以便获取步伐发生时的信号振幅情况
        indx_end = min(int(indice + 0.5 * th * fs), len(data['l2_norm']))

        abs(data['time'][indices[i + 1]] - data['time'][indices[i]])    #相邻步行数据的时间差 
        mag = sum(list(data['l2_norm'][indx_start:indx_end]))   #振幅，这个总和可以量化该时间段内的信号的总体强度
        length = abs(indx_end - indx_start)     #时间窗口内相邻采样数据点的数量，用来衡量步行周期
        #print(f'Step {i+1}: length={length}, mag={mag}')

        steps.append(Step(owner=owner, length=length, mag=mag, true_size=true_size))
        steps[-1].raw_signal = list(data['l2_norm'][indx_start:indx_end])

    
    return steps

#用于计算步长的平均增益，先展平成一个一维列表，计算所有增益的总和再除以步数，得到平均增益，可以评估整个步行数据集的平均增益水平
def calc_mean_gain(steps_list):
    flat_list = flatten_list(steps_list)
    return sum([step.Gk for step in flat_list]) / len(flat_list)


# ---------------------------------------------------- Testing utils --------------------------------------------------
#计算每一步的偏差，遍历步数，每一步，计算预测长度与真实值之间的误差，并将误差平方累加到error变量中
#最后计算所有步的均方根误差
def calc_per_step_errors(steps_list, gk):
    steps_list = flatten_list(steps_list)   #展平步行数据列表
    error = 0   #用于累积每一步的误差值
    for step in steps_list:
        error += (gk * np.cbrt(step.magnitude / step.length) - step.true_size) ** 2
    return np.sqrt(error / len(steps_list))

#计算每次行走的总体误差，返回的是均方误差，表现整个行走过程的准确性。
#计算方法：遍历每一个行走数据，计算估计的行走长度，与真实距离的差值得到相对误差，对误差进行平方，然后除以真实距离的平方，得到均方误差
def calc_per_walk_errors(walking_sessions_list, gk, true_distance=20):
    error = 0
    for walk in walking_sessions_list:
        error += ((true_distance-get_length(walk, gk))/true_distance)**2
        #相对误差，表示估计长度相对于真实距离的偏差程度
    return np.sqrt(error / len(walking_sessions_list))

#估计步行数据的总长度，由步行数据对象中的振幅、步长以及增益参数gk组合得到，提供一个方法来评估步行过程的长度
def get_length(steps_list, gk):
    length = 0
    for step in steps_list:
        length += gk * np.cbrt(step.magnitude / step.length)     #计算每个估计的步长
    # print('Calculated Length =', length)
    return length

#估计每一步的步长
def test(acc_vec, n_steps, gk=None, method='kim'):
    if method == 'kim':
        if gk:
            return np.cbrt(acc_vec) * gk    #计算估计的步长值
        else:
            print('please input Gk gain param')
            return False
    else:
        print('method not supported')
        return False

