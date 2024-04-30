from pdr_utils import *
import transforms3d
import plotlyHelper
import plotly.graph_objects as go

'''
## General flow:

1. Load the data: Acc, Gyro (anything else?)    加载数据和应用校准数据来校正传感器偏差

** apply calibration data

2. Set origin loc (P_0) and orientation init (W_0)  设置起始位置和初始方向
3. Generate orientation vector: 
    a. Rotate angular velocity  陀螺仪数据计算角速度
    b. Integrate to get next pose/attitude  积分角速度来更新姿态roll,pitch,yaw

4. Generate linear velocity vector:
    a. Rotate acceleration to inertial frame and integrate 将加速度转换到惯性参考系，并积分得到速度

5. Generate Position vector
    a. Integrate velocity   速度积分来计算新的位置

Gyro:Orientation:  
Y = roll
X = pitch
Z = yaw
结果可视化:展示加速度,速度,位置和姿态随时间的变化。绘制3D位置图,展示行人的运动轨迹
'''

#raw2euler函数的作用是将陀螺仪数据转换为欧拉角，并将欧拉角转换为旋转矩阵。  
#首先，将陀螺仪数据转换为角速度，即将陀螺仪数据除以采样周期dt，得到角速度。  
#然后，将角速度转换为欧拉角，欧拉角的计算公式为：  
#roll = arctan(y/z)                              绕Z轴旋转角速度y/z后的X轴方向
#pitch = arctan(-x/sqrt(y^2+z^2))                 绕Z轴旋转角速度y/z后的Y轴方向
#yaw = arctan(x/y)                                绕Z轴旋转角速度y/z后的Z轴方向
#最后，将欧拉角转换为旋转矩阵，旋转矩阵的计算公式为：  
#第一列代表了绕Z轴旋转angle_vec[1]后，绕X轴angle_vec[0]偏航角后的Y轴方向，   
#第二列代表了绕Z轴旋转angle_vec[1]后，绕X轴angle_vec[0]偏航角后的Z轴方向，  
#第三列代表了绕Z轴旋转angle_vec[1]后，绕Y轴angle_vec[0]俯仰角后的Z轴方向。  

def raw2euler(angle_vec):
    TFM = np.array([[1, np.sin(angle_vec[0]) * np.tan(angle_vec[1]), np.cos(angle_vec[0]) * np.tan(angle_vec[1])],
                    [0, np.cos(angle_vec[0]), -np.sin(angle_vec[0])],
                    [0, np.sin(angle_vec[0]) / np.cos(angle_vec[1]), np.cos(angle_vec[0]) / np.cos(angle_vec[1])]])
    return TFM

#定义函数，将欧拉角向量转换为变换矩阵，使用transforms3d库   
#euler_angle_vec为欧拉角向量，格式为(roll, pitch, yaw)      
#返回值是旋转矩阵，格式为(3, 3)             
def euler2tfm(euler_angle_vec):
    quaternion = transforms3d.euler.euler2quat(*euler_angle_vec)       #将欧拉角转化为四元数表示
    return transforms3d.quaternions.quat2mat(quaternion)    #将四元数表示转换为对应的旋转矩阵
     
#定义一些辅助字典，用于将索引映射到陀螺仪的轴标签           
gyro_dict = {0: 'y', 1: 'x', 2: 'z'}

#定义陀螺仪和加速度计传感器的校准值函数                                                                              
#calibration_dir为校准数据所在的目录，user为用户名称，返回校准值字典gyro_bias, acc_bias, acc_scale, acc_bias_full                   
def get_calibration_values(calibration_dir, user='matan'):
    
    gyro_bias, acc_bias = {}, {}    #初始化陀螺仪和加速度计的偏差字典
    # acc_bias_full = np.zeros((3, 3))
    acc_scale = {}  #初始化加速度计的比例因子字典
    acc_bias_full = {'x': {}, 'y': {}, 'z': {}} #初始化加速度计的全轴偏差字典
    axes = ['x', 'y', 'z']  #初始化轴的列表
    #根据用户输入，设置校准文件的后缀
    if user in ['matan', 'Matan']:
        suffix = '-M'
    else:
        suffix = ''

    #遍历每个轴
    for axis in axes:
        #加载相应轴的校准数据
        data, _ = load_session(f'{calibration_dir}/{axis}-calibration{suffix}')
        #获取陀螺仪和加速度计和重力数据
        gyro = data['Gyroscope']
        acc = data['Accelerometer']
        gravity = data['Gravity']
        acc_with_gravity = acc
        #计算加速度计数据（减去重力）
        acc_with_gravity = acc_with_gravity[['x', 'y', 'z']] + gravity[['x', 'y', 'z']]
        #计算陀螺仪和加速度计的偏差
        gyro_bias[axis], acc_bias[axis] = [gyro[axis].mean()], [acc[axis].mean()]
        #计算全轴偏差
        acc_bias_full['x'][axis] = acc['x'].mean()
        acc_bias_full['y'][axis] = acc['y'].mean()
        acc_bias_full['z'][axis] = acc['z'].mean()
        #计算加速度计的比例因子
        acc_scale[axis] = [abs(acc_with_gravity[axis].mean()) / 9.8065]
        #打印加速度计的统计信息
        print(f'\nAcceleration Stats for axis {axis} pointing down')
        print(acc[['x', 'y', 'z']].describe())

    #返回包含偏差值的DataFrame（陀螺仪偏差、加速度计全轴偏差、比例因子和全轴偏差）
    return pd.DataFrame.from_dict(gyro_bias), \
           pd.DataFrame.from_dict(acc_bias_full).max(axis=0), \
           pd.DataFrame.from_dict(acc_scale), \
           pd.DataFrame.from_dict(acc_bias_full)

#定义获得初始对齐角度的函数，便于将设备的坐标系与 World 坐标系对齐
#gravity_vec为重力数据，格式为(x, y, z)         
def get_initial_alignment(gravity_vec):
    #计算初始化重力向量的均值
    init_gravity = gravity_vec[['x', 'y', 'z']].head(5).mean()
    #计算初始的俯仰角和偏航角
    phi_0 = np.arctan2(-init_gravity['x'], -init_gravity['z'])
    theta_0 = np.arctan2(-init_gravity['y'], np.linalg.norm(init_gravity[['z', 'x']]))
    #返回计算除的俯仰角和偏航角
    return phi_0, theta_0


# -------------------- Dead Reckoning ------------------------------------------------------------------------------
#定义行人航位推算函数，用于估计移动路径
def dead_reckon(dir_name, remove_bias=False, title='', sma=0):  #包含传感器数据的目录名称，是否移除传感器偏差，图表标题，简单移动平均的窗口大小
    #解析目录名称，获取姿态、步数、用户名称等信息
    pose, steps, _, name = dir_name.split('-')
    #加载指定目录下的传感器数据，并将数据和采样频率fs分别赋值给data，fs
    data, fs = load_session(dir_name)
    #获取陀螺仪数据
    gyro = data['Gyroscope']
    #计算陀螺仪数据的采样周期，即采样频率的倒数
    dt_gyro = 1 / fs['Gyroscope']
    #获取加速度计数据
    acc = data['Accelerometer']
    #计算加速度计数据的采样周期
    dt_acc = 1 / fs['Accelerometer']

    #获取重力数据
    gravity = data['Gravity']
    #获取初始的俯仰角和偏航角
    phi_0, theta_0 = get_initial_alignment(gravity)
    #如果需要移除传感器偏差
    if remove_bias:
        #获取陀螺仪和加速度计的校准值
        gyro_biases, acc_biases, acc_scale, acc_bias_per_axis = get_calibration_values('calibration', user=name)
        #从陀螺仪数据中减去偏差，以进行校准
        gyro[['x', 'y', 'z']] = gyro[['x', 'y', 'z']] - gyro_biases[['x', 'y', 'z']].to_numpy()
    else:
        #如果不移除偏差，设置默认的偏差和比例因子，所有值设置为0或1
        acc_biases = pd.DataFrame({'x': [0], 'y': [0], 'z': [0]})   #创建一个包含加速度计比例偏差的DataFrame，所有值都设置为0
        acc_scale = pd.DataFrame({'x': [1], 'y': [1], 'z': [1]})    #创建一个包含加速度计比例因子的DateFrame，所有值都设置为1
        acc_bias_per_axis = acc_biases
        gyro_biases = acc_biases        #这里不是很理解为什么用加速度计偏差初始化陀螺仪偏差

    #如果需要进行简单移动平均处理，对加速度计进行简单移动平均处理，窗口大小为sma，这个参数可以设置为0，表示不进行移动平均处理，也可以修改为其他值
    #简单移动平均处理的目的是为了平滑加速度计数据，去除噪声，提高数据的可靠性
    #.mean表示计算窗口内的均值                                                                               
    if sma:  # simple-moving-average
        acc['x'] = acc['x'].rolling(window=sma, min_periods=1).mean()
        acc['y'] = acc['y'].rolling(window=sma, min_periods=1).mean()
        acc['z'] = acc['z'].rolling(window=sma, min_periods=1).mean()

    # -------------------- Init arrays: ------------------------------------------------------------------------------

    start_pose = np.array([phi_0, theta_0, 0])  # pose (attitude) at t=0 表示设备在起始时刻的姿态，表示在空间中的方向
    start_loc = np.array([0, 0, 0])  # location at t=0   表示设备在起始时刻的位置，这个位置通常被设定为坐标原点
    start_velocity = np.array([0, 0, 0])  # velocity at t=0 表示设备在起始时刻的速度，0意味着静止
    start_acc = np.array([0, 0, 0])  # acceleration (in ref frame) at t=0   #起始加速度为0，意味着没有任何外部力的作用
    #创建姿态、速度、位置和加速度的数组，用于存储计算结果
    pose = np.array([start_pose])  # roll (phi), pitch (theta), yaw (psy)
    velocity = np.array([start_velocity])
    location = np.array([start_loc])
    acceleration = np.array([start_acc])

    # -------------------- Calc position and orientation: ------------------------------------------------------------
    #作用：根据陀螺仪数据和加速度计数据，计算每个时刻的姿态，并不断更新姿态数组
    #开始计算位置和方向
    #遍历陀螺仪和加速度计的数据行
    for row_gyro, row_acc in zip(gyro.itertuples(), acc.itertuples()):
        #计算角速度，减去陀螺仪的偏差
        angular_velocity = np.array([row_gyro.y-gyro_biases['y'], row_gyro.x-gyro_biases['x'], row_gyro.z-gyro_biases['z']])
        #计算新的姿态，通过将上一个姿态旋转角速度并乘以采样周期
        new_pose = pose[-1, :] + np.matmul(raw2euler(pose[-1, :]), angular_velocity) * dt_gyro
        #将新的姿态添加到姿态数组中，不断更新姿态数组
        pose = np.vstack((pose, new_pose))

        # in our case the gravity vector is already subtracted from the acceleration but we have other problems...
        #将最新的姿态转换为变换矩阵
        tfm = euler2tfm(pose[-1, :])

        # as bias errors (in our sensor) are not constant per axis per orientation (weird non-linearity)
        # we calculate a mean bias term for each axis in the three main orientations (x,y,z)
        #计算新的加速度，将加速度计数据转换到世界坐标系，并减去偏差和应用比例因子
        new_acc = np.matmul(tfm.T, np.array([(row_acc.y - acc_biases['y']) * acc_scale['y'],
                                           (row_acc.x - acc_biases['x']) * acc_scale['x'],
                                           (row_acc.z - acc_biases['z']) * acc_scale['z']])).squeeze()
        #将新的加速度添加到加速度数组中
        acceleration = np.vstack((acceleration, new_acc))
        #计算新的速度，通过对加速度进行积分来估计速度的变化量，将将其加到前一个时刻的速度上，以获得下一个时刻的速度估计值
        new_v = velocity[-1, :] + np.mean([acceleration[-2, :], acceleration[-1, :]], axis=0) * dt_acc
        #将新的速度添加到速度数组中
        #print(f'Velocity: '+str(new_v))
        #print(str(velocity))
        #quit()
        velocity = np.vstack((velocity, new_v))
        #print(velocity)
        #print(new_v)
        #计算新的位置，通过对速度进行积分得到位移增量即位置的变化量，将前一个时刻的位置向量与位移量相加，以获得下一个时刻的位置估计值
        new_loc = location[-1, :] + np.mean([velocity[-1, :], velocity[-2, :]], axis=0) * dt_acc
        #将新的位置添加到位置数组中
        location = np.vstack((location, new_loc))


        #输出速度信息
        print("Speed data over time:")
        for i in range(len(velocity)):
            print(f"Time {i* dt_acc}: Speed {velocity[i]}")



    # -------------------- Plot results: -----------------------------------------------------------------------------

    fig = go.Figure()
    #遍历横滚（roll）、俯仰（pitch）和偏航（yaw）这三个指标，向图表添加原始加速度数据的轨迹
    for i, metric in enumerate(['roll', 'pitch', 'yaw']):
        fig.add_trace(go.Scatter(x=list(acc['str_time']),
                                 y=list(acc[gyro_dict[i]]),
                                 name=f'Acc (raw) {gyro_dict[i]}', line=dict(width=4)))
        #向图表添加世界坐标系下加速度的轨迹
        fig.add_trace(go.Scatter(x=list(acc['str_time']),
                                 y=list(acceleration[:, i]),
                                 name=f'Acc (world frame) {gyro_dict[i]}', line=dict(width=4)))
        #向图表添加速度的轨迹
        fig.add_trace(go.Scatter(x=list(acc['str_time']),
                                 y=list(velocity[:, i]),
                                 name=f'Velocity {gyro_dict[i]}', line=dict(width=4, dash='dash')))
        #向图表添加位置的轨迹
        fig.add_trace(go.Scatter(x=list(acc['str_time']),
                                 y=list(location[:, i]),
                                 name=f'Position {gyro_dict[i]}', line=dict(width=4)))
        #向图表添加姿态的轨迹
        fig.add_trace(go.Scatter(x=list(acc['str_time']), 
                                 y=list(pose[:, i]),
                                 name=metric, line=dict(width=4)))
    #更新X轴和Y轴的标题和样式
    fig.update_xaxes(title='time', tickfont_size=6, **plotlyHelper.axisStyle)
    fig.update_yaxes(title='m', tickfont_size=6, **plotlyHelper.axisStyle)
    #更新图表的布局，包括标题和样式
    fig.update_layout(title_text=f'<b>{title}</b>', **plotlyHelper.layoutStyle)
    #显示图表
    fig.show()
    #创建一个新的Plotly图表对象，用于3D位置图
    fig2 = go.Figure()
    #向图表添加3D位置数据的轨迹
    fig2.add_trace(go.Scatter3d(x=list(location[:, 1]),
                                y=list(location[:, 0]),
                                z=list(location[:, 2]),
                                name=f'Position 3D plot', line=dict(width=4)))
    #更新3D图表的布局，包括标题和坐标轴标签
    fig2.update_layout(title_text='3D position plot', scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'))
    #显示3D位置图
    fig2.show()

    fig = go.Figure()
    for i in range(3):
        fig.add_trace(go.Scatter(x=range(len(velocity)), y=velocity[:, i], mode='lines', name=f'Velocity {["x", "y", "z"][i]}'))
    fig.update_layout(title_text=f'<b>Speed Over Time</b>', xaxis_title='Time (s)', yaxis_title='Speed (m/s)')
    fig.show()

def main():
    dead_reckon(dir_name='walking/inhand-31-steps-Zyf', remove_bias=True, title='Speed Over Time', sma=0)

if __name__ == "__main__":
    main()