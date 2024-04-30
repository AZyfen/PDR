#这段代码实现了一个步行检测器的校准和评估流程，
# 包括参数优化、步行数据提取、误差计算和跨用户校准测试。
# 通过这个过程，可以优化步行检测器的性能，使其在不同用户和不同步行姿势下都能准确估计步行距离。
from pdr_utils import *
from navigation_utils import get_calibration_values
import os

DISTANCE = 20  # walking distance in meter
#通过分析步行数据来优化步行检测器的参数，以便更准确的估计步行距离

# ======================================== Calibrate Peak Detector ====================================================
#用于校准峰值检测器，遍历一系列预定义峰值高度和距离限制，计算在不同参数下检测到的步数与实际步数之间的误差，最后找到误差最小的参数组合，并返回这些参数
def calibrate_peak_detector(resolution=30, peak_limits=(0.5, 4.5), distance_limits=(0.1, 1), bias=False):
    #创建一个numpy数组，包含在peak_limits范围内，分辨率为resoulution的峰值
    peak_to_test = np.round(np.linspace(peak_limits[0], peak_limits[1], resolution), 4)
    #创建一个numpy数组，包含在diatance_limits范围内，分辨率为resolution的距离值
    distances_to_test = np.round(np.linspace(distance_limits[0], distance_limits[1], resolution), 4)
    #初始化一个三维NumPy数组error_matrix，用于存储每个峰值和距离组合的误差
    error_matrix = np.zeros((len(peak_to_test), len(distances_to_test), 1))

    # print(len(peak_to_test))
    # print(len(distances_to_test))
    
    #遍历该目录下的所有文件
    for filename in os.listdir('data/walking'):
        #为每个文件初始化一个numpy数组temp_error_matrix，用于存储当前文件的误差
        temp_error_matrix = np.zeros((len(peak_to_test), len(distances_to_test)))
        #从文件名中提取姿势pose，步数steps，分隔符和名称name
        pose, steps, _, name = filename.split('-')
        #尝试将步数转换为整数，如果失败，打印错误信息
        try:
            steps = int(steps)
        except Exception() as e:
            print('No valid number of steps in filename', e)

        #加载会话数据，load_session函数可能从文件中读取传感器数据并返回数据和采样频率，bias参数用于指示是否应用偏差校准
        data_calibrate, fs_calibrate = load_session(f'walking/{filename}', biases=bias)
        #遍历所有峰值和距离组合，find_peak函数找到加速度计数据中的峰值索引，计算检测到的步数，计算误差，即实际步数与检测到的步数之间的差异。
        #（从data_calibrate中获取Accelerometer键对应值的l2_norm数据，峰值之间的距离乘以频率得到距离阈值，峰值高度使用pTH，最后返回结果的第一个元素）
        for i, pTH in enumerate(list(peak_to_test)):
            for j, dTH in enumerate(list(distances_to_test)):
                peak_indices = find_peaks(data_calibrate['Accelerometer']['l2_norm'],
                                          distance=dTH * np.round(fs_calibrate['Accelerometer']),
                                          height=pTH)[0]#理论上给定的时间间隔内，理论上有多少个数据点

                steps_counted = len(peak_indices)   #峰值检测器检测到的步数


                temp_error_matrix[i, j] = (steps - steps_counted) / steps

                #print(temp_error_matrix[i, j])

        #将当前文件的误差矩阵堆叠到总误差矩阵中
        error_matrix = np.dstack((error_matrix, temp_error_matrix))
    #计算误差矩阵的平方根，得到每个峰值和距离组合的平均误差
    error_matrix = np.sqrt(np.mean(error_matrix[:, :, 1:] ** 2, axis=2))
    #找到误差矩阵中的最小值及其索引
    min_indexes = np.where(error_matrix == np.amin(error_matrix))
    #打印出最小误差对应的峰值和距离
    print(peak_to_test[min_indexes[0]], distances_to_test[min_indexes[1]])
    #创建一个plotly的热图图表，用于可视化误差
    fig = go.Figure(data=go.Heatmap(
        x=distances_to_test,
        y=peak_to_test,
        z=error_matrix,
        type='heatmap'))
    #在热图上添加一个散点，表示最小误差的位置
    fig.add_trace(go.Scatter(x=distances_to_test[min_indexes[1]],
                             y=peak_to_test[min_indexes[0]], mode='markers',
                             marker=dict(size=15, color='red', symbol='cross'),
                             name='minimum'))

    fig.update_yaxes(title='Peak TH value', tickfont_size=25, **plotlyHelper.axisStyle)
    fig.update_xaxes(title='Distance TH value', tickfont_size=25, **plotlyHelper.axisStyle)

    fig.update_layout(title_text=f'<b>Error vs Peak TH</b>', **plotlyHelper.layoutStyle)
    fig.show()

    return peak_to_test[min_indexes[0][0]], distances_to_test[min_indexes[1][0]]


# ======================================== Calculate Gk ==============================================================
#第一个参数是步行数据文件的目录，dTH：峰值之间距离的阈值，检测加速度数据中的步态峰值来获取每一步的步长数据，pTH：峰值最小高度的阈值，确保检测到的峰值足够高，plot布尔值，如果为true，则绘制传感器数据图
#owner：指定的文件所有者，用于筛选特定用户的文件，pose指定的步行姿势，用于筛选特定姿势的数据，如果为all，则不进行姿势筛选，bias布尔值，指示是否应用偏差校准
#可以批量的从文件中提取步行数据
#从指定目录中提取步行数据，对步行数据进行处理，并从中检测步伐，最后返回一个步行步伐对象的列表。它具体地处理了文件读取、步伐检测、步长计算等多个任务
def extract_all_steps(parent_dir, dTH=0.58, pTH=2, plot=False, owner='all', pose='all', bias=False):
    '''
    Extract the steps from
    :param dTH: TH for distance between peaks
    :param pTH: TH for minimal height of peaks
    :param plot: Boolean, set True to plot sensors
    :param owner: if set to specific name will only extract steps from specified owner
    :param pose: Only extract steps from a certain pose ('inear, inhand, pocket')
    :return: a list with Step objects extracted from all the files in the directory
    '''
    #定义了一个列表包含了需要绘制的传感器类型，初始化一个空列表，用于存储提取的步行步骤对象
    sensors_to_plot = ['Accelerometer', 'Barometer', 'Gravity', 'Gyroscope']
    steps_list = []
    #遍历parent_dir目录中的所有文件，从文件名中提取步行姿势、步数、分隔符和文件所有者
    for filename in os.listdir(parent_dir):

        pose, steps, _, name = filename.split('-')
        #尝试将步数转换为整数，如果失败则打印错误信息
        try:
            steps = int(steps)
        except Exception() as e:
            print('No valid number of steps in filename', e)
        #检查文件的所有者和步行姿势是否符合筛选条件，加载指定文件的步行数据和采样频率，load_session函数用于读取和处理数据；find_peak函数在加速度计数据中找到峰值索引
        if (owner == name or owner == 'all') and (pose == pose or pose == 'all'):
            data_calibrate, fs_calibrate = load_session(f'walking/{filename}', biases=bias)

            peak_indices = find_peaks(data_calibrate['Accelerometer']['l2_norm'],
                                      distance=dTH * np.round(fs_calibrate['Accelerometer']),
                                      height=pTH)[0]

            #如果plot为true，绘制选定的传感器数据，计算真实步长大小，DISTANCE是全局变量，表示步行距离
            if plot:
                plot_sensors({key: value for key, value in data_calibrate.items() if key in sensors_to_plot},
                             fs=fs_calibrate, peaks=peak_indices, title=filename)

            true_size = DISTANCE / steps    #计算每一步的真实步长大小
            steps_list.append(collect_steps(data_calibrate['Accelerometer'],
                                            fs_calibrate['Accelerometer'],
                                            peak_indices, name, true_size=true_size, th=dTH))   #收集每一步的步长数据，并添加到步行数据列表中

    print('Got all the steps!')

    print(f'Total number of steps: {len(steps_list)}')
    print(steps_list)

    return steps_list


#从特定会话中提取步行数据，dTH峰值检测的距离阈值，默认值为0.58，峰值检测的高度阈值，默认值为2
def extract_steps_for_session(filename, dTH=0.58, pTH=2, plot=False, bias=False):
    sensors_to_plot = ['Accelerometer']
    pose, steps, _, name = filename.split('-')
    try:
        steps = int(steps)
    except Exception() as e:
        print('No valid number of steps in filename', e)

    #加载步行数据以及相应的采样频率，find_peak函数找到加速度计数据中的峰值索引
    data_calibrate, fs_calibrate = load_session(f'walking/{filename}', biases=bias)

    peak_indices = find_peaks(data_calibrate['Accelerometer']['l2_norm'],
                              distance=dTH * np.round(fs_calibrate['Accelerometer']),
                              height=pTH)[0]

    if plot:
        plot_sensors({key: value for key, value in data_calibrate.items() if key in sensors_to_plot},
                     fs=fs_calibrate, peaks=peak_indices, title=filename, peakTH=pTH)

    true_size = DISTANCE / steps
    steps_list = collect_steps(data_calibrate['Accelerometer'],
                               fs_calibrate['Accelerometer'],
                               peak_indices, name, true_size=true_size, th=dTH) #收集每一步的步长数据，并添加到步行数据列表中

    return steps_list


# ======================================== Some tests ==============================================================
#用于测试跨用户的校准情况，这个函数首先尝试移除偏差，然后校准峰值检测器，接着从两个不同的用户数据中提取步行数据来评估校准的准确性
#计算步长增益GK、计算步长误差和会话误差，并打印结果。
def test_cross_user_calibration(remove_bias=False, test_run=None):                            
    #remove_bias为真，获取Ido用户的校准偏差值，否则不移除偏差                
    if remove_bias:
        _, biases, _, _ = get_calibration_values('calibration', user='Ido')
    else:
        biases = False
    #校准峰值检测器，传入偏差值，返回两个值pth峰值高度阈值和dth峰值距离阈值，即为最佳的峰值和距离参数设置
    pth, dth = calibrate_peak_detector(bias=biases)
    #提取Ido用户的所有步行数据，返回一个包含步行步骤对象的列表
    list_of_steps_ido = extract_all_steps(parent_dir='data/walking',
                                          owner='Ido',
                                          # pose='inear',
                                          bias=biases,
                                          dTH=dth,
                                          pTH=pth)
    #同样，提取Matan用户的步行数据，根据是否移除偏差，打印不同的结果标题
    list_of_steps_matan = extract_all_steps(parent_dir='data/walking',
                                            owner='Matan',
                                            # pose='inear',
                                            bias=biases,
                                            dTH=dth,
                                            pTH=pth)

    if remove_bias:
        print('Results with bias REMOVED:')
    else:
        print('Results with bias:')

    # ---- test calibrating User 1 and running on both:
    #计算Ido用户步行数据的平均增益值Gk
    Gk = calc_mean_gain(list_of_steps_ido)
    print('Mean gain value (Gk) for Ido bias is:', Gk)
    #计算Ido用户校准下Matan用户数据的步长误差和会话误差
    step_error = round(calc_per_step_errors(list_of_steps_matan, Gk), 3)
    session_error = round(calc_per_walk_errors(list_of_steps_matan, Gk), 3)
    print(f'RMS Error (Ido/Matan) is:{step_error} per step, {session_error} per walk')
    #计算Ido用户校准下Ido用户步行数据的步长误差和会话误差
    step_error = round(calc_per_step_errors(list_of_steps_ido, Gk), 3)
    session_error = round(calc_per_walk_errors(list_of_steps_ido, Gk), 3)
    print(f'RMS Error (Ido/Ido) is:{step_error} per step, {session_error}m per walk')

    #计算Matan用户步行数据的平均增益值Gk
    # ---- test calibrating on user 2 and running on both:
    Gk = calc_mean_gain(list_of_steps_matan)
    print('Mean gain value (Gk) bias for Matan is:', Gk)
    # 计算Matan用户校准下Ido用户数据的步长误差和会话误差
    step_error = round(calc_per_step_errors(list_of_steps_ido, Gk), 3)
    session_error = round(calc_per_walk_errors(list_of_steps_ido, Gk), 3)
    print(f'RMS Error (Matan/Ido) is:{step_error} per step, {session_error}m per walk')
    #计算Matan用户步行数据的步长误差和会话误差
    step_error = round(calc_per_step_errors(list_of_steps_matan, Gk), 3)
    session_error = round(calc_per_walk_errors(list_of_steps_matan, Gk), 3)
    print(f'RMS Error (Matan/Matan) is:{step_error} per step, {session_error}m per walk')

    # --- Plot a test file to see some graphs (along with peaks detected)
    if test_run is not None:
        try:
            extract_steps_for_session(test_run, plot=True, pTH=pth, dTH=dth, bias=biases)
        except Exception as e:
            print('No such test file found, exception:', e)


# ======================================== MAIN =====================================================================
#调用两次，一次不移除偏差，一次移除偏差
#定义一个变量filename，表示要测试的文件名
#filename = 'texting-30-steps-Zyf'
filename = 'texting-20-steps-Cxh'

test_cross_user_calibration(remove_bias=False, test_run=filename)
test_cross_user_calibration(remove_bias=True, test_run=filename)


# -- Other Tools and plots:  ----------------------------------------------------------------------------------------

# get_length(list_of_steps)


# plot_steps(list_of_steps)
