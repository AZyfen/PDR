import pandas as pd

# # 读取 Excel 文件
# df = pd.read_csv("gravity.csv")
# print(df.loc[df['y'] > 1.5] )

# print(df.head())

# #df['difference']=df['y']-df['x']

# #df.to_csv('new_gravity.csv', index=False)

# print("+++++++++++")
# df.loc[:,"差值"]=df["y"]-df["x"]
# print(df.loc[:,"差值"])


# # 计算第二列和第三列的差值
# """ df['Difference'] = df.iloc[:, 3] - df.iloc[:, 2]

# # 将修改后的数据保存回 Excel 文件
# df.to_excel('modified_excel_file.xlsx', index=False) """

# 创建一个示例 DataFrame
# data = {'time': pd.to_datetime(['2022-01-01 00:00:00', '2022-01-01 01:00:00', '2022-01-01 03:00:00'])}
# df = pd.DataFrame(data)
# print(df)

# # 计算时间间隔
# time_diff = df['time'].diff()
# print(time_diff)



# # 创建一个DataFrame
# df = pd.DataFrame({
#     'A': [1, 2, 3],
#     'B': [4, 5, 6]
# })

# print(df)
# print("----------")
# # 使用drop方法删除列'A'，设置inplace=True将修改原始DataFrame
# df.drop('A', axis=1, inplace=True)

# # 打印原始DataFrame，可以看到列'A'已经被删除
# print(df)
# # 输出:
# #   B
# # 1  5
# # 2  6
# print("----------")
# print("----------")
# print("----------")


# # 再次使用drop方法，这次不设置inplace参数（默认为False），将返回一个新的DataFrame
# df_dropped = df.drop('B', axis=1, inplace=False)

# # 打印原始DataFrame，这次它没有变化
# print(df)
# # 输出:
# #   A  B
# # 0  1  5
# # 1  2  6
# # 2  3  6
# print("----------")

# # 打印新创建的DataFrame，它不包含列'B'
# print(df_dropped)
# # 输出:
# #   A
# # 0  1
# # 1  2
# # 2  3

# my_list = [1, 2, 3, 4, 5]

# # 使用for循环遍历列表
# for item in my_list:
#     print(item)

# x="a"
# y="b"
# print (x,y)

# dict = {}
# dict['one'] = "This is one"
# dict[2] = "This is two"
 
# tinydict = {'name': 'runoob','code':6734, 'dept': 'sales'}
 
 
# print (dict['one'])          # 输出键为'one' 的值
# print (dict[2])              # 输出键为 2 的值
# print (tinydict)           # 输出完整的字典
# print (tinydict.keys())      # 输出所有键
# print (tinydict.values())    # 输出所有值


# var2 = "Python Runoob"
# print ("输出：="+var2[5:9]+"22222")
# def printme( str ):
#    #"打印任何传入的字符串"
#    print (str)
#    return
 
# # 调用函数
# printme("我要调用用户自定义函数!")
# printme("再次调用同一函数")

# def ChangeInt( a ):
#     a = 10
 
# a = 2
# ChangeInt(a)
# print (a) # 结果是 2

# Money = 2000
# def AddMoney():
#    # 想改正代码就取消以下注释:
#    global Money
#    Money = Money + 1
 
# print (Money)
# AddMoney()
# print (Money)

# str = input("请输入：")
# print ("你输入的内容是: ", str)

# import pandas as pd

# # 创建一个 DataFrame
# data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
#         'Age': [25, 30, 35, 40],
#         'City': ['New York', 'San Francisco', 'Los Angeles', 'Chicago']}
# df = pd.DataFrame(data)

# # 选择特定的列
# selected_columns = df[['Name', 'Age']]

# # 查看 DataFrame 的前几行
# print(df.head())

# print(selected_columns)

# data = {
#     'A': [1, 2, 3, 4, 5],
#     'B': [10, 20, 30, 40, 50]
# }
# df = pd.DataFrame(data)

# # 获取 DataFrame 的描述性统计信息
# description = df.describe()
# print(description)


import sys
from numpy import NaN, Inf, arange, isscalar, asarray, array

def peakdet(v, delta, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    
    Returns two arrays
    
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %      
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
    
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    
    """
    maxtab = []
    mintab = []
       
    if x is None:
        x = arange(len(v))
    
    v = asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    
    lookformax = True
    
    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)

if __name__=="__main__":
    from matplotlib.pyplot import plot, scatter, show
    series = [0,0,0,2,0,0,0,-2,0,0,0,2,0,0,0,-2,0]
    maxtab, mintab = peakdet(series,.3)
    plot(series)
    scatter(array(maxtab)[:,0], array(maxtab)[:,1], color='blue')
    scatter(array(mintab)[:,0], array(mintab)[:,1], color='red')
    show()



