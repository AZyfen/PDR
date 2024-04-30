import numpy as np
import plotly.graph_objs as go

class Step:
    def __init__(self, raw_signal):
        self.raw_signal = raw_signal

def plot_steps(list_of_steps):
    fig = go.Figure()
    for step in list_of_steps:
        fig.add_trace(go.Scatter(x=np.arange(len(step.raw_signal)), y=step.raw_signal,
                                 mode='lines', line=dict(color="RoyalBlue", width=0.5), fillcolor="LightSkyBlue"))
    fig.show()

# 创建 Step 对象的列表
list_of_steps = [
    Step(raw_signal=[1, 2, 3, 4, 5]),
    Step(raw_signal=[5, 4, 3, 2, 1]),
    # ... 更多的 Step 对象 ...
]

# 调用函数以绘制步骤
plot_steps(list_of_steps)