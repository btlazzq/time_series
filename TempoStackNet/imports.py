data_y = data_y.to_numpy().reshape(-1,1)
data_y
np.sum(y_range)
data.shape
# 建立数据集x y 
# x (样本数量，时间窗口，x特征数)
# y (样本数量，步长，y特征数)

n, f = data.shape # n代表样本数量 f代表特征数量
print("data.shape",data.shape)

time_window_192 = 192 # 时间窗口
step = 32 # 预测多步

# 如果不用y来预测y这样写
# ts_x = np.zeros((n-time_window-step+1, time_window, f-1)) # f-1 表示特征数量 (样本数量，时间窗口，特征数)

# 如果用y来预测y这样写
ts_x = np.zeros((n-time_window_192-step+1, time_window_192, f)) # f-1 表示特征数量 (样本数量，时间窗口，特征数)
ts_y = np.zeros((n-time_window_192-step+1, step, 1)) # (样本数量，步长，特征数）
print('ts_x.shape',ts_x.shape)
print('ts_y.shape',ts_y.shape)

time_window_32 = 48 # 时间窗口
step = 32 # 预测多步
ts_x1 = np.zeros((n-time_window_32-step+1, time_window_32, f)) # f-1 表示特征数量 (样本数量，时间窗口，特征数)
ts_y1 = np.zeros((n-time_window_32-step+1, step, 1)) # (样本数量，步长，特征数）
print('ts_x1.shape',ts_x1.shape)
print('ts_y1.shape',ts_y1.shape)

time_window_16 = 48 # 时间窗口
step = 32 # 预测多步
ts_x2 = np.zeros((n-time_window_16-step+1, time_window_16, f)) # f-1 表示特征数量 (样本数量，时间窗口，特征数)
ts_y2 = np.zeros((n-time_window_16-step+1, step, 1)) # (样本数量，步长，特征数）
print('ts_x2.shape',ts_x2.shape)
print('ts_y2.shape',ts_y2.shape)
# 单一输出
time_window_192 = 192 # 时间窗口
for i in range(n-time_window_192-step+1):
    # 如果不用y来预测y这样写
    # ts_x[i] = data[i:i+time_window, :-1]
    
    # 如果用的话
    ts_x[i] = data[i:i+time_window_192, :]
    ts_y[i] = data[i+time_window_192:i+time_window_192+step, -1].reshape(32,1) #预测的未来步数
print(ts_x.shape, ts_y.shape)

time_window_32 = 48 # 时间窗口
for i in range(n-time_window_32-step+1):
    # 如果不用y来预测y这样写
    # ts_x[i] = data[i:i+time_window, :-1]
    
    # 如果用的话
    ts_x1[i] = data[i:i+time_window_32, :]
    ts_y1[i] = data[i+time_window_32:i+time_window_32+step, -1].reshape(32,1) #预测的未来步数
print(ts_x1.shape, ts_y1.shape)

time_window_16 = 48 # 时间窗口
for i in range(n-time_window_16-step+1):
    # 如果不用y来预测y这样写
    # ts_x[i] = data[i:i+time_window, :-1]
    
    # 如果用的话
    ts_x2[i] = data[i:i+time_window_16, :]
    ts_y2[i] = data[i+time_window_16:i+time_window_16+step, -1].reshape(32,1) #预测的未来步数
print(ts_x2.shape, ts_y2.shape)


# 多个set出
# for i in range(n-time_window-step+1):
#     ts_x[i] = data[i:i+time_window, :]
#     ts_y[i] = data[i+time_window:i+time_window+step, :] 
import matplotlib.pyplot as plt

# 创建一个新的图形
plt.figure(figsize=(10, 6))

# 绘制测试集的真实值
plt.plot(y_test_trues[0], label='True', color='blue', linestyle='-')

# 绘制测试集的预测值
plt.plot(y_test_preds[0], label='Predicted', color='red', linestyle='--')

# 添加标题和标签
plt.title('Test Set: True vs Predicted', fontsize=16)
plt.xlabel('Time Step', fontsize=12)
plt.ylabel('Value', fontsize=12)

# 添加图例
plt.legend()

# 显示网格线
plt.grid(True, linestyle=':', alpha=0.5)

# 调整布局使标签不重叠
plt.tight_layout()

# 显示图像
plt.show()
