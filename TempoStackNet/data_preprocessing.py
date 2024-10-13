# data_pd = pd.read_csv('./1min_0.01_db3.csv')
data_pd = pd.read_csv('./btc_data.csv')
data_pd = data_pd.iloc[0:,1:]
data_y = data_pd.iloc[0:,-1]
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(data_pd)
data = scaler.transform(data_pd)
pd.DataFrame(data).head(3)
scaler_y = MinMaxScaler()
scaler_y.fit(data_y)
y_range = scaler_y.data_range_
y_range = np.sum(y_range)
y_min = scaler_y.data_min_
y_min = np.sum(y_min)
print(y_range)
y_min


### 第一个模型的
# 将数据转为torch
from torch.utils.data import DataLoader, TensorDataset, Subset

# 打包x和y
x = torch.tensor(ts_x, dtype=torch.float32)
y = torch.tensor(ts_y, dtype=torch.float32)
dataset = TensorDataset(x, y) # 该类中的tensor 第一维度必须相等

# 切分训练集和测试集
num_example = x.size(0) # 样本量
print('num_example',num_example)
train_index = np.arange(int(num_example * 0.8))
train_num = len(train_index)
print('train_num',train_num)

test_index = np.arange(int(num_example * 0.8), num_example)
test_num = len(test_index)
print('test_num',test_num)
print()

# 生成训练集和测试集
train = Subset(dataset, train_index)
test = Subset(dataset, test_index)

# 分batch_size
train_dl = DataLoader(train, batch_size=64)
test_dl = DataLoader(test, batch_size=64)

# batch_size time_windows feature
for bx, by in train_dl:
    print(bx.shape)
    print(by.shape)
    break
    
print('----------------')

for bx, by in test_dl:
    print(bx.shape)
    print(by.shape)
    break
### 第二个模型的
# 将数据转为torch
from torch.utils.data import DataLoader, TensorDataset, Subset

# 打包x和y
x = torch.tensor(ts_x1, dtype=torch.float32)
y = torch.tensor(ts_y1, dtype=torch.float32)
dataset = TensorDataset(x, y) # 该类中的tensor 第一维度必须相等

# 切分训练集和测试集
num_example = x.size(0) # 样本量
print('num_example',num_example)
train_index = np.arange(int(num_example * 0.8))
train_num = len(train_index)
print('train_num',train_num)

test_index = np.arange(int(num_example * 0.8), num_example)
test_num = len(test_index)
print('test_num',test_num)
print()

# 生成训练集和测试集
train = Subset(dataset, train_index)
test = Subset(dataset, test_index)

# 分batch_size
train_dl1 = DataLoader(train, batch_size=64)
test_dl1 = DataLoader(test, batch_size=64)

# batch_size time_windows feature
for bx, by in train_dl1:
    print(bx.shape)
    print(by.shape)
    break
    
print('----------------')

for bx, by in test_dl1:
    print(bx.shape)
    print(by.shape)
    break
### 第三个模型的
# 将数据转为torch
from torch.utils.data import DataLoader, TensorDataset, Subset

# 打包x和y
x = torch.tensor(ts_x2, dtype=torch.float32)
y = torch.tensor(ts_y2, dtype=torch.float32)
dataset = TensorDataset(x, y) # 该类中的tensor 第一维度必须相等

# 切分训练集和测试集
num_example = x.size(0) # 样本量
print('num_example',num_example)
train_index = np.arange(int(num_example * 0.8))
train_num = len(train_index)
print('train_num',train_num)

test_index = np.arange(int(num_example * 0.8), num_example)
test_num = len(test_index)
print('test_num',test_num)
print()

# 生成训练集和测试集
train = Subset(dataset, train_index)
test = Subset(dataset, test_index)

# 分batch_size
train_dl2 = DataLoader(train, batch_size=64)
test_dl2 = DataLoader(test, batch_size=64)

# batch_size time_windows feature
for bx, by in train_dl2:
    print(bx.shape)
    print(by.shape)
    break
    
print('----------------')

for bx, by in test_dl2:
    print(bx.shape)
    print(by.shape)
    break
