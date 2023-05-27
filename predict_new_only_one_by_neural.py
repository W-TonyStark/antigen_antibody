import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# def plot_curve(data):
#     fig = plt.figure()
#     plt.plot(range(len(data)), data, color='blue')
#     plt.legend(['value'], loc='upper right')
#     plt.xlabel('step')
#     plt.ylabel('value')
#     plt.show()

def plot_curve(data, fig=None):
    if fig is None:
        fig = plt.figure()
    plt.plot(range(len(data)), data)
    plt.xlabel('step')
    plt.ylabel('value')
    return fig

filename = 'D:/Sheng/sheng/Xlab/2023/antigen-antibody/单种被测物.xlsx'

# 处理excel中的数据
df = pd.read_excel(filename, sheet_name='data')
df = df.iloc[:, [0] + list(range(1, df.shape[1], 2))]
df = df.rename(columns={df.columns[0]: 'time_aaa'})
df.columns = df.columns.str.slice(stop=-4)
df_tra = df.transpose()
# 去掉时间列
df_tra = df_tra.drop(df_tra.index[0])
# 特征数据
data = np.array(df_tra)
# 数据标签
label = np.array(df_tra.transpose().columns.values.tolist())
# 进行标签编码，创建一个标签编码器对象
encoder = LabelEncoder()
# 将标签编码为数值标签
label = encoder.fit_transform(label)

# print(data.shape)
# print(label)

data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=0.2, random_state=0)

# define neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(data.shape[1], 180)
        self.fc2 = nn.Linear(180, 52)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# set training parameters
batch_size = 256
learning_rate = 0.001
epochs = 6000

# convert data to tensors and move to GPU
X_train = torch.from_numpy(data_train).float().cuda()
y_train = torch.from_numpy(label_train).long().cuda()
X_test = torch.from_numpy(data_test).float().cuda()
y_test = torch.from_numpy(label_test).long().cuda()

# initialize model and optimizer
net = Net().cuda()

# optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# define loss function
criterion = nn.CrossEntropyLoss()

trans_loss=[]
test_acc=[]

# train model
for epoch in range(epochs):
    running_loss = 0.0
    for i in range(0, len(X_train), batch_size):
        inputs = X_train[i:i+batch_size]
        labels = y_train[i:i+batch_size]

        optimizer.zero_grad()

        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward() #反向传播
        optimizer.step()

        running_loss += loss.item()

    # calculate training loss
    train_loss = running_loss / (len(X_train) / batch_size)
    trans_loss.append(train_loss)
    # calculate test accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            inputs = X_test[i:i+batch_size]
            labels = y_test[i:i+batch_size]

            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total
    test_acc.append(test_accuracy)
    # print statistics
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# make predictions on test data
with torch.no_grad():
    outputs = net(X_test)
    _, predicted = torch.max(outputs.data, 1)

# print predictions
print(predicted)

# 保存模型
# torch.save(net.state_dict(), 'model.pth')

# plot_curve(trans_loss)
# plot_curve(test_acc)
fig = plot_curve(test_acc)
plot_curve(trans_loss, fig=fig)
plt.show()