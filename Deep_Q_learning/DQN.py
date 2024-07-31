import random
import gymnasium as gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from Deep_Q_learning.DQN_parameters import DQN_Parameter


class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)#zip(*transitions) 将 transitions 列表中的元组进行解压缩。它实际上是将元组的第一个元素组合成一个新的元组，将元组的第二个元素组合成另一个元组，以此类推。
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)

class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)#建立输入层和隐藏层神经网络，state_dim定义为state的维度，hidden_dim自己定义为64
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)#建议输出层神经网络，action_dim定义为action的维度

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQN:
    ''' DQN算法 '''
    def __init__(self, DQN_parameter):
        self.action_dim = DQN_parameter.get_action_dim()
        self.q_net = Qnet(DQN_parameter.get_state_dim(), DQN_parameter.get_hidden_dim(),
                          self.action_dim).to(DQN_parameter.get_device())  # Q网络
        # 目标网络
        self.target_q_net = Qnet(DQN_parameter.get_state_dim(), DQN_parameter.get_hidden_dim(),
                          self.action_dim).to(DQN_parameter.get_device())
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=DQN_parameter.get_learning_rate())
        self.gamma = DQN_parameter.get_gamma()  # 折扣因子
        self.epsilon = DQN_parameter.get_epsilon()  # epsilon-贪婪策略
        self.target_update = DQN_parameter.get_target_update()  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = DQN_parameter.get_device()

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.rand(self.action_dim)
        else:
            if isinstance(state,tuple):#判断是否为元组，如果是元组则需要将其中的数组部分带入
                state_array = state[0]
            else:
                state_array = state
            state = torch.tensor(np.array([state_array]), dtype=torch.float).to(self.device)#这一行将当前状态 state 转换为 PyTorch 张量，并将其移动到设备 self.device 上。这是因为神经网络需要在相同的设备上处理数据。
            #self.q_net(state) 返回一个包含每个动作的 Q 值的张量，argmax() 方法用于找到最大 Q 值的动作索引，然后 .item() 方法将其转换为标量值。
            q_values = self.q_net(state)
            action = torch.sigmoid(q_values).cpu().detach().numpy()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)# 是 PyTorch 中的操作，用于改变张量的形状。具体来说，.view() 方法用于重新构造张量的维度， -1 表示该维度将根据张量的总元素数自动计算。
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  #预测当前状态下采取的动作的 Q 值。gather(1, actions) 用于选择与实际动作匹配的 Q 值。这是为了计算当前 Q 值的估计。
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)#max(1) 是行维度，返回一个元组，其中包含两个张量：第一个张量是沿指定维度的最大值，第二个张量是对应的索引。max(0)是列维度
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:#在这里，代理检查是否达到了目标网络更新的频率。
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # #如果达到目标网络更新频率，代理将目标网络的参数更新为当前 Q 网络的参数。这是为了稳定训练和减小目标网络的变化。
        self.count += 1



'''
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()
'''