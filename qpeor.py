import math
import random
import gymnasium as gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from AUV_action.APF import Artificial_Potential_Field
from AUV_action.AUV_based import Base_Parameters, Environment
from map.simulation_map.base_map import base_map
from map.simulation_map.obstacle import Obstacles


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
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)#建立输入层和隐藏层神经网络，state_dim定义为state的维度，hidden_dim自己定义为128
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)#建议输出层神经网络，action_dim定义为action的维度

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)

class DQN:
    ''' DQN算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim,
                          self.action_dim).to(device)  # Q网络
        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            if isinstance(state,tuple):#判断是否为元组，如果是元组则需要将其中的数组部分带入
                state_array = state[0]
            else:
                state_array = state
            state = torch.tensor(np.array([state_array]), dtype=torch.float).to(self.device)#这一行将当前状态 state 转换为 PyTorch 张量，并将其移动到设备 self.device 上。这是因为神经网络需要在相同的设备上处理数据。
            #self.q_net(state) 返回一个包含每个动作的 Q 值的张量，argmax() 方法用于找到最大 Q 值的动作索引，然后 .item() 方法将其转换为标量值。
            act = self.q_net(state)
            act1 = self.q_net(state).argmax()
            action = self.q_net(state).argmax().item()#在这里，代理使用 Q 网络对当前状态进行前向传播，以获得每个动作的 Q 值，并选择具有最大 Q 值的动作。
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

lr = 2e-3
num_episodes = 500
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

obstacles = Obstacles()  # 障碍应该按照左下，右下，右上，左上，左下的方式添加
obstacles.add_obstacles([[(10, 3), (10, 5), (12, 5), (12, 3), (10, 3)]])
obstacles.add_obstacles([[(30, 35), (60, 35), (60, 60), (30, 60), (30, 35)]])
obstacles.add_obstacles([[(10, 20), (20, 20), (20, 25), (10, 25), (10, 20)]])
base_map1 = base_map(100, 100, 10)
base_map1.set_obstacles(obstacles.get_obstacles())
base_map1.set_goal_point([[80,70]])

apf = Artificial_Potential_Field(base_map1)
for i in range(1000):
    #time.sleep(1)
    a = apf.move()
    if(math.sqrt((a[0]-80)**2+(a[1]-70)**2)<=1):
        break
    #base_map1.show()
    print(a)

base_parameters = Base_Parameters(1,1,1,1)
electric = 100
init_point = [10,10]
goal_point = [80,70]
radius = 5
env = Environment(electric,init_point,goal_point,radius,apf.get_init_points(),base_map1,base_parameters)


#env_name = 'CartPole-v1'
#env = gym.make(env_name,render_mode = 'human')
#env = gym.make(env_name)
random.seed(0)
np.random.seed(0)
#if hasattr(env, 'seed'):
#    env.seed(0)
#elif hasattr(env.unwrapped, 'seed'):
#    env.unwrapped.seed(0)

torch.manual_seed(0)
replay_buffer = ReplayBuffer(buffer_size)
#state_dim = env.observation_space.shape[0]#是 NumPy 数组的属性，用于获取数组的第一个维度的大小。
state_dim = 12
#action_dim = env.action_space.n
action_dim = 5
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
            target_update, device)



return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            state = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward,done,truncated = env.step(action)
                if isinstance(state, tuple):
                    state = state[0]
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward
                # 当buffer数据的数量超过一定值后,才进行Q网络训练
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    agent.update(transition_dict)
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)
