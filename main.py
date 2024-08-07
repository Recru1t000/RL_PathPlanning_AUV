import time
from datetime import datetime
import csv
import os
import numpy as np
import torch
from tqdm import tqdm

from Deep_Q_learning.DQN import ReplayBuffer, DQN
from Deep_Q_learning.DQN_environment import DQN_Environment
from Deep_Q_learning.DQN_parameters import State, Init_Parameters, DQN_Parameter
from map.train_points import TrainPoints


def initialize_csv(filepath):
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Iteration', 'Episode', 'init_point','target_point','Return', 'Datetime','Power'])
def save_to_csv(filepath, iteration, episode, init_point,target_point,episode_return,power):
    with open(filepath, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([iteration, episode, init_point,target_point,episode_return, datetime.now(),power])
#设计掩码
def select_action(action_array):
    threshold = 0.8
    mask = np.zeros_like(action_array)
    if np.any(action_array > threshold):
        mask[action_array > threshold] = 1
    else:
        mask[np.argmax(action_array)] = 1
    return mask

lr = 0.001
num_episodes = 500
state_dim = 13
hidden_dim = 64
action_dim = 15
gamma = 0.98
epsilon = 0.99
target_update = 100
buffer_size = 10000
minimal_size = 1000
batch_size = 64
epsilon_min = 0.01
epsilon_decay = 0.995
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

DQN_parameter = DQN_Parameter(state_dim, hidden_dim, action_dim, lr, gamma,
                 epsilon, target_update, device,epsilon_min,epsilon_decay,buffer_size)
torch.manual_seed(0)
replay_buffer = ReplayBuffer(buffer_size)


agent = DQN(DQN_parameter)
init_parameters = Init_Parameters()



env = DQN_Environment(init_parameters)
#env.reset()
#next_state,reward,done,truncated,_= env.step([0,1,0.9,0])
#按上下左右训练，本质上每个格子都是从上下左右进行移动的
agent.load('q_network_final.pth')
csv_filepath = 'training_returns.csv'
initialize_csv(csv_filepath)
return_list = []
#agent.set_epsilon(epsilon)
#epsilon_list = [epsilon,epsilon,epsilon,epsilon,epsilon,epsilon,epsilon,epsilon]
train_points = TrainPoints()
down_up = train_points.get_down_up()
left_right = train_points.get_left_right()
ld_ru = train_points.get_ld_ru()
for i in range(120):
    if i%8==0:#下-上
        down_up_points = down_up.pop(0)
        init_parameters.set_init_start_point(down_up_points[0])
        init_parameters.set_init_target_point(down_up_points[1])
        down_up.append(down_up_points)
    elif i%8==1:#左-右
        left_right_points = left_right.pop(0)
        init_parameters.set_init_start_point(left_right_points[0])
        init_parameters.set_init_target_point(left_right_points[1])
        left_right.append(left_right_points)
    else:
        ld_ru_points = ld_ru.pop(0)
        init_parameters.set_init_start_point(ld_ru_points[0])
        init_parameters.set_init_target_point(ld_ru_points[1])
        ld_ru.append(ld_ru_points)
    '''
    elif i%8==2:#上-下
        init_parameters.set_init_start_point([33, 84])
        init_parameters.set_init_target_point([33, 20])
    elif i%8==3:  #左-右
        init_parameters.set_init_start_point([16, 43])
        init_parameters.set_init_target_point([80, 43])
    '''
    with tqdm(total=int(num_episodes / 10), desc='I %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):

            episode_return = 0
            state = env.reset()
            done = False
            dt = datetime.now()
            print(" "+str(dt.hour)+":"+str(dt.minute)+":"+str(dt.second))
            while not done:
                #输出观看
                init_parameters.set_print_range(i)

                action = agent.take_action(state,init_parameters)
                '''
                if i<2:
                    action = 0
                '''
                next_state, reward,done,truncated, _ = env.step(action+1)
                if isinstance(state, tuple):
                    state = state[0]
                #action = select_action(action)
                #replay_buffer.add(state, action, reward, next_state, done)
                agent.buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward
                # 当buffer数据的数量超过一定值后,才进行Q网络训练
                if agent.buffer.size() > minimal_size:
                    agent.prior_update(batch_size=batch_size, beta=0.4)
                '''
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    b_a = np.vstack(b_a)  # 将 actions 转换为二维数组形式
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    agent.update(transition_dict)
                '''
            #env.show_the_path()

            agent.set_epsilon(max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min))
            #epsilon_list[i%8] = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)
            return_list.append(episode_return)
            #print("reward:"+str(episode_return))
            save_to_csv(csv_filepath, i, i_episode,init_parameters.get_init_start_point(),init_parameters.get_init_target_point(),episode_return,state[0]*init_parameters.get_init_power())

            pbar.set_postfix({
                'epi':
                '%d' % (num_episodes / 10 * i + i_episode + 1),

                'reward':'%.1f' % return_list[-1]
            })
            pbar.update(1)
    agent.save(f'q_network_iteration_{i}.pth')

    # 最终保存模型参数
agent.save('q_network_final.pth')


'''
            if i_episode<= 40:#左下-右上
                init_parameters.set_init_start_point([21, 22])
                init_parameters.set_init_target_point([93, 90])
            elif i_episode<=50:#下-上
                init_parameters.set_init_start_point([41, 9])
                init_parameters.set_init_target_point([41, 80])
            elif i_episode <= 90:#右下-左上
                init_parameters.set_init_start_point([83, 14])
                init_parameters.set_init_target_point([12, 85])
            elif i_episode<=100:#右-左
                init_parameters.set_init_start_point([76, 43])
                init_parameters.set_init_target_point([10, 43])
            elif i_episode<=140:#右上-左下
                init_parameters.set_init_start_point([83, 76])
                init_parameters.set_init_target_point([27,30])
            elif i_episode<=150:#上-下
                init_parameters.set_init_start_point([52, 81])
                init_parameters.set_init_target_point([53, 10])
            elif i_episode<=190:#左上-右下
                init_parameters.set_init_start_point([17, 87])
                init_parameters.set_init_target_point([91, 15])
            else:#左-右
                init_parameters.set_init_start_point([18, 56])
                init_parameters.set_init_target_point([78, 60])
'''