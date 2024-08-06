import time
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from Deep_Q_learning.DQN import ReplayBuffer, DQN
from Deep_Q_learning.DQN_environment import DQN_Environment
from Deep_Q_learning.DQN_parameters import State, Init_Parameters, DQN_Parameter

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

#agent.load('q_network_final.pth')
return_list = []
for i in range(15):
    '''
    if i==5:
        init_parameters.set_init_start_point([14,81])
        init_parameters.set_init_target_point([82,5])
        agent.set_epsilon(epsilon)
    if i==10:
        init_parameters.set_init_start_point([81,83])
        init_parameters.set_init_target_point([20,21])
    '''
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            state = env.reset()
            done = False
            print(datetime.now())
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
            return_list.append(episode_return)
            #print("reward:"+str(episode_return))
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)
    agent.save(f'q_network_iteration_{i}.pth')

    # 最终保存模型参数
agent.save('q_network_final.pth')