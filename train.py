import torch
import numpy as np
from TD3 import TD3
from utils import ReplayBuffer
from env import MdEnviron
import os
import matplotlib.pyplot as plt
import sys
import argparse

parser = argparse.ArgumentParser(description='PyTorch Density Function Training')

# Checkpoints
parser.add_argument('--env_name', default='', type=str)
parser.add_argument('--gpu_id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--reward_type', type=str, help='product, logsum')
parser.add_argument('--gamma', type=float, default=1.0, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--maxstep', type=int, default=50, help='Number of steps.')
parser.add_argument('--maxepisode', type=int, default=1000, help='Number of episodes.')
parser.add_argument('--maxaction', type=float, default=5, help='action magnititude.')
parser.add_argument('--dt', type=float, default=0.05, help='action magnititude.')
parser.add_argument('--bound', type=float, default=-35, help='bound to end the game.')
parser.add_argument('--T', type=float, default=2, help='Shoot time')
parser.add_argument('--beta', type=float, default=6.67, help='beta inverse of temp.')
parser.add_argument('--noise', type=float, default=0.1, help='noise')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
print(state)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id


if not os.path.exists('preTrained'):
    os.makedirs('preTrained')

# save some training progress
class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "w+")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

# use for visualization
def potential(q):
    qx = q[...,0:1]
    qy = q[...,1:2]
    V = 3*np.exp(-qx**2-(qy-1/3)**2)-3*np.exp(-qx**2-(qy-5/3)**2)-5*np.exp(-(qx-1)**2-qy**2)-5*np.exp(-(qx+1)**2-qy**2)+0.2*qx**4+0.2*(qy-0.2)**4
    return V[...,0]

N = 100
xx = np.linspace(-2, 2, N)
yy = np.linspace(-1.25, 2, N)
[X, Y] = np.meshgrid(xx, yy)  # 100*100

pts = np.concatenate((np.expand_dims(X, axis=2), np.expand_dims(Y, axis=2)), axis=2)
W = potential(pts)

def train():
    ######### Hyperparameters #########
    env_name = args.env_name  #"mdenvironment_beta0p1_newreward"
    log_interval = 1           # print avg reward after interval
    random_seed = 0
    gamma = 0.99                # discount for future rewards
    batch_size = 100            # num of transitions sampled from replay buffer
    lr = 0.001
    exploration_noise = args.noise#0.1 
    polyak = 0.995              # target policy update parameter (1-tau)
    policy_noise = 0.2          # target policy smoothing noise
    noise_clip = 0.5
    policy_delay = 2            # delayed policy updates parameter
    max_episodes = args.maxepisode        # max num of episodes
    max_timesteps = args.maxstep      # max timesteps in one episode
    directory = "./preTrained/{}".format(env_name) # save trained models
    if not os.path.exists(directory):
        os.mkdir(directory)
    filename = "TD3_{}_{}".format(env_name, random_seed)

    with open(os.path.join(directory, "Config.txt"), 'w+') as f:
        for (k, v) in args._get_kwargs():
            f.write(k + ' : ' + str(v) + '\n')

    # Save all print out informations
    log_file_name = os.path.join(directory, "output.log")
    sys.stdout = Logger(log_file_name)

    figdirectory = "./fig_{}".format(env_name)
    if not os.path.exists(figdirectory):
        os.mkdir(figdirectory)
    ###################################
    
    env = MdEnviron(args.reward_type, args.gamma, args.maxaction, args.dt, args.bound, args.T, args.beta)
    state_dim = 2
    action_dim = 2
    max_action = 1
    
    policy = TD3(lr, state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer()
    
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        # env.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    # logging variables:
    avg_reward = 0
    ep_reward = 0
    log_f = open(os.path.join(directory, "log.txt"),"w+")
    
    # training procedure:
    for episode in range(1, max_episodes+1):
        state = env.reset()
        trajectory = []
        trajectory.append(state)
        for t in range(max_timesteps):
            # select action and add exploration noise:
            action = policy.select_action(state)
            action = action + np.random.normal(0, exploration_noise, size=env.shape)
            
            # take action in env:
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add((state, action, reward, next_state, float(done)))
            state = next_state
            trajectory.append(state)

            avg_reward += reward
            ep_reward += reward
            
            # if episode is done then update policy:
            if done or t==(max_timesteps-1):
                policy.update(replay_buffer, t, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
                break

        trajectory = np.array(trajectory)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.contourf(X, Y, W, levels=29)
        ax.plot(trajectory[:,0], trajectory[:,1])
        ax.scatter(trajectory[0,0], trajectory[0,1], color='r', s=20)

        # logging updates:
        log_f.write('{},{}\n'.format(episode, ep_reward))
        log_f.flush()
        ep_reward = 0

        
        if episode > 300:
            policy.save(directory, filename)

        if episode % log_interval == 0:
            avg_reward = int(avg_reward / log_interval)
            print("Episode: {}\tAverage Reward: {}".format(episode, avg_reward))
            avg_reward = 0

        plt.savefig(figdirectory+'/{}.png'.format(episode))
        plt.clf()
        plt.close()

if __name__ == '__main__':
    train()
    
