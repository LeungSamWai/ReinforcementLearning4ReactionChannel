import numpy as np
import matplotlib.pyplot as plt

def potential(q):
    qx = q[...,0:1]
    qy = q[...,1:2]
    V = 3*np.exp(-qx**2-(qy-1/3)**2)-3*np.exp(-qx**2-(qy-5/3)**2)-5*np.exp(-(qx-1)**2-qy**2)-5*np.exp(-(qx+1)**2-qy**2)+0.2*qx**4+0.2*(qy-0.2)**4
    return V

def gradV(q):

    qx = q[:, 0:1]
    qy = q[:, 1:2]

    Vx = (-2*qx)*3*np.exp(-qx**2-(qy-1/3)**2)\
         -(-2*qx)*3*np.exp(-qx**2-(qy-5/3)**2)\
         -(-2*(qx-1))*5*np.exp(-(qx-1)**2-qy**2)\
         -(-2*(qx+1))*5*np.exp(-(qx+1)**2-qy**2)\
         +4*0.2*qx**3

    Vy = (-2*(qy-1/3))*3*np.exp(-qx**2-(qy-1/3)**2)\
         -(-2*(qy-5/3))*3*np.exp(-qx**2-(qy-5/3)**2)\
         -(-2*qy)*5*np.exp(-(qx-1)**2-qy**2)\
         -(-2*qy)*5*np.exp(-(qx+1)**2-qy**2)\
         +4*0.2*(qy-0.2)**3

    return np.concatenate((Vx, Vy), axis=1)

class actionspace:
    def __init__(self):
        self.low = -5
        self.high = 5
        self.shape = 2

class MdEnviron:
    def __init__(self, reward_type, gamma, maxaction, dt, bound, T, beta):
        self.Aorg_pt = np.array([-1, 0])
        self.Borg_pt = np.array([1, 0])
        self.repeat = 50
        self.low = -maxaction
        self.high = maxaction
        self.shape = 2
        self.reward_type = reward_type
        self.gamma = gamma
        self.dt = dt
        self.bound = bound
        self.T = T
        self.beta = beta
    def reset(self):
        xL, xR = -2, 2
        yB, yT = -1.25, 2
        self.qnow = np.random.rand(2)
        self.qnow[0] = (xR - xL) * self.qnow[0] + xL
        self.qnow[1] = (yT - yB) * self.qnow[1] + yB
        return self.qnow

    def step(self, action):
        qnext = self.next_action(self.qnow, action)
        # reward = self.get_reward(q0=qnext, repeat=50, T=0.5)#
        reward, logprob = self.get_reward(q0=qnext, repeat=self.repeat, T=self.T)
        self.qnow = qnext
        return self.qnow, reward, logprob<=self.bound, None

    def get_reward(self, q0, repeat, T):
        beta = self.beta#6.67
        trajectories = self.MD(q0=q0, repeat=repeat, everyN=1, T=T, dt=5e-4, beta=beta)#3.5)

        
        numA, numB = self.h_first(trajectories)

        if self.reward_type == 'log_product':
            logprob = -beta * potential(q0)[..., 0]
            if numA + numB == 0:
                sumAB = 1
            else:
                sumAB = numA + numB
            if logprob <= self.bound:
                reward = -50
            else:
                reward = np.log(np.exp(logprob)*numA*numB/sumAB/sumAB+1)
            print(q0, numA, numB, logprob, '  reward: ', reward)
            return reward, logprob


    def MD(self, q0, repeat, everyN, T, dt=5e-3, beta=3.5):
        q0 = np.tile(q0, (repeat, 1))
        Nsteps = int(T / dt)
        trajectories = np.empty((Nsteps // everyN + 1, q0.shape[0], 2))
        q = q0
        for i in range(Nsteps):
            if i % everyN == 0:
                trajectories[i // everyN, :] = q
            q = self.next(q, dt, beta)
        trajectories[-1, :] = q

        return trajectories

    def h_A(self, q):
        qx = q[..., 0]
        qy = q[..., 1]
        return np.logical_and(potential(q)[...,0]<-2, qx<=-0.1)
    
    def h_first(self, q):
        qsub = q[::10]
        qxsub = qsub[..., 0]
        q_pot_basin = potential(qsub)[...,0]<-2
        bs = qsub.shape[1]
        numA = 0
        numB = 0
        for i in range(bs):
            if np.sum(q_pot_basin[:,i])>0:
                first_max_idx = np.argmax(q_pot_basin[:,i])
                if qxsub[first_max_idx, i]<=-0.1:
                    numA += 1
                elif qxsub[first_max_idx, i]>=0.1:
                    numB += 1
        return numA, numB
    
    def h_B(self, q):
        qx = q[..., 0]
        qy = q[..., 1]
        return np.logical_and(potential(q)[...,0]<-2, qx>=0.1)

    def next(self, qnow, dt=5e-3, beta=3.5):
        qnext = qnow + (- gradV(qnow)) * dt + np.sqrt(2 * dt / beta) * np.random.randn(*qnow.shape)
        return qnext

    def next_action(self, qnow, action, dt=0.05):
        qnext = qnow + action * dt
        return qnext