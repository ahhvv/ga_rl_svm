import matplotlib.pyplot as plt
import os
import time
import gym
import numpy as np
import tensorflow as tf
import tensorlayer as tl
import argparse
from sklearn import svm
import data

ALG_NAME = 'SVM-RL'
# add arguments in command  --train/test
parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=False)
args = parser.parse_args()

#####################  hyper parameters  ####################

ENV_ID = 'SVM-RL'  # environment id
RANDOM_SEED = 2  # random seed, can be either an int number or None
RENDER = False  # render while training

ALG_NAME = 'AC'
TRAIN_EPISODES = 200  # number of overall episodes for training
TEST_EPISODES = 10  # number of overall episodes for testing
MAX_STEPS = 300  # maximum time step in one episode
LAM = 0.9  # reward discount in TD error
LR_A = 0.001  # learning rate for actor
LR_C = 0.01  # learning rate for critic

# indi = ga_svm.getdata()



class Actor(object):
    #改成2个输出？，一个输出更新步长 一个输出动作
    #state 就1维 SVM C的值  action 3维（增加 不变 减小） 步长固定0.1
    def __init__(self, state_dim, action_dim, lr=0.001):
        input_layer = tl.layers.Input([None, state_dim])
        layer = tl.layers.Dense(n_units=32, act=tf.nn.relu6)(input_layer)
        layer = tl.layers.Dense(n_units=action_dim)(layer)
        self.model = tl.models.Model(inputs=input_layer, outputs=layer)
        self.model.train()
        self.optimizer = tf.optimizers.Adam(lr)
    #状态 动作 rd_error
    def learn(self, state, action, td_error):
        with tf.GradientTape() as tape:
            state = state.astype(np.float32)
            _logits = self.model(state)
            _exp_v = tl.rein.cross_entropy_reward_loss(
                logits=_logits, actions=[action], rewards=td_error)
        grad = tape.gradient(_exp_v, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))
    #输出动作概率
    def get_action(self, state, greedy=False):
        #
        state = state.astype(np.float32)
        _logits = self.model(state)
        _prob = tf.nn.softmax(_logits).numpy()
        if greedy:
            return np.argmax(_prob.ravel())
        return tl.rein.choice_action_by_probs(_prob.ravel(), [-1, 0, 1])

class Critic(object):

    def __init__(self, state_dim, lr=0.01):
        input_layer = tl.layers.Input([None, state_dim])
        layer = tl.layers.Dense(n_units=32, act=tf.nn.relu)(input_layer)
        layer = tl.layers.Dense(n_units=1, act=None)(layer)

        self.model = tl.models.Model(inputs=input_layer, outputs=layer)
        self.model.train()
        self.optimizer = tf.optimizers.Adam(lr)
    #
    def learn(self, state, reward, state_):
        d = 1

        with tf.GradientTape() as tape:
            state = state.astype(np.float32)
            v = self.model(state)
            v_ = self.model(np.array([state_]))
            td_error = reward + d * LAM * v_ - v
            loss = tf.square(td_error)
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return td_error

class Agent():
    # 1 3
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        #初始化两个神经网络
        self.actor = Actor(self.state_dim, self.action_dim, lr=LR_A)
        self.critic = Critic(self.state_dim, lr=LR_C)

    def train(self, c):
        if args.train:
            return self.train_episode(c)
        if args.test:
            self.load()
            # self.test_episode()

    def train_episode(self, c):
        t0 = time.time()
        all_episode_reward = []
        bestfit = 0
        all_episode_acc = []
        for episode in range(TRAIN_EPISODES):
            #state初始状态 svm参数值 c和gamma
            state = np.array([[c]])

            step = 0
            episode_reward = 0
            while True:
                #动作有3种，加 减 不变
                action = self.actor.get_action(state)
                #参数更新步长设置0.1
                a_step = 0.1
                #state更新公式
                state_ = state+action*a_step
                #如果小于0.1 就设置等于0.1
                if(state_< 0.1):
                    state_ = np.array([[0.1]])
                #C值越大越严格 C值越小越松弛
                clf_state = svm.SVC(kernel='rbf', C=state)
                clf_state.fit(data.train_data, data.train_res)
                reward1 = clf_state.score(data.val_data, data.val_res)

                clf_state_ = svm.SVC(kernel='rbf', C=state_)
                clf_state_.fit(data.train_data, data.train_res)
                reward2 = clf_state_.score(data.val_data, data.val_res)
                reward = reward2 - reward1
                #对于不动呢给个惩罚
                if(reward < 0.000001 and reward> - 0.000001):
                    reward = -0.001
                state_ = state_.astype(np.float32)
                episode_reward += reward

                td_error = self.critic.learn(state, reward, state_)
                if action == -1:
                    action = 2
                self.actor.learn(state, action, td_error)

                state = state_
                step += 1
                if step >= MAX_STEPS:
                    print("Early Stopping")  # Hao Dong: it is important for this task
                    break
                # print('Training  | Episode: {}/{}  | Episode Reward: {}  | Running Time: {:.4f} C:{}' \
                #       .format(episode + 1, TRAIN_EPISODES, episode_reward, time.time() - t0, state_))
            if episode == 0:
                all_episode_reward.append(episode_reward)
            else:
                # 计算滑动平均的reward
                all_episode_reward.append(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)
            clf = svm.SVC(kernel='rbf', C=state_)
            clf.fit(data.train_data, data.train_res)
            acc = clf_state_.score(data.val_data, data.val_res)
            all_episode_acc.append(acc)
            bestfit = acc
            print('Training  | Episode: {}/{}  | Episode Reward: {}  | Running Time: {:.4f} C:{} acc:{}'\
                  .format(episode + 1, TRAIN_EPISODES, episode_reward, time.time() - t0, state_, acc))
            # Early Stopping for quick check
            # if step >= MAX_STEPS:
            #     print("Early Stopping")  # Hao Dong: it is important for this task
            #     self.save()
            #     #break
        # 画出图片
        plt.plot(all_episode_reward, label='reward')
        plt.plot(all_episode_acc, label='acc')
        if not os.path.exists('image'):
            os.makedirs('image')
        pic_name = 'c'+str(c)+'.png'
        plt.savefig(os.path.join('image', pic_name))
        plt.close()
        # 返回最后的一代的准确率
        return bestfit


    def save(self):
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        if not os.path.exists(path):
            os.makedirs(path)
        tl.files.save_npz(self.actor.model.trainable_weights, name=os.path.join(path, 'model_actor.npz'))
        tl.files.save_npz(self.critic.model.trainable_weights, name=os.path.join(path, 'model_critic.npz'))
        print('Succeed to save model weights')

    def load(self):
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        tl.files.load_and_assign_npz(name=os.path.join(path, 'model_critic.npz'), network=self.critic.model)
        tl.files.load_and_assign_npz(name=os.path.join(path, 'model_actor.npz'), network=self.actor.model)
        print('Succeed to load model weights')


# 返回最后参数结果
def getFitness(c):
    state_dim = 1
    action_dim = 3
    agent = Agent(state_dim, action_dim)
    a = agent.train(c)
    return a



if __name__ == '__main__':

    a = getFitness(c=1.0)
    print('a:', a)