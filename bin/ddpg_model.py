# -- coding: utf-8 --
import tensorflow as tf
import numpy as np
import gym
import time

#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
GAMMA = 0.99  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 5000
BATCH_SIZE = 32
HIDDEN_SIZE = 20
AGENT_SIZE = 10


###############################  Muti-agent DDPG  ####################################


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound, agent_number):  # a_dim=10,s_dim=520
        '''
        :param a_dim: action dimension for each agent
        :param s_dim: state dimension for each agent (regard as a 1-dimension vector)
        :param a_bound: action bound
        :param agent_number: agent number
        '''
        self.memory = np.zeros((MEMORY_CAPACITY, agent_number, s_dim * 2 + 1 + a_dim),
                               dtype=np.float32)  # reward变为agent_size维
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound, self.agent_number = a_dim, s_dim, a_bound, agent_number
        self.S = tf.placeholder(tf.float32, [None, s_dim * agent_number],
                                's')  # stretch the state dimension into 1-dimension
        self.S_ = tf.placeholder(tf.float32, [None, s_dim * agent_number],
                                 's_')  # stretch the state dimension into 1-dimension
        self.R = tf.placeholder(tf.float32, [None, agent_number], 'r')  # reward变为Agent_size维

        self.a = self._build_a(self.S, )
        q = self._build_c(self.S, self.a, )
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]  # soft update operation
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)  # replaced target parameters
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)  # maximize the q, 这里我改为reduce_sum
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):  # soft replacement happened at here
            q_target = self.R + GAMMA * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        #s needs to stretch into 1-dimention vector
        s=np.reshape(s,[self.agent_number*self.s_dim])
        action=self.sess.run(self.a, {self.S: s[np.newaxis, :]})
        return np.reshape(action,[self.agent_number,self.a_dim])  # np.newaxis意思应该和reshape一样。

    def learn(self):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]  # transitions
        bs = np.reshape(bt[:, :, :self.s_dim], [-1, self.s_dim*self.agent_number])  # states
        ba = np.reshape(bt[:, :, self.s_dim:self.s_dim+self.a_dim], [-1, self.agent_number*self.a_dim])
        br = bt[:, :, self.s_dim+self.a_dim]  # rewards
        bs_ = np.reshape(bt[:, :, -self.s_dim:], [-1, self.agent_number*self.s_dim])  # next_state

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack(
            (np.reshape(s, [self.agent_number, self.s_dim]), np.reshape(a, [self.agent_number, self.a_dim]),
             np.reshape(r, [self.agent_number, 1]), np.reshape(s_, [self.agent_number, self.s_dim])))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(s, 300, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim*self.agent_number, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 300
            w1_s = tf.get_variable('w1_s', [self.s_dim*self.agent_number, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.agent_number * self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, self.agent_number, trainable=trainable)  # Q(s,a)
