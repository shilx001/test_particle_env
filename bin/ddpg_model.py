# -- coding: utf-8 --
import tensorflow as tf
import numpy as np
import gym
import time

#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.0001  # learning rate for actor
LR_C = 0.0002  # learning rate for critic
GAMMA = 0.99  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 5000
BATCH_SIZE = 32
HIDDEN_SIZE = 20


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
                               dtype=np.float64)  # reward变为agent_size维
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound, self.agent_number = a_dim, s_dim, a_bound, agent_number
        self.S = tf.placeholder(tf.float64, [BATCH_SIZE, s_dim * agent_number],
                                's')  # stretch the state dimension into 1-dimension
        self.S_ = tf.placeholder(tf.float64, [BATCH_SIZE, s_dim * agent_number],
                                 's_')  # stretch the state dimension into 1-dimension
        self.R = tf.placeholder(tf.float64, [BATCH_SIZE, agent_number], 'r')  # reward变为Agent_size维

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
        s = np.reshape(s, [self.agent_number * self.s_dim]) * np.ones([BATCH_SIZE, 1])
        action = self.sess.run(self.a, {self.S: s})
        action = action[0, :]
        return np.reshape(action, [self.agent_number, self.a_dim])

    def learn(self):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]  # transitions
        bs = np.reshape(bt[:, :, :self.s_dim], [-1, self.s_dim * self.agent_number])  # states
        ba = np.reshape(bt[:, :, self.s_dim:self.s_dim + self.a_dim], [-1, self.agent_number * self.a_dim])
        br = bt[:, :, self.s_dim + self.a_dim]  # rewards
        bs_ = np.reshape(bt[:, :, -self.s_dim:], [-1, self.agent_number * self.s_dim])  # next_state

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack(
            (np.reshape(s, [self.agent_number, self.s_dim]), np.reshape(a, [self.agent_number, self.a_dim]),
             np.reshape(r, [self.agent_number, 1]), np.reshape(s_, [self.agent_number, self.s_dim])))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def shape(self, tensor):
        s = tensor.get_shape()
        return tuple([s[i].value for i in range(0, len(s))])

    def mlp(self, scope_name, x, c):
        # communication core for CommNet
        # batch computing for efficiency
        t_shape = self.shape(x)
        dim = t_shape[2]
        with tf.variable_scope(scope_name):
            w1_x = tf.multiply(
                tf.Variable(tf.truncated_normal([1, dim, HIDDEN_SIZE], stddev=0.1, dtype=tf.float64)),
                tf.ones([BATCH_SIZE, dim, HIDDEN_SIZE], dtype=tf.float64))
            hidden1_x = tf.nn.relu(tf.matmul(x, w1_x))
            w1_c = tf.multiply(
                tf.Variable(tf.truncated_normal([1, dim, HIDDEN_SIZE], stddev=0.1, dtype=tf.float64)),
                tf.ones([BATCH_SIZE, dim, HIDDEN_SIZE], dtype=tf.float64))
            hidden1_c = tf.nn.relu(tf.matmul(x, w1_c))
            w2 = tf.multiply(
                tf.Variable(tf.truncated_normal([1, HIDDEN_SIZE * 2, HIDDEN_SIZE], stddev=0.1, dtype=tf.float64)),
                tf.ones([BATCH_SIZE, HIDDEN_SIZE * 2, HIDDEN_SIZE], dtype=tf.float64))
            return tf.nn.relu(tf.matmul(tf.concat([hidden1_x, hidden1_c], axis=2), w2))

    def attention(self, scope, input_value):
        # input: [batch_size, agent_number, hidden_size]
        with tf.variable_scope(scope):
            w1 = tf.multiply(
                tf.Variable(tf.truncated_normal([1, HIDDEN_SIZE, self.agent_number], stddev=0.1, dtype=tf.float64)),
                tf.ones([BATCH_SIZE, HIDDEN_SIZE, self.agent_number], dtype=tf.float64))
            weight = tf.nn.softmax(
                tf.matmul(input_value, w1))  # attention weight: [batch_size, self.agent_number, self.agent_number]
            output_value = tf.matmul(weight, input_value)
        return output_value

    def _build_a(self, s, reuse=None, custom_getter=None):  # BiCNet/MemNet
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            input_state = tf.reshape(s, [BATCH_SIZE, self.agent_number, self.s_dim])
            c = tf.zeros_like(input_state)
            net = self.mlp('a_layer1', input_state, c)
            c = tf.reshape(tf.reduce_mean(net, axis=1), [BATCH_SIZE, 1, HIDDEN_SIZE])
            net = self.mlp('a_layer2', net, c)  # output: [batch_size, self.agent_number, hidden_zise]
            net = self.attention('attention_layer', net)  # add attention layer
            w = tf.multiply(
                tf.Variable(tf.truncated_normal([1, HIDDEN_SIZE, self.a_dim], stddev=0.1, dtype=tf.float64)),
                tf.ones([BATCH_SIZE, HIDDEN_SIZE, self.a_dim], dtype=tf.float64))
            x = tf.nn.tanh(tf.matmul(net, w)) * self.a_bound
            return tf.reshape(x, [BATCH_SIZE, self.agent_number * self.a_dim])

    def _build_c(self, s, a, reuse=None, custom_getter=None):  # BiCNet/MemNet
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            input_action = tf.reshape(a, [BATCH_SIZE, self.agent_number, self.a_dim])
            input_state = tf.reshape(s, [BATCH_SIZE, self.agent_number, self.s_dim])
            input_all = tf.concat([input_action, input_state], axis=2)
            c = tf.zeros_like(input_state)
            net = self.mlp('c_layer1', input_all, c)
            c = tf.reshape(tf.reduce_mean(net, axis=1), [BATCH_SIZE, 1, HIDDEN_SIZE])
            net = self.mlp('c_layer2', net, c)
            net=self.attention('attention',net)
            w = tf.multiply(tf.Variable(tf.truncated_normal([1, HIDDEN_SIZE, 1], stddev=0.1, dtype=tf.float64)),
                            tf.ones([BATCH_SIZE, HIDDEN_SIZE, 1], dtype=tf.float64))
            Q = tf.nn.tanh(tf.matmul(net, w))
            return tf.reshape(Q, [BATCH_SIZE, self.agent_number])
