import numpy as np
import matplotlib.pyplot as plt
import pickle

average_reward=pickle.load(open("average_reward_bicnet"))
average_reward_fc=pickle.load(open('average_reward_fc'))
average_reward_lstm=pickle.load(open('average_reward450'))

plt.plot(np.array(average_reward[25:])/200,label='BiCNet')
plt.plot(np.array(average_reward_fc[25:450])/200,label='Fully-Connected')
plt.plot(np.array(average_reward_lstm[25:])/200,label='BiLSTM')

plt.title("Avarage reward for each episode")
plt.xlabel('episode')
plt.ylabel('reward')
plt.legend(loc='lower right')
plt.show()