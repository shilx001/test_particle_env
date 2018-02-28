import numpy as np
import matplotlib.pyplot as plt
import pickle

average_reward=pickle.load(open("average_reward450"))

plt.plot(np.array(average_reward[25:])/200,label='CommNet')

plt.title("Avarage reward for each episode")
plt.xlabel('episode')
plt.ylabel('reward')
plt.legend(loc='lower right')
plt.show()