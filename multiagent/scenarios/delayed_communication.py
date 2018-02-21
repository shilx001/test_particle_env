# -- coding: utf-8 --
import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

# the number of agents and landmarks
NUMBER_OF_AGENTS = 10
NUMBER_OF_LANDMARKS = 15
LANDMARK_POSITION = np.random.uniform(-0.5, 0.5, [NUMBER_OF_LANDMARKS, 2])


AGENT_INFO=np.random.permutation(NUMBER_OF_AGENTS)#agents known of agents.
TARGET=np.random.choice(NUMBER_OF_LANDMARKS,NUMBER_OF_AGENTS)#targets of the known agents

class Scenario(BaseScenario):

    def make_world(self):
        # make the world
        world = World()
        world.agents = [Agent() for i in range(NUMBER_OF_AGENTS)]  # create agents
        world.landmarks = [Agent() for i in range(NUMBER_OF_LANDMARKS)]  # create landmarks
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent=True
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # assign the goal to each agent.

        AGENT_POSITION = np.random.uniform(-1, 1, [NUMBER_OF_AGENTS, 2])


        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None
        #randomize the goal of each agent
        for i, agent in enumerate(world.agents):
            agent.goal_a=world.agents[AGENT_INFO[i]]
            agent.goal_b=world.landmarks[TARGET[i]]
        for i,agent in enumerate(world.agents):
            agent.color=np.array([0.25,0.75,0.25])#set default color,green for agent
            agent.state.p_pos=AGENT_POSITION[i]#set default position
            agent.state.p_vel=np.zeros(world.dim_p)#set default velocity
            #print agent.state.p_pos
        for i,landmark in enumerate(world.landmarks):
            landmark.color=np.array([0.75,0.25,0.25])#red for landmark
            landmark.state.p_pos=LANDMARK_POSITION[i]
            landmark.state.p_vel=np.zeros(world.dim_p)
            #print landmark.state.p_pos
        #print "reset_world!"

    def reward(self, agent, world):
        if agent.goal_a is None or agent.goal_b is None:
            return 0.0
        dist2 = np.sum(np.square(agent.goal_a.state.p_pos - agent.goal_b.state.p_pos))
        return -dist2*np.ones([NUMBER_OF_AGENTS])  # np.exp(-dist2)

    def observation(self,agent,world):
        entity_pos=[]
        for entity in world.landmarks:#与landmark的相对位置
            entity_pos.append(entity.state.p_pos-agent.state.p_pos)
        for entity in world.agents:
            entity_pos.append(entity.state.p_pos-agent.state.p_pos)
        return np.concatenate([agent.state.p_vel]+entity_pos)