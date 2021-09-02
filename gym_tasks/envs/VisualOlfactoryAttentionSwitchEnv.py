'''
A belief state RL task from the lab of Adil Ghani Khan.

Aditya Gilra, 31 Aug 2021.
'''

from gym import Env
from gym.spaces import Discrete
from gym.utils import colorize, seeding
import numpy as np
import sys

class VisualOlfactoryAttentionSwitchEnv(Env):
    blocks = ['visual','olfactory']
    visual_stimuli = ['|||','///']
    olfactory_stimuli = ['odor1', 'odor2']
    actions = ['NoLick','Lick']
    observations = ['blank'] + visual_stimuli + olfactory_stimuli + ['end']

    def __init__(self, reward_size=10, punish_factor=0.5, lick_without_reward_factor=0.2):
        super(VisualOlfactoryAttentionSwitchEnv, self).__init__()

        # reward for correct response
        self.reward_size = reward_size
        # wrong response means reward of -punish_factor*reward_size
        self.punish_factor = punish_factor
        # licking without reward is wasteful, so
        #  reward of -lick_without_reward_factor*reward_size
        self.lick_without_reward_factor = lick_without_reward_factor
        
        self.observation_space = Discrete(len(self.observations))
        self.action_space = Discrete(len(self.actions))

        # various attributes that are set by seed() and reset()
        self.block_number = None
        self.observation_number = None
        self.last_action = None
        self.reward = None
        self.consecutive_correct_number = None

        self.time_index = None
        self.trial_number = None
        self.done_trial = None

        self.np_random = None

        # set / reset some attributes of this class
        self.seed()
        self.reset()
        
        self.outfile = sys.stdout

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.block_number = 0
        self.observation_number = 0
        self.target_action_number = 0
        self.last_action = None
        self.reward = 0
        self.consecutive_correct_number = 0

        self.trial_number = 0
        self.time_index = 0
        self.done_trial = False

        return self.observation_number

    def _reward_and_end_trial(self, target_observation_number, action, last_target_action_number):
        # give reward / punishement
        if self.observation_number == target_observation_number:
            if action == last_target_action_number:
                # give reward
                self.reward = self.reward_size
                self.consecutive_correct_number += 1
            else:
                # give punishment, restart count of consecutive correct responses
                self.reward = -self.punish_factor*self.reward_size
                self.consecutive_correct_number = 0

        if self.observation_number == target_observation_number+1:
            if action == self.target_action_number:
                # no reward, but counted as a correct trial
                self.consecutive_correct_number += 1
            else:
                # give punishment, restart count of consecutive correct responses
                self.reward = -self.punish_factor*self.reward_size
                self.consecutive_correct_number = 0

        # end trial
        self.done_trial = True        
        # end state is shown here
        self.observation_number = 5

    def _needless_lick(self, action):
        # licking to blank stimulus (or visual stimulus in olfactory block) is wasteful
        if action != self.target_action_number:
            self.reward = -self.lick_without_reward_factor*self.reward_size
            self.consecutive_correct_number = 0

    def step(self, action):
        assert self.action_space.contains(action)

        if self.consecutive_correct_number >= 20:
            self.block_number = 1 - self.block_number
            self.consecutive_correct_number = 0

        self.time_index += 1
        
        # if the last trial had finished, start a new trial
        if self.done_trial:
            self.done_trial = False
            self.trial_number += 1
            self.time_index = 0

        # set default variables, unless changed below
        self.reward = 0
        last_target_action_number = self.target_action_number
        self.target_action_number = 0

        # visual block
        if self.block_number == 0:
            if self.time_index == 0:
                # blank stimulus is shown now
                self.observation_number = 0
                
            elif self.time_index == 1:
                # licking to blank stimulus is wasteful
                self._needless_lick(action)

                # one of visual stimuli is shown now
                self.observation_number = \
                        1 + int(self.np_random.uniform()*2)

                # lick on next time step if first visual stimuli
                if self.observation_number == 1:
                    self.target_action_number = 1
                    
            elif self.time_index == 2:
                self._reward_and_end_trial(1,action,last_target_action_number)
                
        # olfactory block
        elif self.block_number == 1:
            if self.time_index == 0:
                # blank stimulus is shown now
                self.observation_number = 0
                
            elif self.time_index == 1:
                # licking to blank stimulus is wasteful
                self._needless_lick(action)

                if self.np_random.uniform() < 0.7:
                    # one of visual stimuli is shown now, unrewarded
                    self.observation_number = \
                            1 + int(self.np_random.uniform()*2)
                else:
                    # one of olfactory stimuli is shown now
                    self.observation_number = \
                            3 + int(self.np_random.uniform()*2)
                    # lick on next time step if first olfactory stimuli
                    if self.observation_number == 3:
                        self.target_action_number = 1
                    
            elif self.time_index == 2:
                if self.observation_number >= 3:
                    self._reward_and_end_trial(3,action,last_target_action_number)
                else:
                    # licking to visual stimulus in olfactory block is wasteful
                    self._needless_lick(action)
                        
                    # one of olfactory stimuli is shown now
                    self.observation_number = \
                            3 + int(self.np_random.uniform()*2)

                    # lick on next time step if first olfactory stimuli
                    if self.observation_number == 3:
                        self.target_action_number = 1

            elif self.time_index == 3:
                self._reward_and_end_trial(3,action,last_target_action_number)

        self.last_action = action

        return self.observation_number, self.reward, self.done_trial, \
                        {'target_act': self.target_action_number,\
                        'block': self.block_number}

    def render(self, mode='human'):
        self.outfile.write("block: "+self.blocks[self.block_number]+", ")
        self.outfile.write("trial number: "+str(self.trial_number)+", ")
        self.outfile.write("time step: "+str(self.time_index)+", ")
        self.outfile.write("last action: "+("None" if self.last_action is None else str(self.actions[self.last_action]))+", ")
        self.outfile.write("last reward: "+str(self.reward)+", ")
        self.outfile.write("stimulus: "+self.observations[self.observation_number]+", ")
        self.outfile.write("next target action: "+self.actions[self.target_action_number]+".\n")
        return
        
