'''
A belief state RL task from the lab of Adil Ghani Khan.

Compared to VisualOlfactoryAttentionSwitchEnv,
 here starting blank cue is not shown as we don't need the extra initial time step.

Version 0:

Block Visual:
2 time steps in a trial:
 step 0: lick is punished, one of 2 visual stimuli is shown
 step 2: lick leads to reward if visual cue 1, punishment if visual cue 2

Block Olfactory:
2 or 3 time steps in a trial:
 30%
 step 1: lick is punished, one of 2 odor stimuli is given
 step 2: lick leads to reward if odor 1, punishment if odor 2
 70%
 step 1: lick is punished, one of 2 visual stimuli is shown
 step 2: lick is punished, one of 2 odor stimuli is given
 step 3: lick leads to reward if odor 1, punishment if odor 2

We transition to the other block after 20 consecutively correct trials.
A lick for cue 1 or a no lick for cue 2 in the final reward step counts as correct response.
No lick in all other time steps in these trials is also required.

Aditya Gilra, 31 Aug 2021.
'''

from gym import Env
from gym.spaces import Discrete
from gym.utils import colorize, seeding
import numpy as np
import sys

class VisualOlfactoryAttentionSwitchNoBlankEnv(Env):
    blocks = ['visual','olfactory']
    visual_stimuli = ['|||','///']
    olfactory_stimuli = ['odor1', 'odor2']
    actions = ['NoLick','Lick']
    observations = visual_stimuli + olfactory_stimuli + ['end']

    def __init__(self, reward_size=10, punish_factor=0.5, lick_without_reward_factor=0.2):
        super(VisualOlfactoryAttentionSwitchNoBlankEnv, self).__init__()

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
        self.observation_number = 4

    def _needless_lick(self, action):
        # licking to end or blank stimulus,
        #  or visual stimulus in olfactory block, is wasteful
        if action == 1:
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
                # licking to blank stimulus is wasteful
                self._needless_lick(action)

                # one of visual stimuli is shown now
                self.observation_number = \
                        int(self.np_random.uniform()*2)

                # lick on next time step if first visual stimuli
                if self.observation_number == 0:
                    self.target_action_number = 1
                    
            elif self.time_index == 1:
                self._reward_and_end_trial(0,action,last_target_action_number)
                
        # olfactory block
        elif self.block_number == 1:
            if self.time_index == 0:
                # licking to blank stimulus is wasteful
                self._needless_lick(action)

                if self.np_random.uniform() < 0.7:
                    # one of visual stimuli is shown now, unrewarded
                    self.observation_number = \
                            int(self.np_random.uniform()*2)
                else:
                    # one of olfactory stimuli is shown now
                    self.observation_number = \
                            2 + int(self.np_random.uniform()*2)
                    # lick on next time step if first olfactory stimuli
                    if self.observation_number == 2:
                        self.target_action_number = 1
                    
            elif self.time_index == 1:
                if self.observation_number >= 2:
                    self._reward_and_end_trial(2,action,last_target_action_number)
                else:
                    # licking to visual stimulus in olfactory block is wasteful
                    self._needless_lick(action)
                        
                    # one of olfactory stimuli is shown now
                    self.observation_number = \
                            2 + int(self.np_random.uniform()*2)

                    # lick on next time step if first olfactory stimuli
                    if self.observation_number == 2:
                        self.target_action_number = 1

            elif self.time_index == 2:
                self._reward_and_end_trial(2,action,last_target_action_number)

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
        
