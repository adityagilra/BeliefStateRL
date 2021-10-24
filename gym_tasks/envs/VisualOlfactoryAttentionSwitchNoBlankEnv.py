'''
A belief state RL task from the lab of Adil Ghani Khan.

Compared to VisualOlfactoryAttentionSwitchEnv,
 here starting blank cue is not shown as we don't need the extra initial time step.
 rest is same as VisualOlfactoryAttentionSwitchEnv, so this just inherits from that class.

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

from gym_tasks.envs.VisualOlfactoryAttentionSwitchEnv import VisualOlfactoryAttentionSwitchEnv

class VisualOlfactoryAttentionSwitchNoBlankEnv(VisualOlfactoryAttentionSwitchEnv):

    def __init__(self, reward_size=10, punish_factor=0.5, lick_without_reward_factor=0.2):
        super(VisualOlfactoryAttentionSwitchNoBlankEnv, self).__init__(
                    reward_size, punish_factor, lick_without_reward_factor )

        # override some of the attributes set by the parent class, to remove 'blank' cue
        self.observations = self.visual_stimuli + self.olfactory_stimuli + ['end']
        self.observation_space = Discrete(len(self.observations))
        self.end_of_trial_observation_number = len(self.observations)-1

    def _step(self, action, last_target_action_number):
        """
        Local method which computes the next state, reward etc.
        Don't call this directly, call self.step(action).
        """
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
