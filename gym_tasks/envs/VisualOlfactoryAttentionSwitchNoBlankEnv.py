'''
A belief state RL task from the lab of Adil Ghani Khan.

Compared to VisualOlfactoryAttentionSwitchEnv,
 here starting blank cue is not shown as we don't need the extra initial time step.
 rest is same as VisualOlfactoryAttentionSwitchEnv, so this just inherits from that class.

Version 0:

Block Visual:
2 time steps in a trial:
 step 0: needless lick is mildly punished (internal cost to mouse), one of 2 visual stimuli is shown
 step 2: lick leads to reward if visual cue 1, punishment (in exp, it's a timeout) if visual cue 2

Block Olfactory:
2 or 3 time steps in a trial:
 30%
 step 1: needless lick is mildly punished (internal cost to mouse), one of 2 odor stimuli is given
 step 2: lick leads to reward if odor 1, punishment if odor 2
 70%
 step 1: needless lick is mildly punished (internal cost to mouse), one of 2 visual stimuli is shown
 step 2: needless lick is mildly punished (internal cost to mouse), one of 2 odor stimuli is given
 step 3: lick leads to reward if odor 1, punishment (in exp, it's a timeout) if odor 2

After any block transition, shaping trials are given,
 where always a rewarded visual stimulus is shown in both blocks,
  until the mouse gets 3 consecutive correct responses i.e. lick in visual block, no lick in olfactory block.
 2023 May update: if first_V2O_visual_is_irrelV2 == True,
                    then first trial after V2O block change has irrelevant 'unrewarded' visual cue 2,
                    later trials have visual cue 1 until 3 consecutive correct no-licks.
                  if first_V2O_visual_is_irrelV2 == False,
                    then after V2O block change, irrelevant 'rewarded' visual cue 1 is shown until 3 consecutive correct no-licks.
We transition to the other block after at least 80% correct in past 30 trials,
 and 100% correct in ignoring past 10 irrelevant visual stimuli,
 and there should be at least 30 trials after initial shaping trials after a block transition.
A lick for cue 1 or a no lick for cue 2 in the final reward step counts as correct response.
No lick in all other time steps in these trials is also required.

Aditya Gilra, 31 Aug 2021.
'''

from gym import Env
from gym.spaces import Discrete
import numpy as np
import sys

from gym_tasks.envs.VisualOlfactoryAttentionSwitchEnv import VisualOlfactoryAttentionSwitchEnv

class VisualOlfactoryAttentionSwitchNoBlankEnv(VisualOlfactoryAttentionSwitchEnv):

    def __init__(self, reward_size=10, punish_factor=0.5,
                        lick_without_reward_factor=0.2, seed=1, first_V2O_visual_is_irrelV2=False):
        super(VisualOlfactoryAttentionSwitchNoBlankEnv, self).__init__(
                    reward_size, punish_factor, lick_without_reward_factor, seed )

        # override some of the attributes set by the parent class, to remove 'blank' cue
        self.observations = self.visual_stimuli + self.olfactory_stimuli + ['end']
        self.observation_space = Discrete(len(self.observations))
        self.end_of_trial_observation_number = len(self.observations)-1
        self.first_V2O_visual_is_irrelV2 = first_V2O_visual_is_irrelV2

    def _step(self, action):
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
                if self.shaping_trials_correct < 3:
                    # rewarded visual stimulus is shown until 3 correct in a row
                    self.observation_number = 0
                else:
                    self.observation_number = \
                            int(self.rng.uniform()*2)

                # lick on next time step if first visual stimuli
                if self.observation_number == 0:
                    self.target_action_number = 1
                    
            elif self.time_index == 1:
                self._reward_and_end_trial(0,action)
                
        # olfactory block
        elif self.block_number == 1:
            if self.time_index == 0:
                # licking to blank stimulus is wasteful
                self._needless_lick(action)

                if self.shaping_trials_correct < 3:
                    if self.first_V2O_visual_is_irrelV2 and self.trial_number_from_start_of_block == 0:
                        self.observation_number = 1 # 2023 May update, 1st trial after V2O transition has V2 if first_V2O_visual_is_irrelV2==True
                    else:
                        # to-ignore rewarded visual stimulus V1 is shown until 3 correct in a row
                        self.observation_number = 0
                else:
                    if self.rng.uniform() < 0.7:
                        # one of visual stimuli is shown now, unrewarded
                        self.observation_number = \
                                int(self.rng.uniform()*2)
                    else:
                        # one of olfactory stimuli is shown now
                        self.observation_number = \
                                2 + int(self.rng.uniform()*2)
                        # lick on next time step if first olfactory stimuli
                        if self.observation_number == 2:
                            self.target_action_number = 1
                    
            elif self.time_index == 1:
                if self.observation_number >= 2:
                    self._reward_and_end_trial(2,action)
                else:
                    # licking to visual stimulus in olfactory block is wasteful
                    self._needless_lick_to_visual_in_olfactory_block(action)
                        
                    # one of olfactory stimuli is shown now
                    self.observation_number = \
                            2 + int(self.rng.uniform()*2)

                    # lick on next time step if first olfactory stimuli
                    if self.observation_number == 2:
                        self.target_action_number = 1

            elif self.time_index == 2:
                self._reward_and_end_trial(2,action)
