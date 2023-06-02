'''
A belief state RL task from the lab of Adil Ghani Khan.

Version 0: 
We're using VisualOlfactoryAttentionSwitchNoBlankEnv which inherits from this but modifies _step()

Block Visual:
3 time steps in a trial:
 step 1: blank cue
 step 2: lick is punished, one of 2 visual stimuli is shown
 step 3: lick leads to reward if visual cue 1, punishment if visual cue 2

Block Olfactory:
3 or 4 time steps in a trial:
 step 1: blank cue
 30%
 step 2: needless lick is mildly punished (internal cost to mouse), one of 2 odor stimuli is given
 step 3: lick leads to reward if odor 1, punishment (in exp, it's a timeout) if odor 2
 70%
 step 2: needless lick is mildly punished (internal cost to mouse), one of 2 visual stimuli is shown
 step 3: needless lick is mildly punished (internal cost to mouse), one of 2 odor stimuli is given
 step 4: lick leads to reward if odor 1, punishment (in exp, it's a timeout) if odor 2

After any block transition, shaping trials are given,
 where always a rewarded visual stimulus is shown in both blocks,
  until the mouse gets 3 correct responses i.e. lick in visual block, no lick in olfactory block.
We transition to the other block after at least 80% correct in past 30 trials,
 and 100% correct in ignoring past 10 irrelevant visual stimuli,
 and there should be at least 30 trials after initial shaping trials after a block transition.
A lick for cue 1 or a no lick for cue 2 in the final reward step counts as correct response.
No lick in all other time steps in these trials is also required.

Aditya Gilra, 31 Aug 2021.
'''

from gym import Env
from gym.spaces import Discrete
#from gym.utils import colorize, seeding
import numpy as np
import sys

class VisualOlfactoryAttentionSwitchEnv(Env):
    blocks = ['visual','olfactory']
    visual_stimuli = ['|||','///'] # rewarded, unrewarded respectively
    olfactory_stimuli = ['odor1', 'odor2'] # rewarded, unrewarded respectively
    actions = ['NoLick','Lick']
    observations = ['blank'] + visual_stimuli + olfactory_stimuli + ['end']

    def __init__(self, reward_size=10, punish_factor=0.5,
                        lick_without_reward_factor=0.2, seed=1):
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
        
        self.end_of_trial_observation_number = len(self.observations)-1

        # various attributes that are set by reset()
        self.block_number = None
        self.observation_number = None
        self.last_action = None
        self.reward = None
        self.corrects_vec = np.zeros(30) # 80% of past 30 trials must be correct
        self.corrects_idx = 0 # this circulates around the corrects_vec to track past 30
        self.consecutive_ignore_irrelevant_visual = None

        self.time_index = None
        self.trial_number = None
        self.done_trial = None

        self.rng = None

        # set / reset some attributes of this class
        self.set_seed(seed)
        self.reset()
        
        self.outfile = sys.stdout

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)
        #self.np_random, seed = seeding.np_random(seed)

    def reset(self):
        self.block_number = 0
        self.observation_number = 0
        self.target_action_number = 0
        self.last_action = None
        self.reward = 0
        self.corrects_vec = np.zeros(30)
        self.corrects_idx = 0
        self.consecutive_ignore_irrelevant_visual = 0

        self.trial_number = 0
        self.trial_number_from_start_of_block = 0
        self.trial_number_in_block = 0
        self.shaping_trials_correct = 0
        self.time_index = 0
        self.done_trial = False

        return self.observation_number

    def correct_wrong(self,val):
        "put correct or wrong i.e. val into the corrects_vec to be able to count 80% correct of past 30"
        self.corrects_vec[self.corrects_idx] = val
        self.corrects_idx = (self.corrects_idx + 1) % 30

    def _reward_and_end_trial(self, target_observation_number, action):
        """target_observation_number = 
            1 for visual block, 3 for olfactory block, if blanks at start of trial
            0 for visual block, 2 for olfactory block, if no blanks at start of trial
        """
        # give reward / punishment
        if self.observation_number == target_observation_number:
            if action == 1: # lick
                # give reward
                self.reward = self.reward_size
                self.correct_wrong(1) # correct
            else: # no lick
                # give punishment, and wrong
                #self.reward = -self.punish_factor*self.reward_size
                self.correct_wrong(0)

        if self.observation_number == target_observation_number+1:
            if action == 0: # no lick
                # no reward, but counted as a correct trial
                self.correct_wrong(1)
            else: # lick
                # give punishment, and wrong
                self.reward = -self.punish_factor*self.reward_size
                self.correct_wrong(0)

        # check responses to shaping trials in visual block
        if self.block_number == 0 and self.shaping_trials_correct < 3:
            # in shaping trials, always rewarded visual stimulus is shown
            if action == 0:
                # no lick is incorrect
                self.shaping_trials_correct = 0
            else:
                # need 3 licks to rewarded visual stimulus at start of a visual block
                self.shaping_trials_correct += 1

        # end trial
        self.done_trial = True        
        # end state is shown here
        self.observation_number = self.end_of_trial_observation_number
        # debug print
        #print(self.block_number,self.trial_number_in_block,
        #        self.consecutive_ignore_irrelevant_visual,
        #        self.corrects_vec,self.shaping_trials_correct)

    def _needless_lick(self, action):
        # licking to end or blank stimulus is wasteful,
        #  so mild punishment, equivalent to waste of energy by mouse
        if action == 1:
            self.reward = -self.lick_without_reward_factor*self.reward_size

    def _needless_lick_to_visual_in_olfactory_block(self, action):
        if action == 1:
            # licking to visual stimulus in olfactory block is wasteful
            #  so mild punishment, equivalent to waste of energy by mouse
            self.reward = -self.lick_without_reward_factor*self.reward_size
            self.consecutive_ignore_irrelevant_visual = 0
            # once 3 correct shaping trials are done,
            #  don't set shaping_trials_correct back to 0, even for needless lick
            # but if 3 correct shaping trials are not done,
            #  then set shaping_trials_correct back to 0, until they are done
            if self.shaping_trials_correct < 3:
                self.shaping_trials_correct = 0
        else:
            # need 3 ignores to 'rewarded visual stimulus' at start of an olfactory block
            if self.shaping_trials_correct < 3:
                self.shaping_trials_correct += 1
            # need 10 consecutive ignores to both visual stimuli before O2V transition
            self.consecutive_ignore_irrelevant_visual += 1

    def step(self, action):
        assert self.action_space.contains(action)

        self.time_index += 1
        
        # debug print
        #print(self.block_number,self.trial_number_in_block,
        #        self.consecutive_ignore_irrelevant_visual,
        #        self.corrects_vec,self.shaping_trials_correct,
        #        self.time_index, self.done_trial)
        
        # if the last trial had finished, start a new trial
        if self.done_trial:
            self.done_trial = False
            self.trial_number += 1
            self.trial_number_from_start_of_block += 1
            if self.shaping_trials_correct >= 3:
                self.trial_number_in_block += 1
            self.time_index = 0

            # within `if self.done_trial:`
            #  as we should only switch block at end of a trial,
            #  else may switch on reaching 10 consecutive ignore visuals,
            #  having reached other criteria earlier!
            # switch block after 80% consecutive correct in past 30 trials (0.8*30=24)
            #  and at least 30 trials in this block after shaping trials
            #  and if olfactory block, at least 10 consecutive ignore visuals
            if (self.block_number == 0 \
                    and np.sum(self.corrects_vec) >= 24 \
                    and self.trial_number_in_block >= 30) \
                or \
                (self.block_number == 1 \
                    and sum(self.corrects_vec) >= 24 \
                    and self.trial_number_in_block >= 30 \
                    and self.consecutive_ignore_irrelevant_visual >= 10):
                self.block_number = 1 - self.block_number
                self.corrects_vec = np.zeros(30)
                self.corrects_idx = 0
                self.consecutive_ignore_irrelevant_visual = 0
                self.trial_number_in_block = 0
                self.shaping_trials_correct = 0
                self.trial_number_from_start_of_block = 0

        # set default variables, unless changed in _step()
        self.reward = 0
        self.target_action_number = 0

        # local method which computes the next state, reward etc.
        self._step(action)

        self.last_action = action

        return self.observation_number, self.reward, self.done_trial, \
                        {'target_act': self.target_action_number,\
                        'block': self.block_number}

    def _step(self, action):
        """
        Local method which computes the next state, reward etc.
        Don't call this directly, call self.step(action).
        """
        # visual block
        if self.block_number == 0:
            if self.time_index == 0:
                # licking to end stimulus is wasteful
                self._needless_lick(action)
                # blank stimulus is shown now
                self.observation_number = 0
                
            elif self.time_index == 1:
                # licking to blank stimulus is wasteful
                self._needless_lick(action)

                # one of visual stimuli is shown now
                if self.shaping_trials_correct < 3:
                    # rewarded visual stimulus is shown until 3 correct in a row
                    self.observation_number = 1
                else:
                    self.observation_number = \
                            1 + int(self.rng.uniform()*2)

                # lick on next time step if first visual stimuli
                if self.observation_number == 1:
                    self.target_action_number = 1
                    
            elif self.time_index == 2:
                self._reward_and_end_trial(1,action)
                
        # olfactory block
        elif self.block_number == 1:
            if self.time_index == 0:
                # licking to end stimulus is wasteful
                self._needless_lick(action)
                # blank stimulus is shown now
                self.observation_number = 0
                
            elif self.time_index == 1:
                # licking to blank stimulus is wasteful
                self._needless_lick(action)

                if self.shaping_trials_correct < 3:
                    # to-ignore rewarded visual stimulus is shown until 3 correct in a row
                    self.observation_number = 1
                else:
                    if self.rng.uniform() < 0.7:
                        # one of visual stimuli is shown now, unrewarded
                        self.observation_number = \
                                1 + int(self.rng.uniform()*2)
                    else:
                        # one of olfactory stimuli is shown now
                        self.observation_number = \
                                3 + int(self.rng.uniform()*2)
                        # lick on next time step if first olfactory stimuli
                        if self.observation_number == 3:
                            self.target_action_number = 1
                    
            elif self.time_index == 2:
                if self.observation_number >= 3:
                    self._reward_and_end_trial(3,action)
                else:
                    # licking to visual stimulus in olfactory block is wasteful
                    self._needless_lick_to_visual_in_olfactory_block(action)
                        
                    # one of olfactory stimuli is shown now
                    self.observation_number = \
                            3 + int(self.rng.uniform()*2)

                    # lick on next time step if first olfactory stimuli
                    if self.observation_number == 3:
                        self.target_action_number = 1

            elif self.time_index == 3:
                self._reward_and_end_trial(3,action)

    def render(self, mode='human'):
        self.outfile.write("block: "+self.blocks[self.block_number]+", ")
        self.outfile.write("trial number: "+str(self.trial_number)+", ")
        self.outfile.write("time step: "+str(self.time_index)+", ")
        self.outfile.write("last action: "+("None" if self.last_action is None else str(self.actions[self.last_action]))+", ")
        self.outfile.write("last reward: "+str(self.reward)+", ")
        self.outfile.write("stimulus: "+self.observations[self.observation_number]+", ")
        self.outfile.write("next target action: "+self.actions[self.target_action_number]+".\n")
        return
        
