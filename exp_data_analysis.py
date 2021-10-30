"""
Simple data analysis of mice behaviour on olfactory/visual task switches

by Aditya Gilra, 24 Oct 2021
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as scipyio
import sys

# reproducible random number generation
np.random.seed(1)

# Obsolete -- see Nick's new data file aligned at transitions below
# Control without ACC silencing: OAP003_B1_20210706_141403.mat
# With ACC silenced: OAP003_B1_20210707_145836.mat
#mouse_behaviour_data = scipyio.loadmat("experiment_data/OAP003_B1_20210706_141403.mat",struct_as_record=True)
# The mat file has a structure called fsm, with some arrays which have a number for each trial.
#print(type(mouse_behaviour_data['fsm'][0,0]))

# Data aligned at transitions provided by Nick
#  exp is with ACC silenced, control is without
#The structure is:
#Condition (control or exp)
#Switching direction (v2o = visual to odour block, o2v = odour to visual block).
# SessionV2O/O2V is the data separated into each session,
#  with the 8 mice as columns and the 3 (or in one case 2) sessions for each mice as the rows.
# MouseV2O/O2V is the data only separated by mice, with the data from multiple sessions concatenated.
#Data type - 3 fields each with columns representing trial (relative to switch),
# and row representing each switch. I [Nick] exported this with 10 trials either side of the switch,
# so the 11th column is trial 0 (i.e. the first trial following the switch). The fields are:
#- stimulus: 1 = Rewarded visual grating, relevant
#     2 = Unrewarded visual grating, relevant
#     3 = Rewarded visual grating, irrelevant
#     4 = Unrewarded visual grating, irrelevant
#     5 = Rewarded odour
#     6 = Unrewarded odour
#- lick : 1 if mouse licked, 0 if not
#- RT: reaction time in seconds. If mouse did not respond will be NaN.

# This .mat file has ans as a Matlab struct with fields control and exp as 1x1 Matlab structs
#  control and exp each contain sessionV2O, mouseV2O, sessionO2V and mouseO2V
#   as 3x8, 1x8, 3x8 and 1x8 Matlab cells respectively.
#   each cell is a 1x1 Matlab struct (or empty) having fields stimulus, lick and RT each ??x31 doubles.
# Read https://docs.scipy.org/doc/scipy/reference/tutorial/io.html#matlab-structs
# on how to access Matlab structs and 
# https://docs.scipy.org/doc/scipy/reference/tutorial/io.html#matlab-cell-arrays
# on how to access Matlab cell arrays

# window of size 31, 10 before and 21 after
#mouse_behaviour_data = scipyio.loadmat(
#                    "experiment_data/exported_behavioural_data_Control_vs_ACC_silencing.mat",
#                                        struct_as_record=True)
#transition_index = 9

# window of size 61, 30 before and 31 after
mouse_behaviour_data = scipyio.loadmat(
                    "experiment_data/exported_behavioural_data_Control_vs_ACC_silencing_-30_+30_trials.mat",
                                        struct_as_record=True)
transition_index = 29

# example access:
#print(mouse_behaviour_data['ans']['control'][0,0]['sessionV2O'][0,0][2,7][0,0]['stimulus'])

reward_size=10
punish_factor=0.5
# the exp data doesn't contain blanks and end of trial cues as in the model task, so not taken into account
#lick_without_reward_factor=0.2

def get_exp_reward_around_o2v_transition():
    behaviour_data = mouse_behaviour_data['ans']['control'][0,0]['mouseO2V'][0,0]
    window = behaviour_data[0,0][0,0]['stimulus'].shape[1]
    number_of_mice = len(mouse_behaviour_data['ans']['control'][0,0]['mouseO2V'][0,0][0])
    mice_average_reward_around_o2v_transtion = np.zeros((number_of_mice,window))
    across_mice_average_reward = np.zeros(window)
    mice_average_action_to_stimulus = np.zeros((number_of_mice,6,window,2)) # 6 stimuli, 2 actions

    for mouse_number in range(number_of_mice):
        # steps around transitions for one mouse
        # both these numpy arrays of doubles have size T x window,
        #  where T are the number of transitions
        olfactory_to_visual_transition_stimuli = behaviour_data[0,mouse_number][0,0]['stimulus']
        olfactory_to_visual_transition_licks = behaviour_data[0,mouse_number][0,0]['lick']
        
        ######### reward around transition
        # positive reward if mouse licks and rewarded stimulus, else negative reward
        # cannot use python or and and operators for element-wise operations,
        # instead using numpy | and & operators
        olfactory_to_visual_transition_positive_rewards = \
                    ( (olfactory_to_visual_transition_stimuli==1) | \
                        (olfactory_to_visual_transition_stimuli==5) ) \
                    & (olfactory_to_visual_transition_licks==1) 
        olfactory_to_visual_transition_negative_rewards = \
                    ( (olfactory_to_visual_transition_stimuli==2) | \
                        (olfactory_to_visual_transition_stimuli==6) ) \
                    & (olfactory_to_visual_transition_licks==1)
        # 10 for lick to rewarded stimulus,
        # 0 for nolick to rewarded or unrewarded stimulus,
        # -5 for lick to unrewarded stimulus
        olfactory_to_visual_transition_rewards = \
            olfactory_to_visual_transition_positive_rewards * reward_size \
                - olfactory_to_visual_transition_negative_rewards * punish_factor*reward_size
        print("shape of rewards array around transition",
                    olfactory_to_visual_transition_rewards.shape)
        average_reward_around_o2v_transition = np.mean(olfactory_to_visual_transition_rewards,axis=0)
        
        mice_average_reward_around_o2v_transtion[mouse_number,:] = average_reward_around_o2v_transition
        across_mice_average_reward += average_reward_around_o2v_transition

        ######### actions given stimuli around transition
        for stimulus_number in range(6):
            # bitwise and takes precedence over equality testing, so need brackets
            # stimuli are saved in experiment as 1 to 6, while stimulus_number goes from 0 to 5
            # since not all time steps will have a particular stimulus,
            #  I encode lick as 1, no lick (with stimulus) as -1, and no stimulus as 0 
            mice_average_action_to_stimulus[mouse_number,stimulus_number,:,0] -= \
                   np.mean((olfactory_to_visual_transition_stimuli==stimulus_number+1) \
                            & (olfactory_to_visual_transition_licks==0) , axis=0 )
            mice_average_action_to_stimulus[mouse_number,stimulus_number,:,1] += \
                   np.mean((olfactory_to_visual_transition_stimuli==stimulus_number+1) \
                            & (olfactory_to_visual_transition_licks==1) , axis=0 )

    across_mice_average_reward /= number_of_mice

    return (number_of_mice, across_mice_average_reward, \
                mice_average_reward_around_o2v_transtion, \
                mice_average_action_to_stimulus)

if __name__ == "__main__":
    number_of_mice, across_mice_average_reward, \
        mice_average_reward_around_o2v_transtion, \
        mice_average_action_to_stimulus = \
            get_exp_reward_around_o2v_transition()

    fig1 = plt.figure()
    for mouse_number in range(number_of_mice):
        plt.plot(mice_average_reward_around_o2v_transtion[mouse_number,:],',-',color=(0,0,0,0.25))
    plt.plot(across_mice_average_reward,',-k')
    plt.plot([transition_index,transition_index],\
                [np.min(mice_average_reward_around_o2v_transtion),\
                    np.max(mice_average_reward_around_o2v_transtion)],',-g')
    plt.xlabel('time steps around olfactory to visual transition')
    plt.ylabel('average reward on time step')
    
    fig2, axes = plt.subplots(2, 3)
    for stimulus_number in range(6):
        row = stimulus_number//3
        col = stimulus_number%3
        for mouse_number in range(number_of_mice):
            axes[row,col].plot(mice_average_action_to_stimulus[mouse_number,stimulus_number,:,0],',-',color=(1,0,0,0.25))
            axes[row,col].plot(mice_average_action_to_stimulus[mouse_number,stimulus_number,:,1],',-',color=(0,0,1,0.25))
        axes[row,col].plot(np.mean(mice_average_action_to_stimulus[:,stimulus_number,:,0],axis=0),',-r')
        axes[row,col].plot(np.mean(mice_average_action_to_stimulus[:,stimulus_number,:,1],axis=0),',-b')
        axes[row,col].plot([transition_index,transition_index],\
                [np.min(mice_average_action_to_stimulus),\
                    np.max(mice_average_action_to_stimulus)],',-g')
        axes[row,col].set_xlabel('steps at O2V transition')
        axes[row,col].set_ylabel('average action on stimulus '+str(stimulus_number+1))
    fig2.subplots_adjust(wspace=0.5,hspace=0.5)

    plt.show()
