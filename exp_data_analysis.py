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

# This .mat file has 'ans' as a Matlab struct with fields 'control' and 'exp' as 1x1 Matlab structs
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
    mice_actionscount_to_stimulus = np.zeros((number_of_mice,6,window,2)) # 6 stimuli, 2 actions
    mice_actionscount_to_stimulus_trials = np.zeros((number_of_mice,6,window,2)) # 6 stimuli, 2 actions

    for mouse_number in range(number_of_mice):
        # steps around transitions for one mouse
        # both these numpy arrays of doubles have size T x window,
        #  where T are the number of transitions
        olfactory_to_visual_transition_stimuli = behaviour_data[0,mouse_number][0,0]['stimulus']
        # looks like 'lick' field doesn't contain licks but correct responses!
        olfactory_to_visual_transition_corrects = behaviour_data[0,mouse_number][0,0]['lick']
        
        ######### reward around transition
        # positive reward if mouse licks and rewarded stimulus, else negative reward
        # cannot use python or and and operators for element-wise operations,
        # instead using numpy | and & operators
        olfactory_to_visual_transition_positive_rewards = \
                    ( (olfactory_to_visual_transition_stimuli==1) | \
                        (olfactory_to_visual_transition_stimuli==5) ) \
                    & (olfactory_to_visual_transition_corrects==1) 
        olfactory_to_visual_transition_negative_rewards = \
                    ( (olfactory_to_visual_transition_stimuli==2) | \
                        (olfactory_to_visual_transition_stimuli==6) ) \
                    & (olfactory_to_visual_transition_corrects==0) # licked, 0 => incorrect
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
        for stimulus_index in range(6):
            # bitwise and takes precedence over equality testing, so need brackets
            # stimuli are saved in experiment as 1 to 6, while stimulus_index goes from 0 to 5
            #  olfactory_to_visual_transition_corrects encodes 0 as incorrect, 1 as correct response
            #  I transform to counts of nolicks, and counts of licks
            if stimulus_index in (0,4): lick_to_correct = (0,1)
            else: lick_to_correct = (1,0)

            ##### steps around transition
            ##### not all time steps will have a particular stimulus!
            mice_actionscount_to_stimulus[mouse_number,stimulus_index,:,0] += \
                   np.sum((olfactory_to_visual_transition_stimuli==stimulus_index+1) \
                            & (olfactory_to_visual_transition_corrects==lick_to_correct[0]),
                        axis=0 )
            mice_actionscount_to_stimulus[mouse_number,stimulus_index,:,1] += \
                   np.sum((olfactory_to_visual_transition_stimuli==stimulus_index+1) \
                            & (olfactory_to_visual_transition_corrects==lick_to_correct[1]),
                        axis=0 )

            ##### trials around transition
            ##### convert from steps to trials for each session, while maintaining transition index at the center
            for session_index in range(olfactory_to_visual_transition_stimuli.shape[0]):
                # we show time steps involving visual stimuli only or olfactory stimuli only
                #  if stimulus_index denotes a visual or olfactory stimulus respectively
                if stimulus_index in (0,1): relevant_stimuli = (1,2)
                elif stimulus_index in (2,3): relevant_stimuli = (3,4)
                elif stimulus_index in (4,5): relevant_stimuli = (5,6)
                trials_indices_before_transition = \
                        np.where( (olfactory_to_visual_transition_stimuli\
                                    [session_index,:transition_index]==relevant_stimuli[0]) \
                                  | (olfactory_to_visual_transition_stimuli\
                                    [session_index,:transition_index]==relevant_stimuli[1]) )[0]
                trials_indices_after_transition = \
                        np.where( (olfactory_to_visual_transition_stimuli\
                                    [session_index,transition_index:]==relevant_stimuli[0]) \
                                  | (olfactory_to_visual_transition_stimuli\
                                    [session_index,transition_index:]==relevant_stimuli[1]) )[0]
                for lick in (0,1):
                    # align before transition
                    if len(trials_indices_before_transition)>0:
                        licks_trials = (olfactory_to_visual_transition_corrects\
                                        [session_index,trials_indices_before_transition]==lick_to_correct[lick])
                        mice_actionscount_to_stimulus_trials[mouse_number,stimulus_index,
                                transition_index-len(licks_trials):transition_index,lick] += licks_trials
                    # align after transition
                    if len(trials_indices_after_transition)>0:
                        licks_trials = (olfactory_to_visual_transition_corrects\
                                        [session_index,trials_indices_after_transition]==lick_to_correct[lick])
                        mice_actionscount_to_stimulus_trials[mouse_number,stimulus_index,
                                transition_index:len(licks_trials)+transition_index,lick] += licks_trials

    across_mice_average_reward /= number_of_mice

    return (number_of_mice, across_mice_average_reward, \
                mice_average_reward_around_o2v_transtion, \
                mice_actionscount_to_stimulus, mice_actionscount_to_stimulus_trials)

def plot_prob_actions_given_stimuli(mice_actionscount_to_stimulus, units='steps'):
    # normalize over the actions (last axis i.e. -1) to get probabilities
    # do not add a small amount to denominator to avoid divide by zero!
    # allowing nan so that irrelvant time steps are not plotted
    mice_probability_action_given_stimulus = mice_actionscount_to_stimulus \
                / np.sum(mice_actionscount_to_stimulus,axis=-1)[:,:,:,np.newaxis] #\
                        #+ np.finfo(np.double).eps )
    mean_probability_action_given_stimulus = np.sum(mice_actionscount_to_stimulus,axis=0) \
                / np.sum(mice_actionscount_to_stimulus,axis=(0,-1))[:,:,np.newaxis] #\
                        #+ np.finfo(np.double).eps )    

    window = mice_actionscount_to_stimulus.shape[2]
    xvec = range(-window//2+1,window//2+1)
    fig, axes = plt.subplots(2, 3)
    for stimulus_number in range(6):
        row = stimulus_number//3
        col = stimulus_number%3
        for mouse_number in range(number_of_mice):
            axes[row,col].plot(xvec, mice_probability_action_given_stimulus\
                                    [mouse_number,stimulus_number,:,0],',-',color=(1,0,0,0.25))
            axes[row,col].plot(xvec, mice_probability_action_given_stimulus\
                                    [mouse_number,stimulus_number,:,1],',-',color=(0,0,1,0.25))
        axes[row,col].plot(xvec, mean_probability_action_given_stimulus\
                                    [stimulus_number,:,0],',-r',label='nolick')
        axes[row,col].plot(xvec, mean_probability_action_given_stimulus\
                                    [stimulus_number,:,1],',-b',label='lick')
        axes[row,col].plot([0,0],[0,1],',-g')
        axes[row,col].set_xlabel(units+' around O2V transition')
        axes[row,col].set_ylabel('P(action|stimulus='+str(stimulus_number+1)+')')
        axes[row,col].set_xlim([-window//2+1,window//2+1])
    axes[row,col].legend()
    fig.subplots_adjust(wspace=0.5,hspace=0.5)

if __name__ == "__main__":
    number_of_mice, across_mice_average_reward, \
        mice_average_reward_around_o2v_transtion, \
        mice_actionscount_to_stimulus, mice_actionscount_to_stimulus_trials = \
            get_exp_reward_around_o2v_transition()

    fig1 = plt.figure()
    for mouse_number in range(number_of_mice):
        plt.plot(mice_average_reward_around_o2v_transtion[mouse_number,:],',-',color=(0,0,0,0.25))
    plt.plot(across_mice_average_reward,',-k')
    plt.plot([transition_index,transition_index],\
                [np.min(mice_average_reward_around_o2v_transtion),\
                    np.max(mice_average_reward_around_o2v_transtion)],',-b')
    plt.xlabel('time steps around olfactory to visual transition')
    plt.ylabel('average reward on time step')
    
    plot_prob_actions_given_stimuli(mice_actionscount_to_stimulus)
    
    plot_prob_actions_given_stimuli(mice_actionscount_to_stimulus_trials,'trials')
    
    plt.show()
