"""
Simple data analysis of mice behaviour on olfactory/visual task switches

by Aditya Gilra, 24 Oct 2021
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as scipyio
import sys

# to have reproducible random number generation
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
#mouse_behaviour_data = scipyio.loadmat(
#                    "experiment_data/exported_behavioural_data_Control_vs_ACC_silencing_-30_+30_trials.mat",
#                                        struct_as_record=True)
# Nick sent new data confirming a few stimuli were shown not as intended as per task design
# so he has removed them
#mouse_behaviour_data = scipyio.loadmat(
#                    "experiment_data/exported_behavioural_data_Control_vs_ACC_silencing_-30_+30_trials_errors_removed.mat",
#                                        struct_as_record=True)
# Nick said that 50% olfactory performance after v2o switch was an "error with separating the first few trials following a switch"
# so he has re-exported the data correctly now:
# Using the new neural recordings + behaviour data below to fit 'ACC on' (control) behaviour,
#  but still need this to fit and comare control (ACC on) vs exp (ACC off), as new dataset below doesn't have 'exp' field.
mouse_behaviour_data = scipyio.loadmat(
                    "experiment_data/exported_behavioural_data_Control_vs_ACC_silencing_-30_+30_trials_secondbug_corrected.mat",
                                        struct_as_record=True)

transition_index = 29

# example access:
#print(mouse_behaviour_data['ans']['control'][0,0]['sessionV2O'][0,0][2,7][0,0]['stimulus'])

#================ neural data and related behaviour ==================

mouse_neural_data = scipyio.loadmat(
                    "experiment_data/exported_imaging_data_class1_neurons_v2.mat",
                                    struct_as_record=True)
mouse_neural_mismatch = mouse_neural_data['dF']['expResponses'][0,0][0]
mouse_correct_switch = mouse_neural_data['dF']['correctSwitch'][0,0][0]
# From Nick Cole: 
#"""exported imaging data for class 1 (positive mismatch) and class 4 (negative mismatch) neurons.
# These correspond to the behavioural data I exported from these imaging sessions:
#  each has 13 cells (one per session), with each row corresponding to each switch from odour to visual, and each column corresponding to each neuron of that class
#  (so a 5 x 4 array would be 5 switches and 4 neurons). There are two things I should point out about these that might not make sense:
#  Not every session has mismatch neurons - I think of the 13 sessions 2 have no class 1 neurons and 2 have no class 4 neurons, these are the empty cells. ...."""
# Imaging data: rows = switches, columns = mismatch neurons (so a 4 x 6 array is the mean mismatch response amplitude for the same 4 switches over the 6 neurons in that session that are class 1 or class 4 mismatch neurons, depending on which dataset it is)
# Imaging data has two fields: 'expResponses' is the mismatch response amplitudes for each mismatch neuron, in the format mentioned earlier. 'correctSwitch' is what you asked for, it's a logical array with one value for each switch in the session. If 1 it means that the trials after that mismatch trial was correct

# Behavioural data: rows = switches, columns = timepoints (so a 4 x 61 array is 4 switches with 61 timepoints, with timepoint 31 being the first stimulus after the switch)
# This is used to fit 'control' behaviour, but doesn't have exp (ACC off) behavioiur.
mouse_behaviour_for_neural_data = scipyio.loadmat(
                    "experiment_data/exported_behavioural_data_from_imaging_sessions.mat",
                                    struct_as_record=True)


#==============================================================

reward_size=10
punish_factor=0.5
# the exp data doesn't contain blanks and end of trial cues as in the model task, so not taken into account
#lick_without_reward_factor=0.2

def get_exp_reward_around_transition(trans='O2V',ACC='control',
                                        mice_list=None,sessions_list=None,):
    # ACC can be 'control' (without ACC silenced) or 'exp' (with ACC silenced)
    behaviour_data = mouse_behaviour_data['expData'][ACC][0,0]['mouse'+trans][0,0]
    window = behaviour_data[0,0][0,0]['stimulus'].shape[1]
    number_of_mice = len(mouse_behaviour_data['expData'][ACC][0,0]['mouse'+trans][0,0][0])
    if mice_list is None: mice_list = range(number_of_mice)
    mice_average_reward_around_transtion = np.zeros((number_of_mice,window))
    across_mice_average_reward = np.zeros(window)
    mice_actionscount_to_stimulus = np.zeros((number_of_mice,6,window,2)) # 6 stimuli, 2 actions
    mice_actionscount_to_stimulus_trials = np.zeros((number_of_mice,6,window,2)) # 6 stimuli, 2 actions
    sessions_actionscount_to_stimulus = [[[],[]],[[],[]],[[],[]],[[],[]],[[],[]],[[],[]]] # 6 stimuli, 2 actions
    transitions_actionscount_to_stimulus = [[[],[]],[[],[]],[[],[]],[[],[]],[[],[]],[[],[]]] # 6 stimuli, 2 actions

    for mouse_number in mice_list:
        # steps around transitions for one mouse
        # both these numpy arrays of doubles have size T x window,
        #  where T are the number of transitions
        transition_stimuli = behaviour_data[0,mouse_number][0,0]['stimulus']
        # earlier, 'lick' field didn't contain licks but correct responses!
        # now, it does, and there's a 'correct' field as well,
        # but further below, I continue to use corrects and convert corrects to licks
        #transition_corrects = behaviour_data[0,mouse_number][0,0]['lick']
        transition_corrects = behaviour_data[0,mouse_number][0,0]['correct']
        # would have been easier to use but given later, so not used now
        transition_licks = behaviour_data[0,mouse_number][0,0]['lick']
        
        ######### reward around transition
        # positive reward if mouse licks and rewarded stimulus, else negative reward
        # cannot use python or and and operators for element-wise operations,
        # instead using numpy | and & operators
        transition_positive_rewards = \
                    ( (transition_stimuli==1) | \
                        (transition_stimuli==5) ) \
                    & (transition_corrects==1) 
        transition_negative_rewards = \
                    ( (transition_stimuli==2) | \
                        (transition_stimuli==6) ) \
                    & (transition_corrects==0) # licked, 0 => incorrect
        # 10 for lick to rewarded stimulus,
        # 0 for nolick to rewarded or unrewarded stimulus,
        # -5 for lick to unrewarded stimulus
        transition_rewards = \
            transition_positive_rewards * reward_size \
                - transition_negative_rewards * punish_factor*reward_size
        #print("shape of rewards array around transition",
        #            transition_rewards.shape)
        average_reward_around_transition = np.mean(transition_rewards,axis=0)
        
        mice_average_reward_around_transtion[mouse_number,:] = average_reward_around_transition
        across_mice_average_reward += average_reward_around_transition

        ######### actions given stimuli around transition
        for stimulus_index in range(6):
            # `bitwise and` takes precedence over equality testing, so need brackets
            # stimuli are saved in experiment as 1 to 6, while stimulus_index goes from 0 to 5
            #  transition_corrects encodes 0 as incorrect, 1 as correct response
            #  I transform to counts of nolicks, and counts of licks
            if stimulus_index in (0,4): lick_to_correct = (0,1)
            else: lick_to_correct = (1,0)

            if sessions_list is None: sessions_list_thismouse = range(transition_stimuli.shape[0])
            else: sessions_list_thismouse = sessions_list

            for session_index in sessions_list_thismouse:
                ##### steps around transition
                ##### not all time steps will have a particular stimulus!
                stim_nolick = (transition_stimuli[session_index,:]==stimulus_index+1) \
                                & (transition_corrects[session_index,:]==lick_to_correct[0])
                stim_lick = (transition_stimuli[session_index,:]==stimulus_index+1) \
                                & (transition_corrects[session_index,:]==lick_to_correct[1])
                mice_actionscount_to_stimulus[mouse_number,stimulus_index,:,0] += stim_nolick
                mice_actionscount_to_stimulus[mouse_number,stimulus_index,:,1] += stim_lick
                # for mean and SD of action prob across sessions (obsolete):
                sessions_actionscount_to_stimulus[stimulus_index][0].append(stim_nolick)
                sessions_actionscount_to_stimulus[stimulus_index][1].append(stim_lick)
                # transitions_actionscount_to_stimulus will have dimensions:
                #  stimuli x actions x transitions x window_size
                transitions_actionscount_to_stimulus[stimulus_index][0].append(stim_nolick)
                transitions_actionscount_to_stimulus[stimulus_index][1].append(stim_lick)

                ##### trials around transition -- since an olfactory trial may have 1 or 2 'steps' (cf. above)
                ##### convert from steps to trials for each session, while maintaining transition index at the center
                # we pick time steps involving visual stimuli only or olfactory stimuli only
                #  if stimulus_index denotes a visual or olfactory stimulus respectively
                if stimulus_index in (0,1): relevant_stimuli = (1,2)
                elif stimulus_index in (2,3): relevant_stimuli = (3,4)
                elif stimulus_index in (4,5): relevant_stimuli = (5,6)
                trials_indices_before_transition = \
                        np.where( (transition_stimuli\
                                    [session_index,:transition_index]==relevant_stimuli[0]) \
                                  | (transition_stimuli\
                                    [session_index,:transition_index]==relevant_stimuli[1]) )[0]
                trials_indices_after_transition = \
                        np.where( (transition_stimuli\
                                    [session_index,transition_index:]==relevant_stimuli[0]) \
                                  | (transition_stimuli\
                                    [session_index,transition_index:]==relevant_stimuli[1]) )[0]
                for lick in (0,1):
                    # align before transition
                    if len(trials_indices_before_transition)>0:
                        licks_trials = (transition_corrects\
                                        [session_index,trials_indices_before_transition]==lick_to_correct[lick])
                        mice_actionscount_to_stimulus_trials[mouse_number,stimulus_index,
                                transition_index-len(licks_trials):transition_index,lick] += licks_trials
                    # align after transition
                    if len(trials_indices_after_transition)>0:
                        licks_trials = (transition_corrects\
                                        [session_index,trials_indices_after_transition]==lick_to_correct[lick])
                        mice_actionscount_to_stimulus_trials[mouse_number,stimulus_index,
                                transition_index:len(licks_trials)+transition_index,lick] += licks_trials

    across_mice_average_reward /= number_of_mice

    # normalize over the actions (last axis i.e. -1) to get probabilities
    # do not add a small amount to denominator to avoid divide by zero
    #  since I am allowing nan-s so that irrelvant time steps are not plotted,
    #  and for fitting nan-s as well
    mice_probability_action_given_stimulus = mice_actionscount_to_stimulus \
                / np.sum(mice_actionscount_to_stimulus,axis=-1)[:,:,:,np.newaxis] #\
                        #+ np.finfo(np.double).eps )

    ## obsolete mean (was only keeping mouse resolution, want session resolution for SD) -- replaced below
    #mean_probability_action_given_stimulus = np.sum(mice_actionscount_to_stimulus,axis=0) \
    #            / np.sum(mice_actionscount_to_stimulus,axis=(0,-1))[:,:,np.newaxis] #\
    #                    #+ np.finfo(np.double).eps )

    # mean across all mice and sessions
    # should not just take mean across sessions, as each session has variable number of trials
    # rather divide by total number of actions taken for each stimulus.
    #  i.e. normalize lick & nolick counts to give probability
    # sessions_actionscount_to_stimulus has dimensions stimuli x actions x sessions x window
    mean_probability_action_given_stimulus = \
        np.sum(sessions_actionscount_to_stimulus,axis=2) \
                / np.sum(sessions_actionscount_to_stimulus,axis=(1,2))[:,np.newaxis,:]
    # SD in p(lick|stimulus) of a session compared to mean across all mice and sessions
    # p(lick|session) separately for each session, normalize lick & nolick counts to give probability
    # most of these are nan-s as in a session, as there are only 2-4 transitions,
    #  and then each time step in the window doesn't have each of the 6 stimuli, so nan for that stimulus
    sessions_probability_action_given_stimulus = \
        sessions_actionscount_to_stimulus \
                    / np.sum(sessions_actionscount_to_stimulus,axis=1)[:,np.newaxis,:,:]
    # SD of above, but even one nan above makes the SD nan for that time step, hence mostly nan-s.
    deviation = sessions_probability_action_given_stimulus \
                    - mean_probability_action_given_stimulus[:,:,np.newaxis,:]
    SD_probability_action_given_stimulus = np.std(deviation,axis=2)

    # swap window and action axes, to keep returned variable consistent with before
    mean_probability_action_given_stimulus = \
        np.swapaxes(mean_probability_action_given_stimulus,1,2)
    # swap window and action axes, to keep returned variable consistent with before
    SD_probability_action_given_stimulus = \
        np.swapaxes(SD_probability_action_given_stimulus,1,2)
    # not worth returning the SD, as mostly nan-s!

    return (number_of_mice, across_mice_average_reward,
                mice_average_reward_around_transtion,
                mice_actionscount_to_stimulus,
                mice_actionscount_to_stimulus_trials,
                mice_probability_action_given_stimulus,
                mean_probability_action_given_stimulus,
                np.array(transitions_actionscount_to_stimulus))

def plot_prob_actions_given_stimuli(units='steps', trans='O2V'):
    # no need to pass in:
    #  mice_probability_action_given_stimulus,
    #  mean_probability_action_given_stimulus
    #  as these are available in the global workspace
    #  and we don't modify them here, only plot them

    window = mean_probability_action_given_stimulus.shape[1]

    # debug print
    #for stimulus_index in range(6):
    #    print(stimulus_index+1,\
    #            mean_probability_action_given_stimulus\
    #                [stimulus_index,window//2-8:window//2+8,1])
    #    print(stimulus_index+1,
    #            np.sum(mice_actionscount_to_stimulus,axis=0)\
    #                [stimulus_index,window//2-8:window//2+8,1],\
    #            np.sum(mice_actionscount_to_stimulus,axis=(0,-1))\
    #                [stimulus_index,window//2-8:window//2+8])

    xvec = range(-window//2+1,window//2+1)
    fig, axes = plt.subplots(2, 3)
    figall, axall = plt.subplots(1,1)
    #figall = plt.figure()
    #axall = figall.add_axes([0.1, 0.1, 0.9, 0.9])
    axall.plot([0,0],[0,1],',k',linestyle='--')
    #colors = ['r','g','y','c','b','m']
    colors = ['b','r','b','r','g','y']
    labels = ['+v','-v','/+v','/-v','+o','-o']
    for stimulus_index in range(6):
        row = stimulus_index//3
        col = stimulus_index%3
        axes[row,col].plot([0,0],[0,1],',-g')
        for mouse_number in range(number_of_mice):
            axes[row,col].plot(xvec, mice_probability_action_given_stimulus\
                                    [mouse_number,stimulus_index,:,0],'.-',color=(1,0,0,0.25))
            axes[row,col].plot(xvec, mice_probability_action_given_stimulus\
                                    [mouse_number,stimulus_index,:,1],'.-',color=(0,0,1,0.25))
        axes[row,col].plot(xvec, mean_probability_action_given_stimulus\
                                    [stimulus_index,:,0],'.-r',label='nolick')
        axes[row,col].plot(xvec, mean_probability_action_given_stimulus\
                                    [stimulus_index,:,1],'.-b',label='lick')
        axes[row,col].set_xlabel(units+' around '+trans+' transition')
        axes[row,col].set_ylabel('P(action|stimulus='+str(stimulus_index+1)+')')
        axes[row,col].set_xlim([-window//2+1,window//2+1])
        
        # lick probability given stimuli all in one axes
        axall.plot(xvec,mean_probability_action_given_stimulus\
                            [stimulus_index,:,1], marker='.',
                            color=colors[stimulus_index],
                            label=labels[stimulus_index])
        # SD has mostly nan-s, so no point plotting
        #axall.fill_between(xvec, 
        #        mean_probability_action_given_stimulus[stimulus_index,:,1]-\
        #                SD_probability_action_given_stimulus[stimulus_index,:,1],\
        #        mean_probability_action_given_stimulus[stimulus_index,:,1]+\
        #                SD_probability_action_given_stimulus[stimulus_index,:,1],\
        #        color=colors[stimulus_index],alpha=0.2)
        axall.set_xlabel(units+' around '+trans+' transition')
        axall.set_ylabel('P(lick|stimulus)')
        axall.set_xlim([-window//2+1,window//2+1])

    axes[row,col].legend()
    fig.subplots_adjust(wspace=0.5,hspace=0.5)
    axall.legend()
    figall.tight_layout()


def analyse_neural_mismatch():
    ACC = 'control'
    trans = 'O2V'
    behaviour_data = mouse_behaviour_for_neural_data['expData'][ACC][0,0]['mouse'+trans][0,0]
    window = behaviour_data[0,0][0,0]['stimulus'].shape[1]
    #number_of_mice = len(mouse_behaviour_for_neural_data['expData'][ACC][0,0]['mouse'+trans][0,0][0])
    # number of mice (or is it number of sessions?) is 13 in the behaviour data,
    #  and number of sessions is also 13 in the neural data,
    # but number of transitions is much less in the neural data
    
    number_of_sessions = len(mouse_neural_mismatch)
    print('num sessions =',number_of_sessions)

    # By correct and incorrect/wrong transitions, I mean perfect (one-shot) and imperfect transitions.
    print('mismatch vs in/correct O2V transitions')
    mismatches = [[],[]]
    mismatches_flat = [[],[]]
    mismatch_difference = [[],[]]
    valid_sessions = []

    for session_idx in range(number_of_sessions):
        print('session num: ',session_idx)
        session_data = mouse_neural_mismatch[session_idx]

        # the behavioural data has a lot more transitions compared to the number in the neural data
        #transition_corrects = behaviour_data[0,session_idx][0,0]['correct']
        #print(transition_corrects)
        # transitions from neural data (lot less in number than behavioural data?!)
        transition_corrects = mouse_correct_switch[session_idx]

        if len(session_data[0])>0: # some sessions have no mismatch neurons
            print('num transitions =',len(session_data),', num mismatch neurons =', len(session_data[0]))
            correct_transition_idxs = (mouse_correct_switch[session_idx][0]==1)
            print('correct transition idxs =',correct_transition_idxs)
            # inter-neuron variance in response amplitude (noise here) is larger than
            #  inter-neuron difference for imperfect vs perfect transition (signal here),
            #  so signal gets washed out.
            # hence, don't take mean across neurons right at the start
            mean_across_neurons__mismatch_for_wrong_transitions = np.mean(session_data[~correct_transition_idxs,:],axis=1)
            mean_across_neurons__mismatch_for_correct_transitions = np.mean(session_data[correct_transition_idxs,:],axis=1)
            # take mean across 2 types of transition and subtract theses 2 for each neuron individually,
            #  then take mean across neurons of this difference.
            #  this gives the difference in mismatch signal between 2 types of transitions
            mean_across_transitions__mismatch_for_wrong_transitions = np.mean(session_data[~correct_transition_idxs,:],axis=0)
            mean_across_transitions__mismatch_for_correct_transitions = np.mean(session_data[correct_transition_idxs,:],axis=0)
            difference_across_transitions = \
                mean_across_transitions__mismatch_for_wrong_transitions \
                    - mean_across_transitions__mismatch_for_correct_transitions
            # difference gives nan if all transitions are perfect or imperfect, thus discard
            if not np.isnan(difference_across_transitions[0]):
                print('difference in mismatch for (imperfect - perfect) transitions =',
                             difference_across_transitions)
                mean_mismatch_for_wrong_transitions = [0]
                mismatches[0].append(mean_mismatch_for_wrong_transitions)
                mismatches_flat[0].extend(mean_mismatch_for_wrong_transitions)
                mean_mismatch_for_correct_transitions = [0]
                mismatches[1].append(mean_mismatch_for_correct_transitions)
                mismatches_flat[1].extend(mean_mismatch_for_correct_transitions)
                mismatch_difference[0].append(np.mean(difference_across_transitions))
                mismatch_difference[1].append(np.std(difference_across_transitions))
                valid_sessions.append(session_idx)
        print()

    return mismatches, mismatches_flat, mismatch_difference, valid_sessions

if __name__ == "__main__":
    # choose control for ACC not inhibited, exp for ACC inhibited
    ACC = 'control'
    #ACC = 'exp'
    
    if ACC == 'control':
        # override the data to newest one, for fitting behaviour in 'control' (ACC on) condition
        #  'exp' (ACC off) behaviour has not been recorded in newest version,
        #   so use older data i.e. do not override it..
        mouse_behaviour_data = mouse_behaviour_for_neural_data
    
    number_of_mice, across_mice_average_reward, \
        mice_average_reward_around_transtion, \
        mice_actionscount_to_stimulus, \
        mice_actionscount_to_stimulus_trials, \
        mice_probability_action_given_stimulus, \
        mean_probability_action_given_stimulus, \
        transitions_actionscount_to_stimulus = \
            get_exp_reward_around_transition(trans='O2V',ACC=ACC)

    fig1 = plt.figure()
    for mouse_number in range(number_of_mice):
        plt.plot(mice_average_reward_around_transtion[mouse_number,:],',-',color=(0,0,0,0.25))
    plt.plot(across_mice_average_reward,',-k')
    plt.plot([transition_index,transition_index],\
                [np.min(mice_average_reward_around_transtion),\
                    np.max(mice_average_reward_around_transtion)],',-b')
    plt.xlabel('time steps around olfactory to visual transition')
    plt.ylabel('average reward on time step')
    
    plot_prob_actions_given_stimuli(trans='O2V')
    
    #plot_prob_actions_given_stimuli(mice_actionscount_to_stimulus_trials,'trials')

    number_of_mice, across_mice_average_reward, \
        mice_average_reward_around_transtion, \
        mice_actionscount_to_stimulus, \
        mice_actionscount_to_stimulus_trials, \
        mice_probability_action_given_stimulus, \
        mean_probability_action_given_stimulus, \
        transitions_actionscount_to_stimulus = \
            get_exp_reward_around_transition(trans='V2O',ACC=ACC)

    plot_prob_actions_given_stimuli(trans='V2O')
    
    mismatches_o2v, mismatches_flat_o2v, mismatch_difference_o2v, sessions_o2v = analyse_neural_mismatch()
    """
    fig, ax = plt.subplots(1,1)
    ax.bar( ['wrong switch','correct switch'],
                [np.mean(mismatches_flat_o2v[0]),np.mean(mismatches_flat_o2v[1])],
                yerr=[np.std(mismatches_flat_o2v[0]),np.std(mismatches_flat_o2v[1])] )
    ax.set_ylabel('mismatch error O2V')

    num_sessions = len(mismatches_o2v[0])
    fig, axes = plt.subplots(2,num_sessions//2+1,figsize=(15,6))
    for session_idx in range(num_sessions):
        row = session_idx % 2
        col = session_idx // 2
        axes[row,col].bar( ['wrong switch','correct switch'],
                [np.mean(mismatches_o2v[0][session_idx]),np.mean(mismatches_o2v[1][session_idx])],
                yerr=[np.std(mismatches_o2v[0][session_idx]),np.std(mismatches_o2v[1][session_idx])] )
        axes[row,col].set_title('session '+str(sessions_o2v[session_idx]))
    #fig.suptitle('mismatch error O2V vs (in)correct switch')
    fig.tight_layout()
    """

    fig, ax = plt.subplots(1,1,figsize=(10,4))
    ax.bar( sessions_o2v, mismatch_difference_o2v[0], yerr=mismatch_difference_o2v[1] )
    ax.set_xlabel('session id')
    ax.set_ylabel('mismatch difference (imperfect-perfect transition) O2V')

    plt.show()
