"""
Context-belief- and/or history-based agent

by Aditya Gilra, Sep-Oct-Nov 2021
"""

import gym
import numpy as np
import gym_tasks
import matplotlib as mpl
import matplotlib.pyplot as plt

from exp_data_analysis import get_exp_reward_around_transition
from BeliefHistoryTabularRL import BeliefHistoryTabularRL

# reproducible random number generation
seed = 1
np.random.seed(seed)

# number of steps on each side of the transition to consider
half_window = 30

def process_transitions(exp_step, block_vector_exp_compare,
                        reward_vector_exp_compare, 
                        stimulus_vector_exp_compare,
                        action_vector_exp_compare,
                        O2V=True):
    # exp_step, at end of time loop above, equals end index saved in _exp_compare vectors
    # clip vector at exp_step, to avoid detecting a last spurious transtion in block_vector_exp_compare

    # processing after agent simulation has ended
    # calculate mean reward and mean action given stimulus, around transitions
    transitions = \
        np.where(np.diff(block_vector_exp_compare[:exp_step])==(-1 if O2V else 1))[0] + 1
    # note that block number changes on the first time step of a new trial,
    # debug print
    #print(('O2V' if O2V else 'V2O')+" transition at steps ",transitions)
    
    average_reward_around_transition = np.zeros(half_window*2+1)
    actionscount_to_stimulus = np.zeros((6,half_window*2+1,2)) # 6 stimuli, 2 actions
    num_transitions_averaged = 0
    for transition in transitions:
        # take edge effects into account when taking a window around transition
        # i.e. don't go beyond the start and end of saved data if transition is close to an edge
        window_min = max((0,transition-half_window))
        # exp_step, at end of time loop above, equals end index saved in _exp_compare vectors
        window_max = min((transition+half_window+1,exp_step))
        window_start = window_min-(transition-half_window)
        window_end = window_max-(transition-half_window)
        average_reward_around_transition[window_start:window_end] \
                += reward_vector_exp_compare[window_min:window_max]

        # debug print
        #print(('O2V' if O2V else 'V2O'), transition,
        #        stimulus_vector_exp_compare[transition-5:transition+5],
        #        action_vector_exp_compare[transition-5:transition+5])

        ######### actions given stimuli around transition
        for stimulus_number in range(1,7):
            # bitwise and takes precedence over equality testing, so need brackets
            # stimuli are saved in experiment as 1 to 6
            # since not all time steps will have a particular stimulus,
            #  I encode lick as 1, no lick (with stimulus) as -1, and no stimulus as 0 
            actionscount_to_stimulus[stimulus_number-1,window_start:window_end,0] += \
                   (stimulus_vector_exp_compare[window_min:window_max]==stimulus_number) \
                            & (action_vector_exp_compare[window_min:window_max]==0)
            actionscount_to_stimulus[stimulus_number-1,window_start:window_end,1] += \
                   (stimulus_vector_exp_compare[window_min:window_max]==stimulus_number) \
                            & (action_vector_exp_compare[window_min:window_max]==1)

            # debug print
            #print(stimulus_number,
            #        actionscount_to_stimulus[stimulus_number-1,half_window-5:half_window+5,0],
            #        actionscount_to_stimulus[stimulus_number-1,half_window-5:half_window+5,1])

        num_transitions_averaged += 1
    average_reward_around_transition /= num_transitions_averaged

    # normalize over the actions (last axis i.e. -1) to get probabilities
    # do not add a small amount to denominator to avoid divide by zero!
    # allowing nan so that irrelvant time steps are not plotted
    probability_action_given_stimulus = actionscount_to_stimulus \
                / np.sum(actionscount_to_stimulus,axis=-1)[:,:,np.newaxis] #\
                        #+ np.finfo(np.double).eps )

    return average_reward_around_transition, \
                actionscount_to_stimulus, \
                probability_action_given_stimulus

def plot_prob_actions_given_stimuli(probability_action_given_stimulus,
                                        exp_mean_probability_action_given_stimulus,
                                        detailed_plots, agent_type='',
                                        units='steps', trans='O2V'):    
    # debug print
    #for stimulus_index in range(6):
    #    print(stimulus_index+1,\
    #            probability_action_given_stimulus\
    #                [stimulus_index,half_window-5:half_window+5,1])
    #    print(stimulus_index+1,
    #            actionscount_to_stimulus\
    #                [stimulus_index,half_window-5:half_window+5,1],\
    #            np.sum(actionscount_to_stimulus\
    #                [stimulus_index,half_window-5:half_window+5,:],axis=-1))

    xvec = range(-half_window,half_window+1)
    if detailed_plots:
        fig, axes = plt.subplots(2,3)
    figall, axall = plt.subplots(1,1)
    #figall = plt.figure()
    #axall = figall.add_axes([0.1, 0.1, 0.9, 0.9])
    axall.plot([0,0],[0,1],',k',linestyle='--')
    #colors = ['r','g','y','c','b','m']
    colors = ['b','r','b','r','g','y']
    labels = ['+v','-v','/+v','/-v','+o','-o']

    figabstract, axabstract = plt.subplots(1,1)
    axabstract.plot([0,0],[0,1],',k',linestyle='--')

    for stimulus_index in range(6):
        if detailed_plots:
            row = stimulus_index//3
            col = stimulus_index%3
            axes[row,col].plot([0,0],[0,1],',-g')
            axes[row,col].plot(xvec, probability_action_given_stimulus\
                                        [stimulus_index,:,0],'.-r',label='nolick')
            axes[row,col].plot(xvec, probability_action_given_stimulus\
                                        [stimulus_index,:,1],'.-b',label='lick')
            axes[row,col].set_xlabel(units+' around '+trans+' transition')
            axes[row,col].set_ylabel('P(action|stimulus='+str(stimulus_index+1)+')')
            axes[row,col].set_xlim([-half_window,half_window])

        # lick probability given stimuli all in one axes
        axall.plot(xvec,exp_mean_probability_action_given_stimulus\
                            [stimulus_index,:,1], marker='x',
                            color=colors[stimulus_index],
                            label=labels[stimulus_index])
        axall.plot(xvec,probability_action_given_stimulus\
                            [stimulus_index,:,1], marker='.',
                            color=colors[stimulus_index],
                            linestyle='dotted',
                            label=labels[stimulus_index])
        axall.set_xlabel(units+' around '+trans+' transition')
        axall.set_ylabel('P(lick|stimulus)')
        axall.set_xlim([-half_window,half_window])

        axabstract.plot(xvec,probability_action_given_stimulus\
                            [stimulus_index,:,1], marker='.',
                            color=colors[stimulus_index],
                            linestyle='solid',
                            label=labels[stimulus_index])
        axabstract.set_xlabel(units+' around '+trans+' transition')
        axabstract.set_ylabel('P(lick|stimulus)')
        axabstract.set_xlim([-half_window,half_window])

    if detailed_plots:
        axes[row,col].legend()
        fig.subplots_adjust(wspace=0.5,hspace=0.5)
    axall.legend()
    figall.tight_layout()
    
    figabstract.tight_layout()
    figabstract.savefig('RL_'+agent_type+'_'+trans+'.pdf')
    figabstract.savefig('RL_'+agent_type+'_'+trans+'.svg')

def get_env_agent(agent_type='belief', ACC_off_factor=1., seed=None):
    # use one of these environments,
    #  with or without blank state at start of each trial
    # OBSOLETE - with blanks environment won't work,
    #  as I've HARDCODED observations to stimuli encoding
    #env = gym.make('visual_olfactory_attention_switch-v0')
    # HARDCODED these observations to stimuli encoding
    env = gym.make('visual_olfactory_attention_switch_no_blank-v0',
                    lick_without_reward_factor=1.)

    ############# agent instatiation and agent parameters ###########
    # you need around 1,000,000 steps for enough transitions to average over
    # ensure steps is larger than the defaults of the agent for
    #  exploration_decay_time_steps, learning_time_steps,
    #  and recording_time_steps

    # instantiate the RL agent either with history or belief
    # choose one of the below:

    if agent_type=='basic' or agent_type=='history0_nobelief':
        # with history=0 and beliefRL=False, need to keep learning always on!
        #steps = 1000000
        steps = 2000000
        # 2-param fit using brute to minimize mse,
        #  ended without success, due to exceeding max func evals
        #epsilon, alpha = 0.21886517, 0.72834129
        # does not give a sudden peak/lick in /+V stimulus in V2O transition
        #epsilon, alpha = 0.2, 0.2
        # 2-param fit using brute to minimize mse,
        #  fitting nan-s to nan-s, terminated successfully
        epsilon, alpha = 0.21947144, 0.76400787
        agent = BeliefHistoryTabularRL(env,history=0,beliefRL=False,
                                        alpha=alpha,epsilon=epsilon,seed=seed,
                                        learning_time_steps=steps,
                                        recording_time_steps=steps//2)
    elif agent_type=='history1_nobelief':
        # with history=1 and beliefRL=False, no need to keep learning always on!
        #  learning stops after learning_time_steps,
        #  then 1 step correct performance switch!
        # history=1 takes a bit longer to learn than history=2!
        steps = 2000000
        agent = BeliefHistoryTabularRL(env,history=2,beliefRL=False,seed=seed,
                                        learning_time_steps=steps,
                                        recording_time_steps=steps//2)
    elif agent_type=='history2_nobelief':
        # with history=2 and beliefRL=False, no need to keep learning always on!
        #  learning stops after learning_time_steps,
        #  then 1 step correct performance switch!
        steps = 1000000
        agent = BeliefHistoryTabularRL(env,history=2,beliefRL=False,seed=seed,
                                        learning_time_steps=steps//2,
                                        recording_time_steps=steps//2)
    elif agent_type=='belief' or agent_type=='history0_belief':
        # no history, just belief in one of two contexts - two Q tables
        # keeping exploration & learning always on
        #  for noise and belief uncertainty driven exploration
        #steps = 500000
        steps = 2000000
        # obtained by 1-param fit using powell's minimize mse
        #belief_switching_rate = 0.76837728
        # obtained by 3-param fit using powell's minimize mse -- out of bounds - gives nans
        #belief_switching_rate, epsilon, alpha = 1.694975  , 0.48196603, 2.1
        # obtained by 3-param fit using brute minimize mse
        #belief_switching_rate, epsilon, alpha = 0.6, 0.1, 0.105
        # obtained by 3-param fit using brute minimize mse (alpha fixed at 0.1)
        #belief_switching_rate, epsilon, exploration_add_factor_for_context_prediction_error, alpha \
        #            = 0.52291667, 0.10208333, 8.146875, 0.1
        # obtained by 3-param fit using brute minimize mse with nan-errors (alpha fixed at 0.1)
        belief_switching_rate, epsilon, exploration_add_factor_for_context_prediction_error, alpha \
                    = 0.54162102, 0.09999742, 8.2049604, 0.1
        agent = BeliefHistoryTabularRL(env,history=0,beliefRL=True,
                                        belief_switching_rate=belief_switching_rate,
                                        ACC_off_factor = ACC_off_factor,
                                        alpha=alpha, epsilon=epsilon, seed=seed,
                                        exploration_add_factor_for_context_prediction_error=\
                                            exploration_add_factor_for_context_prediction_error,
                                        learning_time_steps=steps,
                                        recording_time_steps=steps//2)

    return env, agent, steps

if __name__ == "__main__":

    ############## choose / uncomment one of the agents below! #################
    agent_type='belief'
    #agent_type='basic'
    
    # choose whether ACC is inhibited or not
    ACC_off = True
    #ACC_off = False
    if ACC_off:
        ACC_off_factor = 0.5 # inhibited ACC
        ACC_str = 'exp'
    else:
        ACC_off_factor = 1.0 # uninhibited ACC
        ACC_str = 'control'
    
    # Instantiate the env and the agent
    env, agent, steps = get_env_agent(agent_type, ACC_off_factor, seed=seed)
    
    # train the RL agent on the task
    exp_step, block_vector_exp_compare, \
        reward_vector_exp_compare, stimulus_vector_exp_compare, \
            action_vector_exp_compare = \
                agent.train(steps)

    detailed_plots = False

    ## obsolete - start
    #fig1 = plt.figure()
    #plt.plot(reward_vec)
    #smooth = 20 # time steps to smooth over
    #plt.plot(np.convolve(reward_vec,np.ones(smooth))/smooth)
    #plt.plot(block_vector)

    #fig2 = plt.figure()
    #plt.plot(cumulative_reward)
    ## obsolete - end

    # obtain the mean reward and action given stimulus around O2V transition
    # no need to pass above variables as they are not modified, only analysed
    average_reward_around_o2v_transition, \
        actionscount_to_stimulus_o2v, \
        probability_action_given_stimulus_o2v = \
            process_transitions(exp_step, block_vector_exp_compare,
                                reward_vector_exp_compare,
                                stimulus_vector_exp_compare,
                                action_vector_exp_compare,
                                O2V = True)

    if detailed_plots:
        fig3 = plt.figure()
        plt.plot(average_reward_around_o2v_transition)
        plt.plot([half_window,half_window],\
                    [min(average_reward_around_o2v_transition),\
                        max(average_reward_around_o2v_transition)])
        plt.xlabel('time steps around olfactory to visual transition')
        plt.ylabel('average reward on time step')

    # read experimental data
    print("reading experimental data")
    number_of_mice, across_mice_average_reward_o2v, \
        mice_average_reward_around_transtion_o2v, \
        mice_actionscount_to_stimulus_o2v, \
        mice_actionscount_to_stimulus_trials_o2v, \
        mice_probability_action_given_stimulus_o2v, \
        mean_probability_action_given_stimulus_o2v = \
            get_exp_reward_around_transition(trans='O2V',ACC=ACC_str)
    number_of_mice, across_mice_average_reward_v2o, \
        mice_average_reward_around_transtion_v2o, \
        mice_actionscount_to_stimulus_v2o, \
        mice_actionscount_to_stimulus_trials_v2o, \
        mice_probability_action_given_stimulus_v2o, \
        mean_probability_action_given_stimulus_v2o = \
            get_exp_reward_around_transition(trans='V2O',ACC=ACC_str)
    print("finished reading experimental data.")

    plot_prob_actions_given_stimuli(probability_action_given_stimulus_o2v,
                                    mean_probability_action_given_stimulus_o2v,
                                        detailed_plots, agent_type=agent_type)

    # obtain the mean reward and action given stimulus around V2O transition
    # no need to pass above variables as they are not modified, only analysed
    average_reward_around_v2o_transition, \
        actionscount_to_stimulus_v2o, \
        probability_action_given_stimulus_v2o = \
            process_transitions(exp_step, block_vector_exp_compare,
                                reward_vector_exp_compare, 
                                stimulus_vector_exp_compare,
                                action_vector_exp_compare,
                                O2V = False)

    plot_prob_actions_given_stimuli(probability_action_given_stimulus_v2o,
                                    mean_probability_action_given_stimulus_v2o,
                                        detailed_plots, agent_type=agent_type,
                                        trans='V2O')

    plt.show()
