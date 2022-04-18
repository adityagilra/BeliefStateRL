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
from utils import meansquarederror,  process_transitions

# reproducible random number generation
seed = 1
np.random.seed(seed)

# number of steps on each side of the transition to consider
half_window = 30

def plot_prob_actions_given_stimuli(probability_action_given_stimulus,
                                        exp_mean_probability_action_given_stimulus,
                                        context, mismatch_error,
                                        detailed_plots, abstract_plots,
                                        agent_type='',
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
    figall, axall = plt.subplots(1,2)
    #figall = plt.figure()
    #axall = figall.add_axes([0.1, 0.1, 0.9, 0.9])
    # dotted vertical line for transition
    axall[0].plot([0,0],[0,1],',k',linestyle='--')
    axall[1].plot([0,0],[0,1],',k',linestyle='--')
    #colors = ['r','g','y','c','b','m']
    colors = ['b','r','b','r','g','y']
    labels = ['+v','-v','/+v','/-v','+o','-o']

    if abstract_plots:
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
        axall[0].plot(xvec,exp_mean_probability_action_given_stimulus\
                            [stimulus_index,:,1], marker='x',
                            color=colors[stimulus_index],
                            label=labels[stimulus_index])
        axall[0].plot(xvec,probability_action_given_stimulus\
                            [stimulus_index,:,1], marker='.',
                            color=colors[stimulus_index],
                            linestyle='dotted',
                            label=labels[stimulus_index])
        axall[0].set_xlabel(units+' around '+trans+' transition')
        axall[0].set_ylabel('P(lick|stimulus)')
        axall[0].set_xlim([-half_window,half_window])

        if abstract_plots:
            axabstract.plot(xvec,probability_action_given_stimulus\
                                [stimulus_index,:,1], marker='.',
                                color=colors[stimulus_index],
                                linestyle='solid',
                                label=labels[stimulus_index])
            axabstract.set_xlabel(units+' around '+trans+' transition')
            axabstract.set_ylabel('P(lick|stimulus)')
            axabstract.set_xlim([-half_window,half_window])

    if agent_type=='belief':
        # context beliefs and mismatch (context prediction error) signals
        axall[1].plot(xvec,context[:,0],',-c',label='vis')
        axall[1].plot(xvec,context[:,1],',-m',label='olf')
        axall[1].plot(xvec,mismatch_error[:,0],',c',linestyle='dotted',label='vis_mis')
        axall[1].plot(xvec,mismatch_error[:,1],',m',linestyle='dotted',label='olf_mis')
        axall[1].set_xlabel(units+' around '+trans+' transition')
        axall[1].set_ylabel('Belief prob (solid) and mismatch error (dotted)')
        axall[1].set_xlim([-half_window,half_window])

    if detailed_plots:
        axes[row,col].legend()
        fig.subplots_adjust(wspace=0.5,hspace=0.5)
    axall[0].legend()
    axall[1].legend()
    figall.tight_layout()
    
    if abstract_plots:
        if agent_type=='belief':
            axabstract.plot(xvec,context[:,0],',-c',label='vis')
            axabstract.plot(xvec,context[:,1],',-m',label='olf')
        figabstract.tight_layout()
        figabstract.savefig('RL_'+agent_type+'_'+trans+'.pdf')
        figabstract.savefig('RL_'+agent_type+'_'+trans+'.svg')

def plot_mismatch_vs_perfectswitch(mismatch_by_perfectswitch_o2v, mismatch_by_perfectswitch_v2o):
    fig, ax = plt.subplots(1,2)
    ax[0].bar( ['wrong switch','correct switch'],
                [np.mean(mismatch_by_perfectswitch_o2v[0]),np.mean(mismatch_by_perfectswitch_o2v[1])],
                yerr=[np.std(mismatch_by_perfectswitch_o2v[0]),np.std(mismatch_by_perfectswitch_o2v[1])] )
    ax[0].set_ylabel('mismatch error O2V')
    ax[1].bar( ['wrong switch','correct switch'],
                [np.mean(mismatch_by_perfectswitch_v2o[0]),np.mean(mismatch_by_perfectswitch_v2o[1])],
                yerr=[np.std(mismatch_by_perfectswitch_v2o[0]),np.std(mismatch_by_perfectswitch_v2o[1])] )
    ax[1].set_ylabel('mismatch error V2O')
    fig.tight_layout()

def get_env_agent(agent_type='belief', ACC_off_factor=1., seed=None, num_params_to_fit = 3):
    # use one of these environments,
    #  with or without blank state at start of each trial
    # OBSOLETE - with blanks environment won't work,
    #  as I've HARDCODED observations to stimuli encoding
    #env = gym.make('visual_olfactory_attention_switch-v0')
    # HARDCODED these observations to stimuli encoding
    env = gym.make('visual_olfactory_attention_switch_no_blank-v0',
                    reward_size=1., punish_factor=1.,
                    lick_without_reward_factor=1.)

    ############# agent instatiation and agent parameters ###########
    # you need around 1,000,000 steps for enough transitions to average over
    # ensure steps is larger than the defaults of the agent for
    #  exploration_decay_time_steps, learning_time_steps,
    #  and recording_time_steps

    ############# ACHTUNG: setting learning_time_steps =
    # steps means that learning is always on, steps//2 means that learning stops when recording

    # instantiate the RL agent either with history or belief
    # choose one of the below:

    if agent_type=='basic' or agent_type=='history0_nobelief':
        # with history=0 and beliefRL=False, need to keep learning always on!
        #steps = 1000000
        steps = 2000000

        num_params_to_fit = 3
        if num_params_to_fit == 2:
            ### 2-param fit using brute to minimize mse,
            #  ended without success, due to exceeding max func evals
            #epsilon, alpha = 0.21886517, 0.72834129
            ## does not give a sudden peak/lick in /+V stimulus in V2O transition
            #epsilon, alpha = 0.2, 0.2
            ### 2-param fit using brute to minimize mse,
            #  fitting nan-s to nan-s [update: was bug here], terminated successfully
            #epsilon, alpha = 0.21947144, 0.76400787
            ### 2-param fit to only 'rewarded' stimuli +v, /+v, +o
            #  (though /+v i.e. +v in odor block is not rewarded!)
            #  i.e. we avoid fitting to stimuli -v, /-v and -o
            #  which are not rewarded and the punishment value is unclear.
            #  with bug-fixed nan-s to nan-s fitting,
            epsilon, alpha = 0.21976563, 0.96556641 # fit p(lick|stimuli) for only reward-structure-known stimuli # exploration on during testing # fitted successfully with mse = 0.005474
            #epsilon, alpha = 0.20625835, 0.9112167 # fit p(lick|stimuli) for all stimuli # exploration on during testing # fitted successfully with mse = 0.004342
            #unrewarded_visual_exploration_rate = 0.4
        else:
            ### 3-param fit using brute to minimize mse,
            epsilon, alpha, unrewarded_visual_exploration_rate = 0.2, 0.53958333, 0.29758681
            # mse = 0.004276240120155456, did not fit -- max func evals exceeded, fit all 4 curves.

        learning_during_testing = True
        params_all = ((epsilon, alpha, unrewarded_visual_exploration_rate), learning_during_testing)
        agent = BeliefHistoryTabularRL(env,history=0,beliefRL=False,
                                        alpha=alpha,epsilon=epsilon,seed=seed,
                                        unrewarded_visual_exploration_rate=unrewarded_visual_exploration_rate,
                                        learning_during_testing=learning_during_testing)
    elif agent_type=='history1_nobelief':
        # with history=1 and beliefRL=False, no need to keep learning always on!
        #  learning stops after learning_time_steps,
        #  then 1 step correct performance switch!
        # history=1 takes a bit longer to learn than history=2!
        steps = 2000000
        learning_during_testing = False
        agent = BeliefHistoryTabularRL(env,history=2,beliefRL=False,seed=seed,
                                        learning_during_testing=learning_during_testing)
    elif agent_type=='history2_nobelief':
        # with history=2 and beliefRL=False, no need to keep learning always on!
        #  learning stops after learning_time_steps,
        #  then 1 step correct performance switch!
        steps = 1000000
        learning_during_testing = False
        agent = BeliefHistoryTabularRL(env,history=2,beliefRL=False,seed=seed,
                                        learning_during_testing=learning_during_testing)
    elif agent_type=='belief' or agent_type=='history0_belief':
        # no history, just belief in one of two contexts - two Q tables
        steps = 500000
        #steps = 2000000

        # obtained by 1-param fit using powell's minimize mse
        #belief_switching_rate = 0.76837728

        # obtained by 3-param fit using powell's minimize mse -- out of bounds - gives nans
        #belief_switching_rate, epsilon, alpha = 1.694975  , 0.48196603, 2.1

        # obtained by 3-param fit using brute minimize mse
        #belief_switching_rate, epsilon, alpha = 0.6, 0.1, 0.105

        # obtained by 3-param fit using brute minimize mse (alpha fixed at 0.1)
        #belief_switching_rate, epsilon, exploration_add_factor_for_context_uncertainty, alpha \
        #            = 0.52291667, 0.10208333, 8.146875, 0.1

        ## At every time step, in decide_action(), a context is assumed

        # one way of introducing noise in switching i.e. after a transition could be due to
        #  sampling current context from the context belief probability distribution
        # choose whether context_sampling is True or False below
        context_sampling = True
        #context_sampling = False

        # another way to introduce noise in context setting could be due to a noisy integrate to threshold
        # can just add a noise (parameterized by SD below) to the context belief probabilities and then take max
        #context_noiseSD = 
        
        # noise could also be from context error signal in the ACC parametrized by a context_error_noiseSD_factor
        if context_sampling:
            if num_params_to_fit == 2:
                belief_switching_rate, context_error_noiseSD_factor \
                            = 0.45787308, 3.50795822 # fit p(lick|stimuli) for only reward-structure-known stimuli # exploration off during testing, & context_sampling=True # mse = = 0.001151
                #belief_switching_rate, context_error_noiseSD_factor \
                #            = 0.594375  , 4.26855469 # fit p(lick|stimuli) for all 4 stimuli # exploration off during testing, & context_sampling=True # mse = 0.004001
                epsilon, exploration_add_factor_for_context_uncertainty, alpha = 0.1, 0, 0.1
                unrewarded_visual_exploration_rate = 0.4

                #context_error_noiseSD_factor = 0.            
                ##belief_switching_rate, epsilon, exploration_add_factor_for_context_uncertainty, alpha \
                ##            = 0.6, 0.1, 0, 0.1

            elif num_params_to_fit == 3:
                ## obtained by 3-param fit using brute minimize mse
                ##  with nan-errors i.e. nan-s matched with nan-s, else error ~ 1 per nan [with bug]
                ##  alpha fixed at 0.1 for the fitting
                ## params used for CoSyNe abstract
                belief_switching_rate, epsilon, exploration_add_factor_for_context_uncertainty, alpha \
                            = 0.54162102, 0.09999742, 8.2049604, 0.1

            elif num_params_to_fit == 3:
                ##### obtained by 4-param fit
                belief_switching_rate, context_error_noiseSD_factor, epsilon, unrewarded_visual_exploration_rate \
                            = 0.3013795 , 1.6326803 , 0.37694708, 0.62256759 # fit p(lick|stimuli) for all 4 stimuli # exploration on during testing, & context_sampling=True # mse = 0.0017315496769514552, reward structure 1, 1, 1, max func evals exceeded -- did not fit
                exploration_add_factor_for_context_uncertainty, alpha = 0, 0.1

        else:
            if num_params_to_fit == 2:
                ##### obtained by 2-param fit -- note: switching rate is at the border of allowed, so redo
                ##belief_switching_rate, context_error_noiseSD_factor \
                ##           = 0.30890625, 2.01875 # fit p(lick|stimuli) for only reward-structure-known stimuli # exploration on during testing, & context_sampling=False, reward structure 10, 0.5, 1
                             #= 0.285, 2.1 fit p(lick|stimuli) for all stimuli (buggy nan-s fitting) # exploration on during testing, and context_sampling is False, reward structure 10, 0.5, 1
                             ##= 0.490625, 3.065625
                 
                belief_switching_rate, context_error_noiseSD_factor \
                            = 0.43417968, 4.13535097 # fit p(lick|stimuli) for only reward-structure known stimuli # exploration on during testing, & context_sampling=False # mse = 0.0006851, reward structure 1, 1, 1
                            #= 0.45844917, 2.71556625 # fit p(lick|stimuli) for all 4 stimuli # exploration on during testing, & context_sampling=False # mse = 0.004846, reward structure 1, 1, 1

                            #= 0.18940429, 1.34079591 # # fit p(lick|stimuli) for only reward-structure-known stimuli # exploration off during testing, & context_sampling=True
                            #= 0.86282212, 3.09287167 # fit p(lick|stimuli) for all stimuli # exploration off during testing, & context_sampling=True
                epsilon, exploration_add_factor_for_context_uncertainty, alpha \
                            = 0.1, 0, 0.1

            elif num_params_to_fit == 3:
                ##### obtained by 3-param fit
                belief_switching_rate, context_error_noiseSD_factor, epsilon \
                            = 0.3172727, 4.61738219, 0.01069627 # fit p(lick|stimuli) for all 4 stimuli # exploration on during testing, & context_sampling=False # mse = 0.003145373498646748, reward structure 1, 1, 1
                exploration_add_factor_for_context_uncertainty, alpha = 0, 0.1

            elif num_params_to_fit == 4:
                ##### obtained by 4-param fit
                belief_switching_rate, context_error_noiseSD_factor, epsilon, unrewarded_visual_exploration_rate \
                            = 0.46048665, 4.90591006, 0.51447091, 0.44796221 # fit p(lick|stimuli) for all 4 stimuli # exploration on during testing, & context_sampling=False # mse = 0.0021553571995931737, reward structure 1, 1, 1, max func evals exceeded -- did not fit
                exploration_add_factor_for_context_uncertainty, alpha = 0, 0.1

                #belief_switching_rate, context_error_noiseSD_factor, epsilon, alpha \
                #            = 0.3021444 , 0.50150228, 0.49937703, 0.13204827 # fit p(lick|stimuli) for all 4 stimuli # exploration on during testing, & context_sampling=False # mse = 0.002568355577903245, reward structure 1, 1, 1
                #exploration_add_factor_for_context_uncertainty = 0

        # choose one of the below, typically learning is off during testing
        #learning_during_testing = False
        # keep exploration & learning always on (e.g. to match mouse exploration)
        learning_during_testing = True
        
        # choose one of the two below:
        #exploration_is_modulated_by_context_uncertainty = True
        exploration_is_modulated_by_context_uncertainty = False
        if exploration_is_modulated_by_context_uncertainty:
            # learning should be on for exploration driven by belief uncertainty
            learning_during_testing = True

        params_all = ( (belief_switching_rate, context_error_noiseSD_factor, epsilon, unrewarded_visual_exploration_rate,
                                exploration_add_factor_for_context_uncertainty, alpha),
                         learning_during_testing, context_sampling )
        agent = BeliefHistoryTabularRL(env,history=0,beliefRL=True,
                                        belief_switching_rate=belief_switching_rate,
                                        ACC_off_factor = ACC_off_factor,
                                        alpha=alpha, epsilon=epsilon, seed=seed,
                                        learning_during_testing=learning_during_testing,
                                        context_sampling = context_sampling,
                                        context_error_noiseSD_factor = context_error_noiseSD_factor,
                                        unrewarded_visual_exploration_rate = unrewarded_visual_exploration_rate,
                                        exploration_is_modulated_by_context_uncertainty=\
                                            exploration_is_modulated_by_context_uncertainty,
                                        exploration_add_factor_for_context_uncertainty=\
                                            exploration_add_factor_for_context_uncertainty)

    return env, agent, steps, params_all

if __name__ == "__main__":

    ############## choose / uncomment one of the agents below! #################
    #agent_type='belief'
    agent_type='basic'

    # choose one of the below
    #num_params_to_fit = 2 # for both basic and belief RL
    num_params_to_fit = 3 # for both basic and belief RL
    #num_params_to_fit = 4 # only for belief RL

    # choose one of the two below, either fit only rewarded stimuli (+v, /+v, +o),
    #  or both rewarded and unrewarded (internally rewarded) stimuli,
    #fit_rewarded_stimuli_only = True
    fit_rewarded_stimuli_only = False
    
    # choose whether ACC is inhibited or not
    #ACC_off = True
    ACC_off = False
    if ACC_off:
        ACC_off_factor = 0.5 # inhibited ACC
        ACC_str = 'exp'
    else:
        ACC_off_factor = 1.0 # uninhibited ACC
        ACC_str = 'control'
    
    # Instantiate the env and the agent
    num_params_to_fit = 3
    env, agent, steps, params_all = get_env_agent(agent_type, ACC_off_factor, seed=seed,
                                                 num_params_to_fit=num_params_to_fit)
    
    np.random.seed(seed)
    # train the RL agent on the task
    exp_step, block_vector_exp_compare, \
        reward_vector_exp_compare, stimulus_vector_exp_compare, \
            action_vector_exp_compare, context_record, mismatch_error_record = \
                agent.train(steps)

    print('Q-values dict {state: context x action} = ',agent.Q_array)
    
    detailed_plots = False
    abstract_plots = True#False

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
        probability_action_given_stimulus_o2v, \
        context_o2v, mismatch_error_o2v, mismatch_by_perfectswitch_o2v = \
            process_transitions(exp_step, block_vector_exp_compare,
                                reward_vector_exp_compare,
                                stimulus_vector_exp_compare,
                                action_vector_exp_compare,
                                context_record, mismatch_error_record,
                                O2V = True, half_window=half_window)

    if detailed_plots:
        fig3 = plt.figure()
        plt.plot(average_reward_around_o2v_transition)
        plt.plot([half_window,half_window],\
                    [min(average_reward_around_o2v_transition),\
                        max(average_reward_around_o2v_transition)])
        plt.xlabel('time steps around olfactory to visual transition')
        plt.ylabel('average reward on time step')

    # choose one of the two below, either load exp data for 1 session only, or for all mice, all sessions.
    #load_a_session = True
    load_a_session = False
    if load_a_session:
        mice_list = [0]
        sessions_list = [0]
    else:
        mice_list = None
        sessions_list = None

    # read experimental data
    print("reading experimental data")
    number_of_mice, across_mice_average_reward_o2v, \
        mice_average_reward_around_transtion_o2v, \
        mice_actionscount_to_stimulus_o2v, \
        mice_actionscount_to_stimulus_trials_o2v, \
        mice_probability_action_given_stimulus_o2v, \
        mean_probability_action_given_stimulus_o2v = \
            get_exp_reward_around_transition(trans='O2V',ACC=ACC_str,
                                            mice_list=mice_list,sessions_list=sessions_list)
    number_of_mice, across_mice_average_reward_v2o, \
        mice_average_reward_around_transtion_v2o, \
        mice_actionscount_to_stimulus_v2o, \
        mice_actionscount_to_stimulus_trials_v2o, \
        mice_probability_action_given_stimulus_v2o, \
        mean_probability_action_given_stimulus_v2o = \
            get_exp_reward_around_transition(trans='V2O',ACC=ACC_str,
                                            mice_list=mice_list,sessions_list=sessions_list)
    print("finished reading experimental data.")

    plot_prob_actions_given_stimuli(probability_action_given_stimulus_o2v,
                                    mean_probability_action_given_stimulus_o2v,
                                    context_o2v, mismatch_error_o2v,
                                    detailed_plots, abstract_plots,
                                    agent_type=agent_type)

    # obtain the mean reward and action given stimulus around V2O transition
    # no need to pass above variables as they are not modified, only analysed
    average_reward_around_v2o_transition, \
        actionscount_to_stimulus_v2o, \
        probability_action_given_stimulus_v2o, \
        context_v2o, mismatch_error_v2o, mismatch_by_perfectswitch_v2o = \
            process_transitions(exp_step, block_vector_exp_compare,
                                reward_vector_exp_compare, 
                                stimulus_vector_exp_compare,
                                action_vector_exp_compare,
                                context_record, mismatch_error_record,
                                O2V = False, half_window=half_window)

    plot_prob_actions_given_stimuli(probability_action_given_stimulus_v2o,
                                    mean_probability_action_given_stimulus_v2o,
                                    context_v2o, mismatch_error_v2o,
                                    detailed_plots, abstract_plots,
                                    agent_type=agent_type,
                                    trans='V2O')

    plot_mismatch_vs_perfectswitch(mismatch_by_perfectswitch_o2v, mismatch_by_perfectswitch_v2o)

    if agent_type == 'basic':
        print( "(epsilon, alpha, unrewarded_visual_exploration_rate), learning_during_testing)", params_all )
    else:
        print( """(belief_switching_rate, context_error_noiseSD_factor, epsilon, unrewarded_visual_exploration_rate,
                                exploration_add_factor_for_context_uncertainty, alpha),
                         learning_during_testing, context_sampling )""", params_all )
    print( 'Mean squared error = ', 
                meansquarederror(mean_probability_action_given_stimulus_o2v,
                                mean_probability_action_given_stimulus_v2o,
                                probability_action_given_stimulus_o2v,
                                probability_action_given_stimulus_v2o,
                                fit_rewarded_stimuli_only) )

    plt.show()
