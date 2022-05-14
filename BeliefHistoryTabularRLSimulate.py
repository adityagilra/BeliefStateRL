"""
Context-belief- and/or history-based agent

by Aditya Gilra, Sep-Oct-Nov 2021
"""

import gym
import numpy as np
import gym_tasks
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as scipyio
import sys

from BeliefHistoryTabularRL import BeliefHistoryTabularRL
from plot_simulation_data import load_plot_simdata
from plot_exp_sim_data import load_plot_expsimdata

# reproducible random number generation -- env and agent use independent RNGs with same seed currently
seed = 1

def get_env_agent(agent_type='belief', ACC_off_factor=1., seed=1, num_params_to_fit = 4):
    # use one of these environments,
    #  with or without blank state at start of each trial
    # OBSOLETE - with blanks environment won't work,
    #  as I've HARDCODED observations to stimuli encoding
    #env = gym.make('visual_olfactory_attention_switch-v0')
    # HARDCODED these observations to stimuli encoding
    env = gym.make('visual_olfactory_attention_switch_no_blank-v0',
                    reward_size=1., punish_factor=1.,
                    lick_without_reward_factor=1., seed=seed)

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

        learning_during_testing = True

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
            #epsilon, alpha = 0.21976563, 0.96556641 # fit p(lick|stimuli) for only reward-structure-known stimuli # exploration on during testing # fitted successfully with mse = 0.005474
            
            epsilon, alpha = 0.24694724, 0.9 # fit p(lick|stimuli) for all stimuli # exploration on during testing # fitted successfully with mean rmse = 0.13484248472193172 across 5 seeds fitting all data using reward structure 1, 1, 1, COBYLA local fit tol=5e-6 successfully

            #epsilon, alpha = 0.20625835, 0.9112167 # fit p(lick|stimuli) for all stimuli # exploration on during testing # fitted successfully with mse = 0.004342
            #epsilon, alpha = 0.25056614, 0.84921649 # fit p(lick|stimuli) for all stimuli using local COBYLA fit # exploration on during testing # fitted successfully with mse =  0.01824815015852233 , old mse =  0.009030705234997848, when running with seed 1: mse = 0.13487851276740936 , old mse =  0.0089601199048506
            
            # setting this to None will make it not be used
            unrewarded_visual_exploration_rate = None
            params_all = ((epsilon, alpha), learning_during_testing)
        else:
            ### 3-param fit
            #epsilon, alpha, unrewarded_visual_exploration_rate \
                    #= 0.22270257, 0.87529689, 0.3458835 # mse = 0.018739743421187977, old mse = 0.009238249518122875, COBYLA local fit tol=0.0005 successfully
            epsilon, alpha, unrewarded_visual_exploration_rate \
                    = 0.25618055, 0.78719931, 0.30161107 # mse = 0.008409035747337732, brute and then local fit fmin: did not fit -- max func evals exceeded, fit all 4 curves.
            #epsilon, alpha, unrewarded_visual_exploration_rate \
            #        = 0.29128861, 0.81420369, 0.37465497 # mse = 0.01882887557954043 , old mse =  0.009362525753524403, COBYLA local fit tol=0.0005 successfully, starting from: exploration_rate_start = 0.2, larning_rate_start = 0.8, unrewarded_visual_exploration_rate_start = 0.5
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
        #context_sampling = True
        context_sampling = False

        # another way to introduce noise in context setting could be due to a noisy integrate to threshold
        # can just add a noise (parameterized by SD below) to the context belief probabilities and then take max        
        # noise could also be from context error signal in the ACC parametrized by a context_error_noiseSD_factor

        # setting this param to None will make it not be used except when overridden below
        unrewarded_visual_exploration_rate = None

        if context_sampling:
            if num_params_to_fit == 2:
                belief_switching_rate, context_error_noiseSD_factor \
                            = 0.45787308, 3.50795822 # fit p(lick|stimuli) for only reward-structure-known stimuli # exploration off during testing, & context_sampling=True # mse = = 0.001151
                #belief_switching_rate, context_error_noiseSD_factor \
                #            = 0.594375  , 4.26855469 # fit p(lick|stimuli) for all 4 stimuli # exploration off during testing, & context_sampling=True # mse = 0.004001
                epsilon, exploration_add_factor_for_context_uncertainty, alpha = 0.1, 0, 0.1
                #unrewarded_visual_exploration_rate = 0.4

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

            elif num_params_to_fit == 4:
                ##### obtained by 4-param fit
                belief_switching_rate, context_error_noiseSD_factor, epsilon, unrewarded_visual_exploration_rate \
                            = 0.58155763, 0.65325136, 0.10499605, 0.3698203 # fit p(lick|stimuli) for all 4 stimuli # exploration on during testing, & context_sampling=True # 1 set of 5 params in 5-fold CV # seed=1 # rmse = 0.06311081856834604, test rmse = 0.10123140221991846, reward structure 1, 1, 1, COBYLA local fit tol=5e-6 successfully, starting from poor values: 0.73125, 0.50625, 0.50625, 0.45815625
                #belief_switching_rate, context_error_noiseSD_factor, epsilon, unrewarded_visual_exploration_rate \
                #            = 0.69474268, 2.10976159, 0.09317885, 0.46556799 # fit p(lick|stimuli) for all 4 stimuli # exploration on during testing, & context_sampling=True # mse = 0.003098491835327584, old mse = 0.0015597898820416504, reward structure 1, 1, 1, COBYLA local fit tol=0.0005 successfully, starting from values: 0.7, 2., 0.1, 0.4
                #belief_switching_rate, context_error_noiseSD_factor, epsilon, unrewarded_visual_exploration_rate \
                #            = 0.77983832, 0.50027674, 0.50625, 0.45619092 # fit p(lick|stimuli) for all 4 stimuli # exploration on during testing, & context_sampling=True # RMSE across 5 seeds: train =  0.06020246514855136, test =  0.1041516546778553, reward structure 1, 1, 1, COBYLA local fit tol=0.0005 successfully, starting from values: 0.73125, 0.50625, 0.50625, 0.45815625
                #belief_switching_rate, context_error_noiseSD_factor, epsilon, unrewarded_visual_exploration_rate \
                #            = 0.90470671, 1.63768158, 0.26397918, 0.44469277 # fit p(lick|stimuli) for all 4 stimuli # exploration on during testing, & context_sampling=True # mse = 0.003232071343265149 , old mse =  0.0016175240716927593, reward structure 1, 1, 1, COBYLA local fit tol=0.0005 successfully, starting from values: 0.9, 1.625, 0.255, 0.4525 -- not so good mse = 0.009899, sensitive to seed perhaps
                #belief_switching_rate, context_error_noiseSD_factor, epsilon, unrewarded_visual_exploration_rate \
                #            = 0.78155776, 0.50658442, 0.50604311, 0.45813087 # fit p(lick|stimuli) for all 4 stimuli # exploration on during testing, & context_sampling=True # mse = 0.0032915614142155743, old mse =  0.0016503114709468334, reward structure 1, 1, 1, COBYLA local fit tol=0.0005 successfully, starting from values: 0.73125, 0.50625, 0.50625, 0.45815625 -- awful mse = 0.036 when running here, possibly very sensitive to seed?
                #belief_switching_rate, context_error_noiseSD_factor, epsilon, unrewarded_visual_exploration_rate \
                            #= 0.73125, 0.50625, 0.50625, 0.45815625 # fit p(lick|stimuli) for all 4 stimuli # exploration on during testing, & context_sampling=True # mse = 0.0176209908141966, reward structure 1, 1, 1, max func evals exceeded -- did not fit
                exploration_add_factor_for_context_uncertainty, alpha = 0, 0.1

        else: ## if context_sampling == False
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
                            = 0.3172727, 4.61738219, 0.01069627
                #belief_switching_rate, context_error_noiseSD_factor, epsilon, unrewarded_visual_exploration_rate \
                #            = 0.69703441, 0., 0.09965295, 0.44441895
                exploration_add_factor_for_context_uncertainty, alpha = 0, 0.1

            elif num_params_to_fit == 4:
                ##### obtained by 4-param fit
                belief_switching_rate, context_error_noiseSD_factor, epsilon, unrewarded_visual_exploration_rate \
                            = 0.71308534, 2.05177685, 0.08250793, 0.47370851  # fit p(lick|stimuli) for all 4 stimuli # exploration on during testing, & context_sampling=False # rmse = 0.05396551893600141 across 5 seeds fitting all data using reward structure 1, 1, 1, COBYLA local fit tol=5e-6 successfully, starting from good values: 0.7, 2., 0.1, 0.4

                #belief_switching_rate, context_error_noiseSD_factor, epsilon, unrewarded_visual_exploration_rate \
                #            = 0.69703441, 2.01253643, 0.09965295, 0.44441895 # fit p(lick|stimuli) for all 4 stimuli # exploration on during testing, & context_sampling=False # 1 set of 5 params in 5-fold CV # seed=1 # rmse = 0.05923636041587643, test rmse = 0.09877881078233221, reward structure 1, 1, 1, COBYLA local fit tol=5e-6 successfully, starting from good values: 0.7, 2., 0.1, 0.4
                #belief_switching_rate, context_error_noiseSD_factor, epsilon, unrewarded_visual_exploration_rate \
                #            = 0.46048665, 4.90591006, 0.51447091, 0.44796221 # fit p(lick|stimuli) for all 4 stimuli # exploration on during testing, & context_sampling=False # mse = 0.0021553571995931737, reward structure 1, 1, 1, max func evals exceeded -- did not fit
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
    agent_type='belief'
    #agent_type='basic'

    if agent_type == 'basic':
        # choose one of the below
        num_params_to_fit = 2 # for both basic and belief RL
        #num_params_to_fit = 3 # for both basic and belief RL
    else:
        # choose one of the below
        #num_params_to_fit = 2 # for both basic and belief RL
        #num_params_to_fit = 3 # for both basic and belief RL
        num_params_to_fit = 4 # only for belief RL

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
    env, agent, steps, params_all = get_env_agent(agent_type, ACC_off_factor, seed=seed,
                                                 num_params_to_fit=num_params_to_fit)
    
    agent.reset()
    
    # train the RL agent on the task
    exp_step, block_vector_exp_compare, \
        reward_vector_exp_compare, stimulus_vector_exp_compare, \
            action_vector_exp_compare, context_record, mismatch_error_record = \
                agent.train(steps)

    print('Q-values dict {state: context x action} = ',agent.Q_array)

    savefilename = 'simulation_data/simdata_'+agent_type+'_numparams'+str(num_params_to_fit)+'_ACC'+ACC_str+'_seed'+str(seed)+'.mat'
    print('Saving to',savefilename)
    # params_all is a 'ragged' tuple with different dtypes (including NoneType),
    #  not saving it as it cannot be converted to an array to save into a .mat file
    scipyio.savemat(savefilename,
                        {'steps':steps,
                        #'params_all':params_all,
                        'exp_step':exp_step,
                        'fit_rewarded_stimuli_only':fit_rewarded_stimuli_only,
                        'agent_type':agent_type,
                        'num_params_to_fit':num_params_to_fit,
                        'ACC_str':ACC_str,
                        'seed':seed,
                        'block_vector_exp_compare':block_vector_exp_compare,
                        'reward_vector_exp_compare':reward_vector_exp_compare,
                        'stimulus_vector_exp_compare':stimulus_vector_exp_compare,
                        'action_vector_exp_compare':action_vector_exp_compare,
                        'context_record':context_record,
                        'mismatch_error_record':mismatch_error_record})    

    load_plot_expsimdata(savefilename)
    load_plot_simdata(savefilename)
