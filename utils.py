import numpy as np

def process_transitions(exp_step, block_vector_exp_compare,
                        reward_vector_exp_compare, 
                        stimulus_vector_exp_compare,
                        action_vector_exp_compare,
                        context_record, mismatch_error_record,
                        O2V=True, half_window=30):
    # exp_step, at end of time loop above, equals end index saved in _exp_compare vectors
    # clip vector at exp_step, to avoid detecting a last spurious transtion in block_vector_exp_compare

    # processing after agent simulation has ended
    # calculate mean reward and mean action given stimulus, around transitions
    transitions = \
        np.where(np.diff(block_vector_exp_compare[:exp_step])==(-1 if O2V else 1))[0] + 1
    # note that block number changes on the first time step of a new trial,
    # debug print
    #print(('O2V' if O2V else 'V2O')+" transition at steps ",transitions)
    print('Number of '+('O2V' if O2V else 'V2O')+' transitions =', len(transitions))
    
    average_reward_around_transition = np.zeros(half_window*2+1)
    actionscount_to_stimulus = np.zeros((6,half_window*2+1,2)) # 6 stimuli, 2 actions
    # above is by time steps, below is by trials
    actionscount_to_stimulus_bytrials = np.zeros((6,half_window*2+1,2)) # 6 stimuli, 2 actions
    context = np.zeros((half_window*2+1,len(context_record[0])))
    mismatch_error = np.zeros((half_window*2+1,len(context_record[0])))
    mismatch_by_perfectswitch = [[],[]]
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

        ######### actions given stimuli around transition by time steps
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
            
        ######### actions given stimuli around transition by trials
        # for stimuli 1,2 (visual block) counting by steps and by trials is the same
        actionscount_to_stimulus_bytrials[ [0,1], :, :] = actionscount_to_stimulus[ [0,1], :, :]
        # for stimuli 3-6 (olfactory block) need to collapse steps into trials
        trialnum = 0
        # go from transition towards olfactory steps / trials, need to align transitions
        if O2V: stepnums = range(transition,window_min,-1)
        else: stepnums = range(transition,window_max)
        for stepnum in stepnums:
            stimulus = stimulus_vector_exp_compare[stepnum]
            if stimulus in [5,6]: # olfactory stimulus
                if O2V: windownum = half_window-trialnum
                else: windownum = half_window+trialnum
                actionscount_to_stimulus_bytrials[ int(stimulus-1), windownum, \
                                                    int(action_vector_exp_compare[stepnum]) ] += 1
                # was there a visual stimulus of the olfactory block before this?
                if stepnum>window_min: # can't go out of bounds
                    prev_stimulus = stimulus_vector_exp_compare[stepnum-1]
                    if prev_stimulus in [3,4]:
                        actionscount_to_stimulus_bytrials[ int(prev_stimulus-1), windownum, \
                                                            int(action_vector_exp_compare[stepnum-1]) ] += 1
                trialnum += 1

        context[window_start:window_end,:] += context_record[window_min:window_max,:]

        if O2V:
            second_trial_idx = transition + 1
            correct_action = 1
        else:
            second_trial_idx = transition + 2
            correct_action = 0
        # error in both contexts is taken into account -- np.abs() to neglect direction information
        # note that context prediction error is always a factor times (0,0) or (-1,1) or (1,-1)
        if action_vector_exp_compare[second_trial_idx] == correct_action:
            mismatch_by_perfectswitch[1].append( np.sum(np.abs(mismatch_error_record[second_trial_idx-1,:])) )
        else:
            mismatch_by_perfectswitch[0].append( np.sum(np.abs(mismatch_error_record[second_trial_idx-1,:])) )

        mismatch_error[window_start:window_end,:] += mismatch_error_record[window_min:window_max,:]

        num_transitions_averaged += 1

    average_reward_around_transition /= num_transitions_averaged

    # normalize over the actions (last axis i.e. -1) to get probabilities
    # do not add a small amount to denominator to avoid divide by zero!
    # allowing nan so that irrelvant time steps are not plotted
    probability_action_given_stimulus = actionscount_to_stimulus \
                / np.sum(actionscount_to_stimulus,axis=-1)[:,:,np.newaxis] #\
                        #+ np.finfo(np.double).eps )
    probability_action_given_stimulus_bytrials = actionscount_to_stimulus_bytrials \
                / np.sum(actionscount_to_stimulus_bytrials,axis=-1)[:,:,np.newaxis] #\

    context /= num_transitions_averaged
    mismatch_error /= num_transitions_averaged

    return average_reward_around_transition, \
                actionscount_to_stimulus, \
                actionscount_to_stimulus_bytrials, \
                probability_action_given_stimulus, \
                probability_action_given_stimulus_bytrials, \
                context, mismatch_error, mismatch_by_perfectswitch


def rootmeansquarederror(transitions_actionscount_to_stimulus_o2v,
                    transitions_actionscount_to_stimulus_v2o,
                    agent_probability_action_given_stimulus_o2v,
                    agent_probability_action_given_stimulus_v2o,
                    fit_rewarded_stimuli_only, num_params,
                    fold_num=None, num_folds=None, test=False):

    if num_folds is not None:
        print('Using fold',fold_num+1,'of',num_folds)
        # transitions_actionscount_to_stimulus has dimensions:
        #  stimuli x actions x transitions x window_size
        ### o2v
        o2v_transitions_len = transitions_actionscount_to_stimulus_o2v.shape[2]
        fold_cut = o2v_transitions_len//num_folds
        if not test: # take k-1 folds for training
            fold_idxs = np.append( np.arange(0,fold_cut*fold_num,dtype=np.int),\
                                np.arange(fold_cut*(fold_num+1),o2v_transitions_len,dtype=np.int) )
        else: # take left out fold for testing
            fold_idxs = np.arange( fold_cut*fold_num, 
                                    np.min((fold_cut*(fold_num+1),o2v_transitions_len)),
                                    dtype=np.int )
        fold_actionscount_to_stimulus_o2v = transitions_actionscount_to_stimulus_o2v[:,:,fold_idxs,:]
        # divide by total number of actions taken for each stimulus.
        #  i.e. normalize lick & nolick counts to give probability
        # fold_actionscount_to_stimulus has dimensions stimuli x actions x transitions x window_size
        exp_probability_action_given_stimulus_o2v = \
            np.sum(fold_actionscount_to_stimulus_o2v,axis=2) \
                    / np.sum(fold_actionscount_to_stimulus_o2v,axis=(1,2))[:,np.newaxis,:]
        ### v2o
        v2o_transitions_len = transitions_actionscount_to_stimulus_v2o.shape[2]
        fold_cut = v2o_transitions_len//num_folds
        if not test: # take k-1 folds for training
            fold_idxs = np.append( np.arange(0,fold_cut*fold_num,dtype=np.int),\
                                np.arange(fold_cut*(fold_num+1),v2o_transitions_len,dtype=np.int) )
        else: # take left out fold for testing
            fold_idxs = np.arange( fold_cut*fold_num, 
                                    np.min((fold_cut*(fold_num+1),v2o_transitions_len)),
                                    dtype=np.int )
        fold_actionscount_to_stimulus_v2o = transitions_actionscount_to_stimulus_v2o[:,:,fold_idxs,:]
        # divide by total number of actions taken for each stimulus.
        #  i.e. normalize lick & nolick counts to give probability
        # fold_actionscount_to_stimulus has dimensions stimuli x actions x transitions x window_size
        exp_probability_action_given_stimulus_v2o = \
            np.sum(fold_actionscount_to_stimulus_v2o,axis=2) \
                    / np.sum(fold_actionscount_to_stimulus_v2o,axis=(1,2))[:,np.newaxis,:]

    else: # use all transitions in exp data for RMSE
        # divide by total number of actions taken for each stimulus.
        #  i.e. normalize lick & nolick counts to give probability
        # transitions_actionscount_to_stimulus has dimensions stimuli x actions x transitions x window_size
        exp_probability_action_given_stimulus_o2v = \
            np.sum(transitions_actionscount_to_stimulus_o2v,axis=2) \
                    / np.sum(transitions_actionscount_to_stimulus_o2v,axis=(1,2))[:,np.newaxis,:]
        exp_probability_action_given_stimulus_v2o = \
            np.sum(transitions_actionscount_to_stimulus_v2o,axis=2) \
                    / np.sum(transitions_actionscount_to_stimulus_v2o,axis=(1,2))[:,np.newaxis,:]

    # agent array has shape stimuli x windowsize x actions,
    #  while exp array computed here using folds, has stimuli x actions x windowsize
    exp_probability_action_given_stimulus_o2v = np.swapaxes(exp_probability_action_given_stimulus_o2v,1,2)
    exp_probability_action_given_stimulus_v2o = np.swapaxes(exp_probability_action_given_stimulus_v2o,1,2)

    if fit_rewarded_stimuli_only:
        # stimuli are in the order: ['+v','-v','/+v','/-v','+o','-o']
        # we select indices 0,2,4
        agent_probability_action_given_stimulus_v2o = \
            agent_probability_action_given_stimulus_v2o[[0,2,4],:,:]
        agent_probability_action_given_stimulus_o2v = \
            agent_probability_action_given_stimulus_o2v[[0,2,4],:,:]
        exp_probability_action_given_stimulus_v2o = \
            exp_probability_action_given_stimulus_v2o[[0,2,4],:,:]
        exp_probability_action_given_stimulus_o2v = \
            exp_probability_action_given_stimulus_o2v[[0,2,4],:,:]
        
    ### Only compare steps around transition to model where experiment doesn't have a nan
    nonans_o2v = ~np.isnan(exp_probability_action_given_stimulus_o2v)
    nonans_v2o = ~np.isnan(exp_probability_action_given_stimulus_v2o)
    num_nonans_o2v = np.sum(nonans_o2v.astype(int))
    num_nonans_v2o = np.sum(nonans_v2o.astype(int))
    # error is simulated agent transitions - experimental transitions
    error_o2v = agent_probability_action_given_stimulus_o2v[nonans_o2v] \
                    - exp_probability_action_given_stimulus_o2v[nonans_o2v]
    error_v2o = agent_probability_action_given_stimulus_v2o[nonans_v2o] \
                    - exp_probability_action_given_stimulus_v2o[nonans_v2o]
    
    # penalize any model data that is nan, when experiment is not
    #  we already removed exp nan-s, so any nan-s in error are due to agent nan-s
    nan_error_o2v = np.isnan(error_o2v).astype(int)
    nan_error_v2o = np.isnan(error_v2o).astype(int)

    # nan-s penalized above already, so set them to zero here, else sum for loss will become nan
    # in-place (copy=False) replace nan-s by 0-s or as desired in error
    # nan, posinf and neginf are new in numpy 1.17, my numpy is older
    #np.nan_to_num(error_o2v, copy=False, nan=0, posinf=None, neginf=None)
    #np.nan_to_num(error_v2o, copy=False, nan=0, posinf=None, neginf=None)
    # my older version of numpy hardcodes setting nan to 0-s
    error_o2v = np.nan_to_num(error_o2v, copy=False)
    error_v2o = np.nan_to_num(error_v2o, copy=False)

    # here (overall mean) = (mean of means) as same number of elements in _o2v and v2o
    #  but not when handling nan-s separately from non-nan-s
    #  but not bothering about this,
    #   this counts as a changing weighting factor for nan-s vs non nan-s
    old_mse = np.mean(np.power(error_o2v,2)) + np.mean(np.power(error_v2o,2)) \
            + np.mean(np.power(nan_error_o2v,2)) + np.mean(np.power(nan_error_v2o,2))
    old_mse /= 4.0
    mse = np.sum(np.power(error_o2v,2)) + np.sum(np.power(error_v2o,2)) \
            + np.sum(np.power(nan_error_o2v,2)) + np.sum(np.power(nan_error_v2o,2))
    # normalize to number of data points fitted in the model
    #  also subtracting the number of parameters used in the model to normalize by degrees of freedom
    #   -- this is accurate for linear models, no simple fix for non-linear models
    mse /= (num_nonans_o2v + num_nonans_v2o - num_params)
    rmse = np.sqrt(mse) # important to do sqrt after dividing above
    print("root mean squared error = ",rmse,', old mse = ',old_mse)
    
    return rmse


def simulate_and_mse(parameters,
                    agent_type, agent, steps,
                    transitions_actionscount_to_stimulus_o2v,
                    transitions_actionscount_to_stimulus_v2o,
                    fit_rewarded_stimuli_only, num_params_to_fit,
                    half_window, seeds, k_idx, k_validation, test=False):
    print("Training agent with parameters = ",parameters)
    num_seeds = len(seeds)
    rmses = np.zeros(num_seeds)
    # compute rmse for each seed and take mean of rmses across seeds.
    for seed_idx,seed in enumerate(seeds):
        print('Simulation and RMSE computation for agent with seed =',seed,
                            'and fold =',k_idx+1,'of',k_validation,'folds.')
        agent.seed = seed
        agent.reset() # also resets seeds of agent and env

        if agent_type == 'belief':
            belief_switching_rate = parameters[0]
            agent.belief_switching_rate = belief_switching_rate
            context_error_noiseSD_factor = parameters[1]
            agent.context_error_noiseSD_factor = context_error_noiseSD_factor
            if num_params_to_fit == 3:
                exploration_rate = parameters[2]
                agent.epsilon = exploration_rate
            if num_params_to_fit == 4:
                unrewarded_visual_exploration_rate = parameters[3]
                agent.unrewarded_visual_exploration_rate = unrewarded_visual_exploration_rate
            #    learning_rate = parameters[3]
            #    agent.alpha = learning_rate
            #belief_exploration_add_factor = parameters[2]
            #agent.belief_exploration_add_factor = \
            #        belief_exploration_add_factor
            #weak_visual_factor = parameters[2]
            #agent.weak_visual_factor = weak_visual_factor
        else:
            exploration_rate = parameters[0]
            learning_rate = parameters[1]
            agent.alpha = learning_rate
            agent.epsilon = exploration_rate
            if num_params_to_fit == 3:
                unrewarded_visual_exploration_rate = parameters[2]
                agent.unrewarded_visual_exploration_rate = unrewarded_visual_exploration_rate

        # train the RL agent on the task
        exp_step, block_vector_exp_compare, \
            reward_vector_exp_compare, stimulus_vector_exp_compare, \
                action_vector_exp_compare, context_record, mismatch_error_record = \
                    agent.train(steps)

        # obtain the mean reward and action given stimulus around O2V transition
        # no need to pass above variables as they are not modified, only analysed
        average_reward_around_o2v_transition, \
            actionscount_to_stimulus_o2v, \
            actionscount_to_stimulus_bytrials_o2v, \
            probability_action_given_stimulus_o2v, \
            probability_action_given_stimulus_bytrials_o2v, \
            context_o2v, mismatch_error_o2v, mismatch_by_perfectswitch_o2v = \
                process_transitions(exp_step, block_vector_exp_compare,
                                    reward_vector_exp_compare,
                                    stimulus_vector_exp_compare,
                                    action_vector_exp_compare,
                                    context_record, mismatch_error_record,
                                    O2V = True, half_window=half_window)

        # obtain the mean reward and action given stimulus around V2O transition
        # no need to pass above variables as they are not modified, only analysed
        average_reward_around_v2o_transition, \
            actionscount_to_stimulus_v2o, \
            actionscount_to_stimulus_bytrials_v2o, \
            probability_action_given_stimulus_v2o, \
            probability_action_given_stimulus_bytrials_v2o, \
            context_v2o, mismatch_error_v2o, mismatch_by_perfectswitch_v2o = \
                process_transitions(exp_step, block_vector_exp_compare,
                                    reward_vector_exp_compare,
                                    stimulus_vector_exp_compare,
                                    action_vector_exp_compare,
                                    context_record, mismatch_error_record,
                                    O2V = False, half_window=half_window)

        rmses[seed_idx] = rootmeansquarederror(transitions_actionscount_to_stimulus_o2v,
                                transitions_actionscount_to_stimulus_v2o,
                                probability_action_given_stimulus_o2v,
                                probability_action_given_stimulus_v2o,
                                fit_rewarded_stimuli_only, num_params_to_fit,
                                fold_num=k_idx, num_folds=k_validation, test=test)

        print('RMSE =',rmses[seed_idx],'for agent with seed =',seed,
                            'and fold =',k_idx+1,'of',k_validation,'folds.')
    
    rmse = np.mean(rmses)
    print('Cross-validation fold =',k_idx+1,'of',k_validation,'folds.')
    print('RMSEs of',num_seeds,'seeds =',rmses)
    print('Mean RMSE across',num_seeds,'seeds =',rmse,'for params',parameters)
    print()
    return rmse

