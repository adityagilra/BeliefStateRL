import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from exp_data_analysis import get_exp_reward_around_transition
from BeliefHistoryTabularRLSimulate import get_env_agent, \
                                    process_transitions, half_window
import sys
from scipy.optimize import minimize, Bounds, brute

def meansquarederror(parameters,
                        agent_type, agent, steps,
                        mean_probability_action_given_stimulus_o2v,
                        mean_probability_action_given_stimulus_v2o):

    print("Training agent with parameters = ",parameters)
    agent.reset()

    if agent_type == 'belief':
        belief_switching_rate = parameters[0]
        agent.belief_switching_rate = belief_switching_rate
        exploration_rate = parameters[1]
        agent.epsilon = exploration_rate
        #learning_rate = parameters[2]
        #agent.alpha = learning_rate
        belief_exploration_add_factor = parameters[2]
        agent.belief_exploration_add_factor = \
                belief_exploration_add_factor
    else:
        exploration_rate = parameters[0]
        learning_rate = parameters[1]
        agent.alpha = learning_rate
        agent.epsilon = exploration_rate

    # since I use different number of training steps
    #  than the agent was initialized envisaged for,
    #  I need to adjust the learning and recording time steps
    agent.learning_time_steps=steps
    agent.recording_time_steps=steps//2

    # train the RL agent on the task
    exp_step, block_vector_exp_compare, \
        reward_vector_exp_compare, stimulus_vector_exp_compare, \
            action_vector_exp_compare = \
                agent.train(steps)

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

    # replace nan-s by -0.5 in agent behaviour (already done for experiment)
    # this ensures that nan-s are mapped to nan-s for the fitting,
    # while remaining in absolute range of other values which are in [0,1]
    # and also distinguishable from them (being negative)
    # nopes this will force agent fits to be 0 as that is closest to -0.5,
    # unless the fiting algo discovers that nan is possible (but discontinuous jump)
    #probability_action_given_stimulus_o2v[\
    #    np.isnan(probability_action_given_stimulus_o2v)] = -0.5
    #probability_action_given_stimulus_v2o[\
    #    np.isnan(probability_action_given_stimulus_v2o)] = -0.5
    
    # boolean astype(int) subtraction will be 1 or -1 if not same, 0 if same,
    #  so error will be in same absolute range as other errors
    nan_error_o2v = np.isnan(probability_action_given_stimulus_o2v).astype(int) \
                        - np.isnan(probability_action_given_stimulus_o2v).astype(int)
    nan_error_v2o = np.isnan(probability_action_given_stimulus_v2o).astype(int) \
                        - np.isnan(probability_action_given_stimulus_v2o).astype(int)
    
    # error is simulated agent transitions - experimental transitions
    #  nan-0=nan and 0-nan=nan, nan-s already handled above
    #  nan-s here are set to 0 below, so these won't contribute to error
    error_o2v = probability_action_given_stimulus_o2v \
                    - mean_probability_action_given_stimulus_o2v
    error_v2o = probability_action_given_stimulus_v2o \
                    - mean_probability_action_given_stimulus_v2o

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
    mse = np.mean(np.power(error_o2v,2)) + np.mean(np.power(error_v2o,2)) \
            + np.mean(np.power(nan_error_o2v,2)) + np.mean(np.power(nan_error_v2o,2))
    mse /= 4.0
    print("mean squared error = ",mse)

    return mse

if __name__ == "__main__":
    
    # choose whether ACC is inhibited or not
    ACC_off = True
    #ACC_off = False
    if ACC_off:
        ACC_off_factor = 0.5 # inhibited ACC
        ACC_str = 'exp'
    else:
        ACC_off_factor = 1.0 # uninhibited ACC
        ACC_str = 'control'

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

    # replace nan-s by -0.5 in mouse behaviour (done for agent in fitting)
    # this ensures that nan-s are mapped to nan-s for the fitting,
    # while remaining in absolute range of other values which are in [0,1]
    # and also distinguishable from them (being negative)
    # nopes this will force agent fits to be 0 as that is closest to -0.5
    # unless the fiting algo discovers that nan is possible (but discontinuous jump)
    #mean_probability_action_given_stimulus_o2v[\
    #    np.isnan(mean_probability_action_given_stimulus_o2v)] = -0.5
    #mean_probability_action_given_stimulus_v2o[\
    #    np.isnan(mean_probability_action_given_stimulus_v2o)] = -0.5

    agent_type = 'belief'
    #agent_type = 'basic'

    seed = 1

    # Instantiate the env and the agent
    env, agent, steps = get_env_agent(agent_type=agent_type, 
                                        ACC_off_factor=ACC_off_factor,
                                        seed=seed)
    
    # steps return by agent here are much longer
    #  and fitting would take quite long, so using lower number of steps
    if agent_type == 'basic':
        steps = 1000000
    elif agent_type == 'belief':
        steps = 500000

    if agent_type == 'belief':
        belief_switching_rate_start = 0.7
        exploration_rate_start = 0.1
        #learning_rate_start = 0.1
        #parameters = (belief_switching_rate_start,
        #                exploration_rate_start, learning_rate_start)
        #ranges = ((0.5,0.9),(0.1,0.5),(0.1,0.8))
        #bounds_obj = Bounds((0.5,0.,0.),(0.9,1.,1.))
        belief_exploration_add_factor_start = 8 
        parameters = (belief_switching_rate_start,
                        exploration_rate_start,
                        belief_exploration_add_factor_start)
        ranges = ((0.5,0.9),(0.1,0.4),(3,10))
    elif agent_type == 'basic':
        exploration_rate_start = 0.1
        learning_rate_start = 0.1
        parameters = (exploration_rate_start, learning_rate_start)
        ranges = ((0.1,0.5),(0.1,0.9))
        bounds_obj = Bounds((0.,0.),(1.,1.))
    else:
        print('Unimplemented agent type: ',agent_type)
        sys.exit(1)

    # local & global optimization are possible
    #  https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
    #  https://docs.scipy.org/doc/scipy/reference/reference/optimize.html#module-scipy.optimize
    
    # local optimization
    # both nelder-mead and powell don't compute gradients

    # nelder-mead: https://docs.scipy.org/doc/scipy/reference/reference/optimize.minimize-neldermead.html
    #  xatol is the absolute error in solution xopt,
    #  fatol is absolute error in function to optimize
    # runtime warning: nelder-mead cannot handle constraints or bounds!

    # powell: https://docs.scipy.org/doc/scipy/reference/reference/optimize.minimize-powell.html
    #  xtol is the relative error in solution xopt,
    #  ftol is relative error in function to optimize
    # powell respects the bounds inside its 'options' -- see above URL!
    # 'OptimizeWarning: Unknown solver options: bounds': perhaps a version issue?!
    #result = minimize( meansquarederror, parameters,
    #                    args=(agent_type, agent, steps,
    #                            mean_probability_action_given_stimulus_o2v,
    #                            mean_probability_action_given_stimulus_v2o),
    #                    bounds=bounds_obj,
    #                    #method='Nelder-Mead', options={'xatol':0.05, 'fatol':0.5}
    #                    method='Powell', options={'xtol':0.1, 'bounds':bounds_obj, 'return_all':True}
    #                    )
                        
    # global optimization since non-gradient ones above do not respect bounds.
    # global optimization:
    #  https://docs.scipy.org/doc/scipy/reference/reference/optimize.html#module-scipy.optimize
    #  brute force: https://docs.scipy.org/doc/scipy/reference/reference/generated/scipy.optimize.brute.html#scipy.optimize.brute
    #  etc.
    
    # workers=-1 means use all cores available to this process, Ns specify how to divide the grid
    #  but this requires the function to be pickable, which isn't the case for me, so only 1 worker
    # looks like the 'finish' argument is scipy.optimize.fmin by default,
    #  so once grid point minimum is found,
    #  fmin searches locally around this, using best grid point as a starting value
    #  can set finish=None to just return the best grid point
    result = brute(meansquarederror, ranges=ranges, 
                        args=(agent_type, agent, steps,
                        mean_probability_action_given_stimulus_o2v,
                        mean_probability_action_given_stimulus_v2o),
                        Ns=5, full_output=True, disp=True, workers=1)

    print(result)
