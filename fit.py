import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from exp_data_analysis import get_exp_reward_around_transition
from BeliefHistoryTabularRLSimulate import get_env_agent
from utils import simulate_and_mse
import sys
from scipy.optimize import minimize, Bounds, brute
from plot_simulation_data import half_window

if __name__ == "__main__":

    agent_type = 'belief'
    #agent_type = 'basic'
    
    if agent_type == 'basic':
        # choose one of the below
        #num_params_to_fit = 2 
        num_params_to_fit = 3 
    else:
        # choose one of the below
        #num_params_to_fit = 2 
        #num_params_to_fit = 3 
        num_params_to_fit = 4 # only for belief RL

    # choose one of the two below, either fit only rewarded stimuli (+v, /+v, +o),
    #  or both rewarded and unrewarded (internally rewarded) stimuli,
    #fit_rewarded_stimuli_only = True
    fit_rewarded_stimuli_only = False

    seed = 1
    
    # choose whether ACC is inhibited or not
    #ACC_off = True
    ACC_off = False
    if ACC_off:
        ACC_off_factor = 0.5 # inhibited ACC
        ACC_str = 'exp'
    else:
        ACC_off_factor = 1.0 # uninhibited ACC
        ACC_str = 'control'

    # choose one of the two below, either fit a session only, or all mice, all sessions.
    #fit_a_session = True
    fit_a_session = False
    if fit_a_session:
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
        mean_probability_action_given_stimulus_o2v, \
        transitions_actionscount_to_stimulus_o2v = \
            get_exp_reward_around_transition(trans='O2V',ACC=ACC_str,
                                            mice_list=mice_list,sessions_list=sessions_list)
    number_of_mice, across_mice_average_reward_v2o, \
        mice_average_reward_around_transtion_v2o, \
        mice_actionscount_to_stimulus_v2o, \
        mice_actionscount_to_stimulus_trials_v2o, \
        mice_probability_action_given_stimulus_v2o, \
        mean_probability_action_given_stimulus_v2o, \
        transitions_actionscount_to_stimulus_v2o = \
            get_exp_reward_around_transition(trans='V2O',ACC=ACC_str,
                                            mice_list=mice_list,sessions_list=sessions_list)
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

    # Instantiate the env and the agent
    env, agent, steps, params_all = get_env_agent(agent_type=agent_type, 
                                        ACC_off_factor=ACC_off_factor,
                                        seed=seed,
                                        num_params_to_fit=num_params_to_fit)
    
    # steps return by agent here are much longer
    #  and fitting would take quite long, so using lower number of steps
    if agent_type == 'basic':
        steps = 1000000
    elif agent_type == 'belief':
        steps = 500000

    if agent_type == 'belief':
        belief_switching_rate_start = 0.7 #0.73125#0.7
        exploration_rate_start = 0.1 #0.50625#0.1
        learning_rate_start = 0.1
        #parameters = (belief_switching_rate_start,
        #                exploration_rate_start, learning_rate_start)
        #ranges = ((0.5,0.9),(0.1,0.5),(0.1,0.8))
        #bounds_obj = Bounds((0.5,0.,0.),(0.9,1.,1.))
        #belief_exploration_add_factor_start = 8
        #weak_visual_factor_start = 0.3
        unrewarded_visual_exploration_rate_start = 0.4 #0.45815625#0.4
        context_error_noiseSD_factor_start = 2 #0.50625#2
        if num_params_to_fit == 2:
            parameters = (belief_switching_rate_start,
                        context_error_noiseSD_factor_start)
            ranges = ((0.3,0.9),(0.5,5.))
        elif num_params_to_fit == 3:
            parameters = (belief_switching_rate_start,
                        context_error_noiseSD_factor_start,
                        exploration_rate_start)
            ranges = ((0.3,0.9),(0.5,5.),(0.01,0.5))
            #parameters = (belief_switching_rate_start,
            #            unrewarded_visual_exploration_rate_start,
            #            exploration_rate_start)
            #ranges = ((0.3,0.9),(0.01,0.6),(0.01,0.5))
        elif num_params_to_fit == 4:
            parameters = (belief_switching_rate_start,
                        context_error_noiseSD_factor_start,
                        exploration_rate_start,
                        unrewarded_visual_exploration_rate_start)
                        #learning_rate_start)
                        #weak_visual_factor_start)
                        #belief_exploration_add_factor_start)
            ranges = ((0.3,0.9),(0.5,5.),(0.01,0.5),(0.01,0.6))

    elif agent_type == 'basic':
        if num_params_to_fit == 2:
            exploration_rate_start = 0.2
            learning_rate_start = 0.9
            parameters = (exploration_rate_start, learning_rate_start)
            ranges = ((0.1,0.5),(0.1,0.9))
            #bounds_obj = Bounds((0.,0.),(1.,1.))
        elif num_params_to_fit == 3:
            exploration_rate_start = 0.2
            learning_rate_start = 0.8
            unrewarded_visual_exploration_rate_start = 0.5
            parameters = (exploration_rate_start, 
                            learning_rate_start,
                            unrewarded_visual_exploration_rate_start)
            ranges = ((0.01,0.5),(0.1,1.0),(0.01,0.6))

    else:
        print('Unimplemented agent type: ',agent_type)
        sys.exit(1)

    # local & global optimization are possible
    #  https://docs.scipy.org/doc/scipy/reference/optimize.html
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
    
    ## Ns specifies how many points per dimension for the grid.
    ## workers=-1 means use all cores available to this process,
    ##  but this requires the function to be pickleable, which isn't the case for me, so only 1 worker.
    ## Looks like the 'finish' argument is scipy.optimize.fmin by default,
    ##  so once grid point minimum is found,
    ##  fmin searches locally around this, using best grid point as a starting value
    ##  can set finish=None to just return the best point on the grid
    #result = brute(simulate_and_mse, ranges=ranges, 
    #                    args=(agent_type, agent, steps,
    #                    mean_probability_action_given_stimulus_o2v,
    #                    mean_probability_action_given_stimulus_v2o,
    #                    fit_rewarded_stimuli_only, num_params_to_fit, half_window, seed),
    #                    Ns=5, full_output=True, disp=True, workers=1)

    # Brute optimization is too slow; and compared to a coarse grid, manual tuning is better!
    # Set good starting parameters manually and then do local optimization

    # number of seeds used per mse calculation given params
    #seeds = (1,2,3,4,5)
    seeds = (1,)
    k_validation = 5 # how many fold validation
    train_rmses = []
    params_k = []
    test_rmses = []
    for k_idx in range(k_validation):
        result = minimize( simulate_and_mse, parameters,
                            args=(agent_type, agent, steps,
                                    transitions_actionscount_to_stimulus_o2v,
                                    transitions_actionscount_to_stimulus_v2o,
                                    fit_rewarded_stimuli_only, num_params_to_fit,
                                    half_window, seeds, k_idx, k_validation),
                            method='COBYLA', options={'tol':1e-6, 'rhobeg':0.05, 'disp':True}
                            )
        train_rmses.append(result.fun)
        params_k.append(result.x)
        
        test_rmse = simulate_and_mse(result.x, agent_type, agent, steps,
                                    transitions_actionscount_to_stimulus_o2v,
                                    transitions_actionscount_to_stimulus_v2o,
                                    fit_rewarded_stimuli_only, num_params_to_fit,
                                    half_window, seeds, k_idx, k_validation, test=True)
        test_rmses.append(test_rmse)

        print('Training result =',result)
        print('Mean test RMSE =',test_rmse)

    print(k_validation,'-fold cross-validation params:',params_k)
    print(k_validation,'-fold cross-validation training rmses =',train_rmses)
    print(k_validation,'-fold cross-validation test rmses =',test_rmses)

