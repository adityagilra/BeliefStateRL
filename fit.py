import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from exp_data_analysis import get_exp_reward_around_transition
from BeliefHistoryTabularRL import BeliefHistoryTabularRL, get_env_agent, \
                                    process_transitions, half_window
import sys
from scipy.optimize import minimize

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
        learning_rate = parameters[2]
    else:
        exploration_rate = parameters[0]
        learning_rate = parameters[1]

    agent.alpha = learning_rate
    agent.epsilon = exploration_rate

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
    
    # error is simulated agent transitions - experimental transitions
    error_o2v = probability_action_given_stimulus_o2v \
                    - mean_probability_action_given_stimulus_o2v
    error_v2o = probability_action_given_stimulus_v2o \
                    - mean_probability_action_given_stimulus_v2o

    # in-place (copy=False) replace nan-s by 0 in error
    # nan, posinf and neginf are new in numpy 1.17, my numpy is older
    #np.nan_to_num(error_o2v, copy=False, nan=0.0, posinf=None, neginf=None)
    #np.nan_to_num(error_v2o, copy=False, nan=0.0, posinf=None, neginf=None)
    # nan-s are replaced by zeros
    error_o2v = np.nan_to_num(error_o2v, copy=False)
    error_v2o = np.nan_to_num(error_v2o, copy=False)

    # here (overall mean) = (mean of means) as same number of elements in _o2v and v2o
    mse = np.mean(np.power(error_o2v,2)) + np.mean(np.power(error_v2o,2))
    mse /= 2.0
    print("mean squared error = ",mse)

    return mse

if __name__ == "__main__":
    # read experimental data
    print("reading experimental data")
    number_of_mice, across_mice_average_reward_o2v, \
        mice_average_reward_around_transtion_o2v, \
        mice_actionscount_to_stimulus_o2v, \
        mice_actionscount_to_stimulus_trials_o2v, \
        mice_probability_action_given_stimulus_o2v, \
        mean_probability_action_given_stimulus_o2v = \
            get_exp_reward_around_transition(trans='O2V')
    number_of_mice, across_mice_average_reward_v2o, \
        mice_average_reward_around_transtion_v2o, \
        mice_actionscount_to_stimulus_v2o, \
        mice_actionscount_to_stimulus_trials_v2o, \
        mice_probability_action_given_stimulus_v2o, \
        mean_probability_action_given_stimulus_v2o = \
            get_exp_reward_around_transition(trans='V2O')
    print("finished reading experimental data.")

    #agent_type = 'belief'
    agent_type = 'basic'

    env, agent, steps = get_env_agent(agent_type=agent_type)

    if agent_type == 'belief':
        belief_switching_rate_start = 0.7
        exploration_rate_start = 0.1
        learning_rate_start = 0.1
        parameters = (belief_switching_rate_start,
                        exploration_rate_start, learning_rate_start)
    elif agent_type == 'basic':
        exploration_rate_start = 0.1
        learning_rate_start = 0.1
        parameters = (exploration_rate_start, learning_rate_start)
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
    # no runtime warning, but looks like powell can't handle bounds either!
    result = minimize( meansquarederror, parameters,
                        args=(agent_type, agent, steps,
                                mean_probability_action_given_stimulus_o2v,
                                mean_probability_action_given_stimulus_v2o),
                        bounds=((0.5,1),), 
                        #method='nelder-mead', options={'xatol':0.05, 'fatol':0.5}
                        method='powell', options={'xtol':0.1}
                        )
                        
    # could also try global optimization: 
    #  https://docs.scipy.org/doc/scipy/reference/reference/optimize.html#module-scipy.optimize
    #  brute force: https://docs.scipy.org/doc/scipy/reference/reference/generated/scipy.optimize.brute.html#scipy.optimize.brute
    #  etc.

    print(result)
