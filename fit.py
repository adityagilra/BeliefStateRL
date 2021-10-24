import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from exp_data_analysis import get_exp_reward_around_o2v_transition
from beliefRL import beliefRL, half_window
import sys
from scipy.optimize import minimize

def meansquarederror(belief_switching_rate,
                        across_mice_average_around_o2v_transition):

    # call the task function and obtain the mean reward around transition
    print("Simulating agent with belief_switching_rate = ",belief_switching_rate)
    average_reward_around_o2v_transition = beliefRL(belief_switching_rate)
    
    # simulated agent return same window around transition unlike experiment
    error = average_reward_around_o2v_transition[half_window-9:half_window+22] \
            - across_mice_average_around_o2v_transition

    mse = np.mean(np.power(error,2))
    print("mean squared error = ",mse)

    return mse

if __name__ == "__main__":

    belief_switching_rate_start = 0.7

    # read experimental data
    print("reading experimental data")
    across_mice_average_around_o2v_transition, \
        mice_average_reward_around_o2v_transtion, number_of_mice = \
            get_exp_reward_around_o2v_transition()
    print("finished reading experimental data.")

    #mse = meansquarederror(belief_switching_rate_start,\
    #                across_mice_average_around_o2v_transition)

    # runtime warning: Nelder-Mead cannot handle constraints or bounds!
    result = minimize( meansquarederror,(belief_switching_rate_start,),
                        args=(across_mice_average_around_o2v_transition),
                        bounds=((0.5,1),), 
                        method='Nelder-Mead', options={'xatol':0.05, 'fatol':0.5} )

    print(result)
    belief_switching_rate_fitted = result.x[0]

    # call the task function and obtain the mean reward around transition
    average_reward_around_o2v_transition = beliefRL(belief_switching_rate_fitted)

    fig = plt.figure()
    plt.plot(average_reward_around_o2v_transition[half_window-9:half_window+22],',-b')
    plt.plot(across_mice_average_around_o2v_transition,',-r')
    plt.plot([9,9],\
                [min(average_reward_around_o2v_transition),\
                    max(average_reward_around_o2v_transition)],',-k')
    plt.xlabel('time steps around olfactory to visual transition')
    plt.ylabel('average reward on time step')

    plt.show()
