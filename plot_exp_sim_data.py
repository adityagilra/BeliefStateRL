"""
Context-belief- and/or history-based agent

by Aditya Gilra, Sep-Oct-Nov 2021
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as scipyio
import scipy.stats
import sys

from exp_data_analysis import get_exp_reward_around_transition, \
    mouse_behaviour_data,mouse_behaviour_data_newest,mouse_neural_data,mouse_behaviour_for_neural_data
from utils import rootmeansquarederror,  process_transitions, get_switchtimes
from plot_simulation_data import load_simdata_seeds, plot_mismatch_vs_perfectswitch

# number of steps on each side of the transition to consider
half_window = 30

def plot_prob_actions_given_stimuli(probability_action_given_stimulus,
                                        probability_action_given_stimulus_bytrials,
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
    figall, axall = plt.subplots(1,2,figsize=(10, 5))
    
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
        figabstract.savefig('RL_'+agent_type+'_'+trans+'_cosyne.pdf')
        figabstract.savefig('RL_'+agent_type+'_'+trans+'_cosyne.svg')

def load_plot_expsimdata(filenamebase,seeds,new_data,mouse_behaviour_data):

    filenamebase += ( '_newdata' if new_data==1 else ('_newdata2' if new_data==2 else'') )

    exp_steps, ACC_str, agent_type,\
        fit_rewarded_stimuli_only,\
        num_params_to_fit,\
        block_vector_exp_compare,\
        reward_vector_exp_compare,\
        stimulus_vector_exp_compare,\
        action_vector_exp_compare,\
        context_record,\
        mismatch_error_record = load_simdata_seeds(filenamebase,seeds)

    if (new_data in (1,2)) and ACC_str=='exp':
        print('New/newest (behaviour+neural) data does not have data for ACC off i.e. exp condition.')
        sys.exit(1)

    detailed_plots = False
    abstract_plots = False

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
        mean_probability_action_given_stimulus_o2v, \
        transitions_actionscount_to_stimulus_o2v = \
            get_exp_reward_around_transition(trans='O2V',ACC=ACC_str,
                                            mice_list=mice_list,sessions_list=sessions_list,
                                            mouse_behaviour_data=mouse_behaviour_data)
    number_of_mice, across_mice_average_reward_v2o, \
        mice_average_reward_around_transtion_v2o, \
        mice_actionscount_to_stimulus_v2o, \
        mice_actionscount_to_stimulus_trials_v2o, \
        mice_probability_action_given_stimulus_v2o, \
        mean_probability_action_given_stimulus_v2o, \
        transitions_actionscount_to_stimulus_v2o = \
            get_exp_reward_around_transition(trans='V2O',ACC=ACC_str,
                                            mice_list=mice_list,sessions_list=sessions_list,
                                            mouse_behaviour_data=mouse_behaviour_data)
    print("finished reading experimental data.")

    ## obsolete - start
    #fig1 = plt.figure()
    #plt.plot(reward_vec)
    #smooth = 20 # time steps to smooth over
    #plt.plot(np.convolve(reward_vec,np.ones(smooth))/smooth)
    #plt.plot(block_vector)

    #fig2 = plt.figure()
    #plt.plot(cumulative_reward)
    ## obsolete - end
    
    print('Processing simulation data')

    # obtain the mean reward and action given stimulus around O2V transition
    # no need to pass above variables as they are not modified, only analysed
    average_reward_around_o2v_transition, \
        actionscount_to_stimulus_o2v, \
        actionscount_to_stimulus_bytrials_o2v, \
        probability_action_given_stimulus_o2v, \
        probability_action_given_stimulus_bytrials_o2v, \
        context_o2v, mismatch_error_o2v, mismatch_by_perfectswitch_o2v = \
            process_transitions(exp_steps, block_vector_exp_compare,
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

    plot_prob_actions_given_stimuli(probability_action_given_stimulus_o2v,
                                    probability_action_given_stimulus_bytrials_o2v,
                                    mean_probability_action_given_stimulus_o2v,
                                    context_o2v, mismatch_error_o2v,
                                    detailed_plots, abstract_plots,
                                    agent_type=agent_type)

    # obtain the mean reward and action given stimulus around V2O transition
    # no need to pass above variables as they are not modified, only analysed
    average_reward_around_v2o_transition, \
        actionscount_to_stimulus_v2o, \
        actionscount_to_stimulus_bytrials_v2o, \
        probability_action_given_stimulus_v2o, \
        probability_action_given_stimulus_bytrials_v2o, \
        context_v2o, mismatch_error_v2o, mismatch_by_perfectswitch_v2o = \
            process_transitions(exp_steps, block_vector_exp_compare,
                                reward_vector_exp_compare, 
                                stimulus_vector_exp_compare,
                                action_vector_exp_compare,
                                context_record, mismatch_error_record,
                                O2V = False, half_window=half_window)

    plot_prob_actions_given_stimuli(probability_action_given_stimulus_v2o,
                                    probability_action_given_stimulus_bytrials_v2o,
                                    mean_probability_action_given_stimulus_v2o,
                                    context_v2o, mismatch_error_v2o,
                                    detailed_plots, abstract_plots,
                                    agent_type=agent_type,
                                    trans='V2O')

    # params_all is not saved in the .mat file as it is a ragged array with NoneType as well
    if agent_type == 'basic':
        pass
        #print( "(epsilon, alpha, unrewarded_visual_exploration_rate), learning_during_testing)", params_all )
    elif agent_type == 'belief':
        #print( """(belief_switching_rate, context_error_noiseSD_factor, epsilon, unrewarded_visual_exploration_rate,
        #                        exploration_add_factor_for_context_uncertainty, alpha),
        #                 learning_during_testing, context_sampling )""", params_all )
        plot_mismatch_vs_perfectswitch(mismatch_by_perfectswitch_o2v, mismatch_by_perfectswitch_v2o)

    print( 'Root mean squared error = ', 
                rootmeansquarederror(transitions_actionscount_to_stimulus_o2v,
                                transitions_actionscount_to_stimulus_v2o,
                                probability_action_given_stimulus_o2v,
                                probability_action_given_stimulus_v2o,
                                fit_rewarded_stimuli_only, num_params_to_fit) )
    plt.show()

if __name__ == "__main__":

    # chooses one the below to average sim data over seeds
    #seeds = [1,2,3,4,5]
    seeds = [1]

    ## choose one of the below:
    ## use mouse_behaviour_for_neural_data instead mouse_behaviour_data when setting new_data (second last arg) == 1,
    ##  or mouse_behaviour_data_newest when setting new_data == 2, or mouse_behaviour_data when setting new_data == 0..
    ##load_plot_expsimdata('simulation_data/simdata_belief_numparams4_ACCcontrol',seeds,1,mouse_behaviour_for_neural_data) # BeliefRL, ACC on/normal
    load_plot_expsimdata('simulation_data/simdata_belief_numparams4_ACCcontrol',seeds,2,mouse_behaviour_data_newest) # BeliefRL, ACC on/normal
    #load_plot_expsimdata('simulation_data/simdata_belief_numparams4_ACCexp',seeds,0,mouse_behaviour_data) # BeliefRL, ACC off
    #load_plot_expsimdata('simulation_data/simdata_basic_numparams2_ACCcontrol',seeds,1,mouse_behaviour_for_neural_data) # BasicRL, ACC on

