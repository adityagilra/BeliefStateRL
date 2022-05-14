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
import scipy.stats
import sys

from exp_data_analysis import get_exp_reward_around_transition
from utils import rootmeansquarederror,  process_transitions, get_switchtimes

# number of steps on each side of the transition to consider
half_window = 30

def plot_prob_actions_given_stimuli(probability_action_given_stimulus,
                                        probability_action_given_stimulus_bytrials,
                                        exp_mean_probability_action_given_stimulus,
                                        context, mismatch_error,
                                        detailed_plots, abstract_plots, paper_plots,
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

    if paper_plots:
        figpaper, axpaper = plt.subplots(1,1)
        axpaper.plot([0,0],[0,1],',k',linestyle='--')

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

    if paper_plots:
        # combine plots of visual stimuli coded as 0,1 in visual block and coded as 2,3 in olfactory block into 1 plot of visual stimuli
        p_act_compact = np.zeros((4,half_window*2+1))
        if trans == 'O2V':
            rangeV = range(half_window+1,2*half_window+1)
            rangeO = range(0,half_window+1)
        else: # V2O transition
            rangeO = range(half_window,2*half_window+1)
            rangeV = range(0,half_window)
        p_act_compact[0,rangeV] = probability_action_given_stimulus_bytrials[0,rangeV,1] # visual block, visual stim 0 , lick 1
        p_act_compact[1,rangeV] = probability_action_given_stimulus_bytrials[1,rangeV,1] # visual block, visual stim 1 , lick 1
        p_act_compact[0,rangeO] = probability_action_given_stimulus_bytrials[2,rangeO,1] # olfactory block, visual stim 2 , lick 1
        p_act_compact[1,rangeO] = probability_action_given_stimulus_bytrials[3,rangeO,1] # olfactory block, visual stim 3 , lick 1
        for stimulus_index in range(2):
            plot_without_nans(axpaper, xvec, p_act_compact\
                                        [stimulus_index,:], marker='.',
                                        color=colors[stimulus_index],
                                        linestyle='solid',
                                        label=labels[stimulus_index])
        # olfactory plots remain the same as they only appear in olfactory block
        for stimulus_index in range(4,6):
            plot_without_nans(axpaper, xvec, probability_action_given_stimulus_bytrials\
                                        [stimulus_index,:,1], marker='.',
                                        color=colors[stimulus_index],
                                        linestyle='solid',
                                        label=labels[stimulus_index])
        axpaper.set_xlabel('trials around '+trans+' transition')
        axpaper.set_ylabel('P(lick|stimulus)')
        axpaper.set_xlim([-half_window//2,half_window//2])

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

    if paper_plots:
        figpaper.tight_layout()
        figpaper.savefig('RL_'+agent_type+'_'+trans+'.pdf')
        figpaper.savefig('RL_'+agent_type+'_'+trans+'.svg')

def plot_without_nans(ax,x,y,*args,**kwargs):
    nonans_idxs = ~np.isnan(y)
    ax.plot( np.array(x)[nonans_idxs], np.array(y)[nonans_idxs], *args, **kwargs )

def plot_mismatch_vs_perfectswitch(mismatch_by_perfectswitch_o2v, mismatch_by_perfectswitch_v2o):
    print('Numbers of imperfect and perfect switches for O2V =',
            len(mismatch_by_perfectswitch_o2v[0]),len(mismatch_by_perfectswitch_o2v[1]))
    print('Numbers of imperfect and perfect switches for V2O =',
            len(mismatch_by_perfectswitch_v2o[0]),len(mismatch_by_perfectswitch_v2o[1]))
    ranksumstat, pvalue = scipy.stats.ranksums(mismatch_by_perfectswitch_o2v[0],mismatch_by_perfectswitch_o2v[1])
    print('Wilcoxon rank sum: O2V ranksumstat, pvalue =', ranksumstat, pvalue)
    ranksumstat, pvalue = scipy.stats.ranksums(mismatch_by_perfectswitch_v2o[0],mismatch_by_perfectswitch_v2o[1])
    print('Wilcoxon rank sum: V2O ranksumstat, pvalue =', ranksumstat, pvalue)
    res = scipy.stats.mannwhitneyu(mismatch_by_perfectswitch_o2v[0],mismatch_by_perfectswitch_o2v[1])
    print('Mann Whitney U: O2V U statistic for imperfect switch, pvalue =', res.statistic, res.pvalue)
    res = scipy.stats.mannwhitneyu(mismatch_by_perfectswitch_v2o[0],mismatch_by_perfectswitch_v2o[1])
    print('Mann Whitney U: V2O U statistic for imperfect switch, pvalue =', res.statistic, res.pvalue)
    fig, ax = plt.subplots(1,2)
    #ax[0].bar( ['wrong switch','correct switch'],
    #            [np.mean(mismatch_by_perfectswitch_o2v[0]),np.mean(mismatch_by_perfectswitch_o2v[1])],
    #            yerr=[np.std(mismatch_by_perfectswitch_o2v[0]),np.std(mismatch_by_perfectswitch_o2v[1])] )
    ax[0].boxplot( mismatch_by_perfectswitch_o2v )
    ax[0].set_xticklabels(['imperfect switch','perfect switch'])
    ax[0].set_ylabel('block mismatch signal O2V')
    #ax[1].bar( [' switch','correct switch'],
    #            [np.mean(mismatch_by_perfectswitch_v2o[0]),np.mean(mismatch_by_perfectswitch_v2o[1])],
    #            yerr=[np.std(mismatch_by_perfectswitch_v2o[0]),np.std(mismatch_by_perfectswitch_v2o[1])] )
    ax[1].boxplot( mismatch_by_perfectswitch_v2o )
    ax[1].set_xticklabels(['imperfect switch','perfect switch'])
    ax[1].set_ylabel('block mismatch signal V2O')
    fig.tight_layout()
    fig.savefig('mismatch.pdf')
    fig.savefig('mismatch.svg')

def load_simdata(filename):
    simdata = scipyio.loadmat(filename)
    steps = simdata['steps'][0,0]
    params_all = simdata['params_all'][0]
    exp_step = simdata['exp_step'][0,0]
    fit_rewarded_stimuli_only = simdata['fit_rewarded_stimuli_only'][0]
    agent_type = simdata['agent_type'][0]
    num_params_to_fit = simdata['num_params_to_fit'][0,0]
    ACC_str = simdata['ACC_str'][0]
    seed = simdata['seed'][0,0]
    block_vector_exp_compare = simdata['block_vector_exp_compare'][0]
    reward_vector_exp_compare = simdata['reward_vector_exp_compare'][0]
    stimulus_vector_exp_compare = simdata['stimulus_vector_exp_compare'][0]
    action_vector_exp_compare = simdata['action_vector_exp_compare'][0]
    context_record = simdata['context_record']
    mismatch_error_record = simdata['mismatch_error_record']

    return (steps, params_all, exp_step, fit_rewarded_stimuli_only,
            agent_type, num_params_to_fit, ACC_str, seed,
            block_vector_exp_compare,
            reward_vector_exp_compare,
            stimulus_vector_exp_compare,
            action_vector_exp_compare,
            context_record,
            mismatch_error_record)


def load_plot_simdata(filename):

    steps, params_all, exp_step, fit_rewarded_stimuli_only,
    agent_type, num_params_to_fit, ACC_str, seed,
    block_vector_exp_compare,
    reward_vector_exp_compare,
    stimulus_vector_exp_compare,
    action_vector_exp_compare,
    context_record,
    mismatch_error_record = load_simdata(filename)

    detailed_plots = False
    abstract_plots = False
    paper_plots = True

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
                                    detailed_plots, abstract_plots, paper_plots,
                                    agent_type=agent_type)

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

    plot_prob_actions_given_stimuli(probability_action_given_stimulus_v2o,
                                    probability_action_given_stimulus_bytrials_v2o,
                                    mean_probability_action_given_stimulus_v2o,
                                    context_v2o, mismatch_error_v2o,
                                    detailed_plots, abstract_plots, paper_plots,
                                    agent_type=agent_type,
                                    trans='V2O')

    if agent_type == 'basic':
        print( "(epsilon, alpha, unrewarded_visual_exploration_rate), learning_during_testing)", params_all )
    elif agent_type == 'belief':
        print( """(belief_switching_rate, context_error_noiseSD_factor, epsilon, unrewarded_visual_exploration_rate,
                                exploration_add_factor_for_context_uncertainty, alpha),
                         learning_during_testing, context_sampling )""", params_all )
        plot_mismatch_vs_perfectswitch(mismatch_by_perfectswitch_o2v, mismatch_by_perfectswitch_v2o)

    print( 'Root mean squared error = ', 
                rootmeansquarederror(transitions_actionscount_to_stimulus_o2v,
                                transitions_actionscount_to_stimulus_v2o,
                                probability_action_given_stimulus_o2v,
                                probability_action_given_stimulus_v2o,
                                fit_rewarded_stimuli_only, num_params_to_fit) )

def load_plot_ACConvsoff(filenameACCon, filenameACCoff):

    window = 20
    fig, ax = plt.subplots(1,2,figsize=(10, 5))

    (steps, params_all, exp_step, fit_rewarded_stimuli_only,
    agent_type, num_params_to_fit, ACC_str, seed,
    block_vector_exp_compare,
    reward_vector_exp_compare,
    stimulus_vector_exp_compare,
    action_vector_exp_compare,
    context_record,
    mismatch_error_record) = load_simdata(filenameACCon)

    switchtimes_O2V = get_switchtimes(True, window, exp_step, block_vector_exp_compare,
                        stimulus_vector_exp_compare, action_vector_exp_compare)
    switchtimes_V2O = get_switchtimes(False, window, exp_step, block_vector_exp_compare,
                        stimulus_vector_exp_compare, action_vector_exp_compare)
    xvals = range(window)
    ax[0].bar(xvals,switchtimes_O2V,width=1,color=(1,0,0,0.5))
    ax[1].bar(xvals,switchtimes_V2O,width=1,color=(1,0,0,0.5))

    (steps, params_all, exp_step, fit_rewarded_stimuli_only,
    agent_type, num_params_to_fit, ACC_str, seed,
    block_vector_exp_compare,
    reward_vector_exp_compare,
    stimulus_vector_exp_compare,
    action_vector_exp_compare,
    context_record,
    mismatch_error_record) = load_simdata(filenameACCoff)

    switchtimes_O2V = get_switchtimes(True, window, exp_step, block_vector_exp_compare,
                        stimulus_vector_exp_compare, action_vector_exp_compare)
    switchtimes_V2O = get_switchtimes(False, window, exp_step, block_vector_exp_compare,
                        stimulus_vector_exp_compare, action_vector_exp_compare)
    ax[0].bar(xvals,switchtimes_O2V,width=1,color=(0,0,1,0.5))
    ax[1].bar(xvals,switchtimes_V2O,width=1,color=(0,0,1,0.5))
    
    ax[0].set_ylabel('Fraction of switches')
    ax[0].set_xlabel('Number of trials to lick to visual stimulus')
    ax[1].set_xlabel('Number of trials to ignore visual stimulus')
    ax[0].title.set_text('Switching from olfactory to visual block')
    ax[1].title.set_text('Switching from visual to olfactory block')

    fig.savefig('switchtimes.pdf')
    fig.savefig('switchtimes.svg')
        
if __name__ == "__main__":
    #load_plot_simdata('simulation_data/simdata_belief_numparams4_ACCcontrol_seed1.mat')
    load_plot_ACConvsoff('simulation_data/simdata_belief_numparams4_ACCcontrol_seed1.mat',
                        'simulation_data/simdata_belief_numparams4_ACCexp_seed1.mat')
    plt.show()

