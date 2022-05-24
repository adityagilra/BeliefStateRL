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

from utils import process_transitions, get_switchtimes

# number of steps on each side of the transition to consider
half_window = 30

def plot_sim_prob_actions_given_stimuli(probability_action_given_stimulus,
                                        probability_action_given_stimulus_bytrials,
                                        context, mismatch_error,
                                        agent_type='',
                                        units='steps', trans='O2V'):    

    xvec = range(-half_window,half_window+1)
    
    #colors = ['r','g','y','c','b','m']
    colors = ['b','r','b','r','g','y']
    labels = ['+v','-v','/+v','/-v','+o','-o']

    figpaper, axpaper = plt.subplots(1,1)
    axpaper.plot([0,0],[0,1],',k',linestyle='--')

    # combine plots of visual stimuli coded as 0,1 in visual block
    #  and coded as 2,3 in olfactory block into 1 plot of visual stimuli
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
    ## no longer plotting visual plots in visual and olfactory blocks as one line
    #for stimulus_index in range(2):
    #    plot_without_nans(axpaper, xvec, p_act_compact\
    #                                [stimulus_index,:], marker='.',
    #                                color=colors[stimulus_index],
    #                                linestyle='solid',
    #                                label=labels[stimulus_index])
    # plotting the visual plots in the 2 blocks as separate lines
    for stimulus_index in range(4):
        plot_without_nans(axpaper, xvec, probability_action_given_stimulus_bytrials\
                                    [stimulus_index,:,1], marker='.',
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
    #params_all = simdata['params_all'][0]
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

    return (steps, exp_step, fit_rewarded_stimuli_only,
            agent_type, num_params_to_fit, ACC_str, seed,
            block_vector_exp_compare,
            reward_vector_exp_compare,
            stimulus_vector_exp_compare,
            action_vector_exp_compare,
            context_record,
            mismatch_error_record)


def load_plot_simdata(filenamebase, seeds):

    print('Loading simulation data across seeds =',seeds)

    exp_steps = []
    block_vector_exp_compare = []
    reward_vector_exp_compare = []
    stimulus_vector_exp_compare = []
    action_vector_exp_compare = []
    # blank numpy arrays with size of context_record, etc. to allow concatenation
    context_record = np.empty((0,2))
    mismatch_error_record = np.empty((0,2))

    exp_step_count = 0
    for seed in seeds:
        filename = filenamebase + '_seed' + str(seed) + '.mat'

        (steps, exp_step, fit_rewarded_stimuli_only,
        agent_type, num_params_to_fit, ACC_str, seed,
        block_vector_exp_compare_seed,
        reward_vector_exp_compare_seed,
        stimulus_vector_exp_compare_seed,
        action_vector_exp_compare_seed,
        context_record_seed,
        mismatch_error_record_seed) = load_simdata(filename)
        
        exp_step_count += exp_step
        exp_steps.append(exp_step_count)
        block_vector_exp_compare = np.concatenate((block_vector_exp_compare,block_vector_exp_compare_seed[:exp_step]),axis=0)
        reward_vector_exp_compare = np.concatenate((reward_vector_exp_compare,reward_vector_exp_compare_seed[:exp_step]),axis=0)
        stimulus_vector_exp_compare = np.concatenate((stimulus_vector_exp_compare,stimulus_vector_exp_compare_seed[:exp_step]),axis=0)
        action_vector_exp_compare = np.concatenate((action_vector_exp_compare,action_vector_exp_compare_seed[:exp_step]),axis=0)
        context_record = np.concatenate((context_record,context_record_seed[:exp_step]),axis=0)
        mismatch_error_record = np.concatenate((mismatch_error_record,mismatch_error_record_seed[:exp_step]),axis=0)
        
    print('Processing simulation data')
    
    # length of valid elements in the data arrays
    exp_step = block_vector_exp_compare.shape[0]

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

    print('Plotting simulation data')

    plot_sim_prob_actions_given_stimuli(probability_action_given_stimulus_o2v,
                                    probability_action_given_stimulus_bytrials_o2v,
                                    context_o2v, mismatch_error_o2v,
                                    agent_type=agent_type)

    plot_sim_prob_actions_given_stimuli(probability_action_given_stimulus_v2o,
                                    probability_action_given_stimulus_bytrials_v2o,
                                    context_v2o, mismatch_error_v2o,
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
        #plot_mismatch_vs_perfectswitch(mismatch_by_perfectswitch_o2v, mismatch_by_perfectswitch_v2o)
        pass

    plt.show()
        

def load_plot_ACConvsoff(filenameACCon, filenameACCoff):

    window = 20
    fig, ax = plt.subplots(1,2,figsize=(10, 5))

    (steps, exp_step, fit_rewarded_stimuli_only,
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

    (steps, exp_step, fit_rewarded_stimuli_only,
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
    
    plt.show()
        
if __name__ == "__main__":

    # chooses one the below to average sim data over seeds
    seeds = [1,2,3,4,5]
    #seeds = [1]

    ## choose one or more of the below:
    load_plot_simdata('simulation_data/simdata_belief_numparams4_ACCcontrol',seeds) # BeliefRL, ACC on/normal
    #load_plot_simdata('simulation_data/simdata_belief_numparams4_ACCexp',seeds) # BeliefRL, ACC off
    #load_plot_simdata('simulation_data/simdata_basic_numparams2_ACCcontrol.mat',seeds) # BasicRL, ACC on
    ## compare switching times between blocks for control (ACC on/normal) vs exp (ACC off)
    #load_plot_ACConvsoff('simulation_data/simdata_belief_numparams4_ACCcontrol_seed1.mat',
    #                    'simulation_data/simdata_belief_numparams4_ACCexp_seed1.mat')

