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
        order_rewarded_vis_stim = (2,0)
    else: # V2O transition
        rangeO = range(half_window,2*half_window+1)
        rangeV = range(0,half_window)
        order_rewarded_vis_stim = (0,2)
    p_act_compact[0,rangeV] = probability_action_given_stimulus_bytrials[0,rangeV,1] # visual block, visual stim 0 , lick 1
    p_act_compact[1,rangeV] = probability_action_given_stimulus_bytrials[1,rangeV,1] # visual block, visual stim 1 , lick 1
    p_act_compact[0,rangeO] = probability_action_given_stimulus_bytrials[2,rangeO,1] # olfactory block, visual stim 2 , lick 1
    p_act_compact[1,rangeO] = probability_action_given_stimulus_bytrials[3,rangeO,1] # olfactory block, visual stim 3 , lick 1
    ## no longer plotting visual plots in visual and olfactory blocks as one line
    #for stimulus_index in (0,1):
    #    plot_without_nans(axpaper, xvec, p_act_compact\
    #                                [stimulus_index,:], marker='.',
    #                                color=colors[stimulus_index],
    #                                linestyle='solid',
    #                                label=labels[stimulus_index])
    # plotting the visual plots in the 2 blocks as separate lines
    # join the 'blue' rewarded visual line between the blocks across the transition
    axpaper.plot( [-1,0], [ probability_action_given_stimulus_bytrials[order_rewarded_vis_stim[0],half_window-1,1],
                            probability_action_given_stimulus_bytrials[order_rewarded_vis_stim[1],half_window,1] ],
                           marker='.',color=colors[0],linestyle='solid' )
    for stimulus_index in (0,1,2,3):
        plot_without_nans(axpaper, xvec, probability_action_given_stimulus_bytrials\
                                    [stimulus_index,:,1], marker='.',
                                    color=colors[stimulus_index],
                                    linestyle='solid',
                                    label=labels[stimulus_index])
    # olfactory plots remain the same as they only appear in olfactory block
    for stimulus_index in (4,5):
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
    num_transitions = 70 # number of transitions in new behaviour+neural data
    # Also the p-value becomes negligible, and overflows into a high value for say 3000 samples!
    np.random.seed([1])
    num_transitions_data = np.min((len(mismatch_by_perfectswitch_o2v[0]),len(mismatch_by_perfectswitch_o2v[1]),\
                                    len(mismatch_by_perfectswitch_v2o[0]),len(mismatch_by_perfectswitch_v2o[1])))
    num_transitions_taken = np.min((num_transitions,num_transitions_data))
    print('For significance testing, only taking',num_transitions_taken,'transitions to compare with experiment.')
    idxs = np.random.permutation(np.array(range(num_transitions_data),dtype=int))[:num_transitions_taken]
    mismatch_by_perfectswitch_o2v_0 = np.array(mismatch_by_perfectswitch_o2v[0])[idxs]
    mismatch_by_perfectswitch_o2v_1 = np.array(mismatch_by_perfectswitch_o2v[1])[idxs]
    mismatch_by_perfectswitch_v2o_0 = np.array(mismatch_by_perfectswitch_v2o[0])[idxs]
    mismatch_by_perfectswitch_v2o_1 = np.array(mismatch_by_perfectswitch_v2o[1])[idxs]
    ranksumstat, pvalue = scipy.stats.ranksums(mismatch_by_perfectswitch_o2v_0,mismatch_by_perfectswitch_o2v_1)
    print('Wilcoxon rank sum: O2V ranksumstat, pvalue =', ranksumstat, pvalue)
    ranksumstat, pvalue = scipy.stats.ranksums(mismatch_by_perfectswitch_v2o_0,mismatch_by_perfectswitch_v2o_1)
    print('Wilcoxon rank sum: V2O ranksumstat, pvalue =', ranksumstat, pvalue)
    res = scipy.stats.mannwhitneyu(mismatch_by_perfectswitch_o2v_0,mismatch_by_perfectswitch_o2v_1)
    print('Mann Whitney U: O2V U statistic for imperfect switch, pvalue =', res.statistic, res.pvalue)
    res = scipy.stats.mannwhitneyu(mismatch_by_perfectswitch_v2o_0,mismatch_by_perfectswitch_v2o_1)
    print('Mann Whitney U: V2O U statistic for imperfect switch, pvalue =', res.statistic, res.pvalue)

    fig, ax = plt.subplots(1,2)
    #ax[0].bar( ['wrong switch','correct switch'],
    #            [np.mean(mismatch_by_perfectswitch_o2v[0]),np.mean(mismatch_by_perfectswitch_o2v[1])],
    #            yerr=[np.std(mismatch_by_perfectswitch_o2v[0]),np.std(mismatch_by_perfectswitch_o2v[1])] )
    # showfliers = False only suppresses plotting of outliers, to expand whiskers, set whis to (0,100)
    result = ax[0].boxplot( (mismatch_by_perfectswitch_o2v_1,mismatch_by_perfectswitch_o2v_0), whis=(0,100) )#, showfliers=False )
    # unsure how to find number of outliers. result['fliers'] is a list of two Line2D objects.
    #print('Number of outliers in O2V imperfect & perfect =',len(result['fliers'][0]),len(result['fliers'][1]))
    ax[0].set_xticklabels(['one-shot switch','slower switch'])
    ax[0].set_ylabel('block mismatch signal O2V')
    #ax[1].bar( [' switch','correct switch'],
    #            [np.mean(mismatch_by_perfectswitch_v2o[0]),np.mean(mismatch_by_perfectswitch_v2o[1])],
    #            yerr=[np.std(mismatch_by_perfectswitch_v2o[0]),np.std(mismatch_by_perfectswitch_v2o[1])] )
    # showfliers = False only suppresses plotting of outliers, to expand whiskers, set whis to (0,100)
    result = ax[1].boxplot( (mismatch_by_perfectswitch_v2o_1,mismatch_by_perfectswitch_v2o_0), whis=(0,100) )#, showfliers=False )
    #print('Number of outliers in V2O imperfect & perfect =',len(result['fliers'][0]),len(result['fliers'][1]))
    ax[1].set_xticklabels(['one-shot switch','slower switch'])
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

def load_simdata_seeds(filenamebase,seeds):

    print('Loading simulation data across seeds =',seeds)

    exp_steps = []
    block_vector_exp_compare = []
    reward_vector_exp_compare = []
    stimulus_vector_exp_compare = []
    action_vector_exp_compare = []
    # blank numpy arrays with size of context_record, etc. to allow concatenation
    if 'belief' in filenamebase: # a bit of hard-coding hack to read in belief vs basicRL data
        n_contexts = 2
    else:
        n_contexts = 1
    context_record = np.empty((0,n_contexts))
    mismatch_error_record = np.empty((0,n_contexts))

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

    return (exp_steps, ACC_str, agent_type,
            fit_rewarded_stimuli_only,
            num_params_to_fit,
            block_vector_exp_compare,
            reward_vector_exp_compare,
            stimulus_vector_exp_compare,
            action_vector_exp_compare,
            context_record,
            mismatch_error_record)

def load_plot_simdata(filenamebase, seeds):

    exp_steps, ACC_str, agent_type,\
        fit_rewarded_stimuli_only,\
        num_params_to_fit,\
        block_vector_exp_compare,\
        reward_vector_exp_compare,\
        stimulus_vector_exp_compare,\
        action_vector_exp_compare,\
        context_record,\
        mismatch_error_record = load_simdata_seeds(filenamebase,seeds)
        
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
        plot_mismatch_vs_perfectswitch(mismatch_by_perfectswitch_o2v, mismatch_by_perfectswitch_v2o)

    plt.show()
        

def load_plot_ACConvsoff(filenameACCon, filenameACCoff, first_V2O_visual_is_irrelV2):

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
                        stimulus_vector_exp_compare, action_vector_exp_compare, first_V2O_visual_is_irrelV2)
    switchtimes_V2O = get_switchtimes(False, window, exp_step, block_vector_exp_compare,
                        stimulus_vector_exp_compare, action_vector_exp_compare, first_V2O_visual_is_irrelV2)
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
                        stimulus_vector_exp_compare, action_vector_exp_compare, first_V2O_visual_is_irrelV2)
    switchtimes_V2O = get_switchtimes(False, window, exp_step, block_vector_exp_compare,
                        stimulus_vector_exp_compare, action_vector_exp_compare, first_V2O_visual_is_irrelV2)
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
    seeds = [1,2,3,4,5]  # always use this when new_data==2 in simulations, as 2 seeds use new mod to env, and 3 seeds use old env
    #seeds = [1] # only with new_data in (0,1) during simulation -- obsolete

    ## For Fig 1G and Supplementary Fig 7G comparing fits of BasicRL vs BeliefStateRL:
    ## choose one or more of the below to plot BeliefStateRL or BasicRL:
    # BeliefRL, ACC on/normal, newest data (2 seeds with modded task, 3 seeds without), no learning only exploration during testing
    load_plot_simdata('simulation_data/simdata_belief_numparams4_nolearnwithexplore_ACCcontrol_newdata2',seeds)
    # BasicRL, ACC on/normal, newest data (2 seeds with mod to exp, 3 seeds without), you need learning here during testing!
    #load_plot_simdata('simulation_data/simdata_basic_numparams2_ACCcontrol_newdata2',seeds)

    ## For Fig 6C comparing contex error signal mismatch between one-shot vs slower transitions (see only the mismatch.pdf figure, igonre other figures):
    # BeliefRL, ACC on/normal, params as fit to newest data (2 seeds with modded task, 3 seeds without), no learning only exploration during testing.
    #  But here we use only seed in (3,4,5) not in (1,2) as the former were run on older task while the latter were run on newer task,
    #   whereas Fig 6 uses neural recordings which are only available in dataset 1 on older task.
    #  We can still use the params for the agent fitted on dataset 2 as it subsumes dataset 1, 
    #                      (in any case, params fitted to only dataset 1 are not too different)
    #   but we run the agent only on older task, thus choose seed in (3,4,5).
    #  Only 1 seed is needed as only 70 transitions are considered as in dataset 1.
    #load_plot_simdata('simulation_data/simdata_belief_numparams4_nolearnwithexplore_ACCcontrol_newdata2',seeds=[3])

    ## Suppl. Fig 2D: comparing switching times between blocks for control (ACC on/normal) vs exp (ACC off) with new_data=0:
    # when comparing ACCon vs ACCoff, always simulate with new_data=0 in BeliefHistoryTabularRLSimulate.py as only the older data contains ACC on vs off,
    #  which sets below flag to False (older task is used in older data) there, and so we set here as well:
    first_V2O_visual_is_irrelV2 = False
    #load_plot_ACConvsoff('simulation_data/simdata_belief_numparams4_nolearnwithexplore_ACCcontrol_seed1.mat',
    #                     'simulation_data/simdata_belief_numparams4_nolearnwithexplore_ACCexp_seed1.mat',
    #                     first_V2O_visual_is_irrelV2)
