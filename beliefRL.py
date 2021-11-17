"""
Context-belief-based agent

by Aditya Gilra, Sep-Oct 2021
"""

import gym
import numpy as np
import gym_tasks
import matplotlib as mpl
import matplotlib.pyplot as plt

# reproducible random number generation
seed = 1
np.random.seed(seed)

# if history, then store current and previous observations as state
# else just store current observation as state
# uncomment one of these
history = 0 # observation is state
#history = 1 # use previous and current observation as state
#history = 3 # n recent observations as state

# state is a concatenated string of previous history observations
# separator is used to separate observations
separator = '*'

# policy can be one of these
# uncomment one
#policy = 0 # random
policy = 1 # epsilon-greedy


# you need around 1,000,000 steps for enough transitions to average over
steps = 200000

alpha = 0.1 # TD learning rate

epsilon = 0.1 # exploration probability in epsilon-greedy policy
# uncomment one of these, exploration_decay possibly make learning longer
#exploration_decay = True
exploration_decay = False
# if exploration_decay is True,
#  exploration decays to zero over this time scale
exploration_decay_time_steps = 100000

# learning only occurs till this time step
learning_time_steps = exploration_decay_time_steps

#

# context switching rate 
# this rate is applicable after both contextual tasks are learned,
#  but currently, we don't incorporate context learning,
#  so context prediction, detection and switching are applied from the start
belief_switching_rate = 0.7

# uncomment one of these environments,
#  with or without blank state at start of each trial
# OBSOLETE - with blanks environment won't work,
#  as I've HARDCODED observations to stimuli encoding
#env = gym.make('visual_olfactory_attention_switch-v0')
# HARDCODED these observations to stimuli encoding
env = gym.make('visual_olfactory_attention_switch_no_blank-v0')

observations_length = env.observation_space.n
actions_length = env.action_space.n
# end of trial is encoded by (observations_length-1)
end_observation = observations_length-1
# odor observations are encoded by two numbers before end
odor_observations = (observations_length-3,observations_length-2)
# visual observations are encoded by two numbers further before
visual_observations = (observations_length-5,observations_length-4)

# assume agent knows number of contexts before-hand (ideally, should learn!)
n_contexts = 2
context_belief_probabilities = np.zeros(n_contexts)
context_belief_probabilities[0] = 1 # at start, agent assumes context 0

# choose one of the options below, for assuming a context at each step:
# if True, agent weights Q values of contexts by current context probabilities
# if False, agent chooses a context for action, based on current context probabilities
#weight_contexts_by_probabilities = True
weight_contexts_by_probabilities = False

# number of steps on each side of the transition to consider
half_window = 30

def beliefRL(belief_switching_rate):
    # for history, assume earlier observation was 'end'
    previous_observation = end_observation
    env.seed(seed)
    observation = env.reset()
    #env.render() # prints on sys.stdout

    # IMPORTANT: states are encoded as a string of observations
    state = (str(previous_observation)+separator)*history + str(observation)

    # for the first iteration of the loop,
    #  these states/observations are previous ones!
    previous_observation = observation
    previous_state = state
    
    # not all combinations of observations exist in the task,
    #  but we still allocate space for those, they'll just remain 0.
    value_vector = {state: np.zeros(n_contexts)}
    Q_array = {state: np.zeros((n_contexts,actions_length))}

    reward_vector = np.zeros(steps)
    cumulative_reward = np.zeros(steps)
    block_vector = np.zeros(steps)
    # lists where we don't keep steps (e.g. end of trial, blanks)
    reward_vector_exp_compare = np.zeros(steps)
    block_vector_exp_compare = np.zeros(steps)
    stimulus_vector_exp_compare = np.zeros(steps)
    action_vector_exp_compare = np.zeros(steps)
    exp_step = 0 # step index as would be saved in experiment
    
    trial_num = 0

    # needed as I modify this global variable in the function
    global context_belief_probabilities

    for t in range(1,steps):
        ############# choose an action
        if policy == 0:
            # random policy
            action = env.action_space.sample()
        elif policy == 1:
            # Q-value based epsilon-greedy policy
            if exploration_decay:
                # explore with a decreasing probability over exploration_decay_time_steps
                exploration_rate = epsilon*np.clip(1.-t/exploration_decay_time_steps,0,1)
            else:
                exploration_rate = epsilon
            # no exploration after learning stops!
            if t>learning_time_steps:
                exploration_rate = 0.
            if weight_contexts_by_probabilities:
                # agent weights Q values of contexts by current context probabilities
                pass # to implement
                if np.random.uniform() < exploration_rate:
                    action = env.action_space.sample()
                else:
                    # to implement
                    pass
            else:
                # agent chooses a context for action, based on current context probabilities
                context_assumed_now = \
                    np.random.choice( range(n_contexts),\
                                        p=context_belief_probabilities )
                if np.random.uniform() < exploration_rate:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(Q_array[previous_state][context_assumed_now,:])

        ############### take the action in the environment / task
        observation, reward, done, info = env.step(action)
        #print(observation, reward, done, info)
        #env.render() # prints on sys.stdout

        ############### record reward, stimulus/observation and response/action
        # don't keep reward for end of trial steps,
        #  since reward for end of trial 'observation' is not experimentally recorded!
        # also keep rewards only after learning is done!
        # HARDCODED for no-blanks environment: assume blank observation is not present,
        #  so no need to filter out blanks similar to end_observation
        if t>learning_time_steps and previous_observation!=end_observation:
            reward_vector_exp_compare[exp_step] = reward
            # note that block number changes on the first time step of a new trial,
            block_vector_exp_compare[exp_step] = info['block']
            # HARDCODED for no-blanks environment: observation numbers to stimulus numbers mapping
            # stimuli are: 1 rewarded visual, 2 unrewarded visual, 3 irrelevant 'rewarded' visual,
            #               4 irrelevant 'unrewarded', 5 rewarded olfactory, 6 unrewarded olfactory
            # actions are 0 nolick, 1 lick
            if block_vector_exp_compare[exp_step] == 0: # visual block
                stimulus_vector_exp_compare[exp_step] = previous_observation+1 # 0 and 1 mapped to 1 and 2
            else: # olfactory block
                stimulus_vector_exp_compare[exp_step] = previous_observation+3 # 0 to 3 mapped to 3 to 6
            # this is the action to the previous_observation
            action_vector_exp_compare[exp_step] = action

            # increment the running index of the saved vectors
            exp_step += 1

        reward_vector[t] = reward
        cumulative_reward[t] = cumulative_reward[t-1]+reward
        block_vector[t] = info['block']        
        
        ################ observation processing
        # IMPORTANT: states with history are encoded as string of observations
        #  separated by a separator string/character
        states_list = previous_state.split(separator)
        # drop earliest observation in history and add current observation to state
        state = separator.join(states_list[1:])+separator+str(observation)
        ################ add in a new state if not previously encountered
        if state not in value_vector.keys():
            value_vector[state] = np.zeros(n_contexts)
            Q_array[state] = np.zeros((n_contexts,actions_length))
        
        ################ update state and Q(state,action) values
        # values of previous state get updated, not current state
        # should not change values of 'end' state 
        #  (for a finite horizon MDP, value of end state = 0),
        # change only if previous observation is not end state
        # also learning happens only for learning_time_steps
        if previous_observation != end_observation and t<=learning_time_steps:
            if weight_contexts_by_probabilities:
                pass # to implement
            else:
                # value updation
                value_prediction_error = \
                    reward + value_vector[state][context_assumed_now] \
                            - value_vector[previous_state][context_assumed_now]
                value_vector[previous_state][context_assumed_now] += \
                            alpha * value_prediction_error
                # Q-value updation
                Q_array[previous_state][context_assumed_now,action] += \
                        alpha * ( reward + value_vector[state][context_assumed_now] \
                                    - Q_array[previous_state][context_assumed_now,action] )

        ################ end of trial processing, including context belief update
        if done:
            # context prediction error and update context_belief_probabilities
            #  we don't need to predict transitions from each state to the next.
            #  just whether an olfactory cue is expected or not before trial ends
            #  is enough to serve as a context prediction
            #  and thence we compute context prediction error
            if previous_observation in odor_observations: true_context = (0.,1.)
            else: true_context = (1.,0.)
            
            #if weight_contexts_by_probabilities:
            #    context_prediction_error = \
            #        true_context - context_belief_probabilities
            #else:
            #    context_prediction_error = \
            #        true_context - context_assumed_now
            context_prediction_error = \
                true_context - context_belief_probabilities

            # update context belief by context prediction error
            context_belief_probabilities += \
                belief_switching_rate * context_prediction_error
            context_belief_probabilities = np.clip(context_belief_probabilities,0.,1.)
            # normalize context belief probabilities
            context_belief_probabilities /= np.sum(context_belief_probabilities)
            
            trial_num += 1
            # info message
            if trial_num%10000==0:
                print("Finished trial number {} after {} timesteps".format(trial_num,t+1))
            
            ## enforcing that end state has zero value and zero Q_array[end,:]
            ## is NOT needed since I don't update these as enforced above
            ##  via `if previous_observation != observations_length-1:`

        previous_observation = observation
        previous_state = state

    #print(value_vector)
    #print(Q_array)

    return exp_step, block_vector_exp_compare, reward_vector_exp_compare, \
                stimulus_vector_exp_compare, action_vector_exp_compare

def process_transitions(O2V=True):
    # assume that variables:
    #  exp_step, block_vector_exp_compare,
    #  reward_vector_exp_compare, action_vector_exp_compare
    # are available in the global workspace
    # since they are not modified here, only analysed, no need to pass them in!

    # exp_step, at end of time loop above, equals end index saved in _exp_compare vectors
    # clip vector at exp_step, to avoid detecting a last spurious transtion in block_vector_exp_compare

    # processing after agent simulation has ended
    # calculate mean reward and mean action given stimulus, around transitions
    transitions = \
        np.where(np.diff(block_vector_exp_compare[:exp_step])==(-1 if O2V else 1))[0] + 1
    # note that block number changes on the first time step of a new trial,
    print(('O2V' if O2V else 'V2O')+" transition at steps ",transitions)
    average_reward_around_transition = np.zeros(half_window*2)
    actionscount_to_stimulus = np.zeros((6,half_window*2,2)) # 6 stimuli, 2 actions
    num_transitions_averaged = 0
    for transition in transitions:
        # take edge effects into account when taking a window around transition
        # i.e. don't go beyond the end and start of saved data if transition is close to an edge
        window_min = max((0,transition-half_window))
        # exp_step, at end of time loop above, equals end index saved in _exp_compare vectors
        window_max = min((transition+half_window,exp_step))
        window_start = window_min-(transition-half_window)
        window_end = window_max-(transition-half_window)
        average_reward_around_transition[window_start:window_end] \
                += reward_vector_exp_compare[window_min:window_max]

        # debug print
        #print(('O2V' if O2V else 'V2O'), transition,
        #        stimulus_vector_exp_compare[transition-5:transition+5],
        #        action_vector_exp_compare[transition-5:transition+5])

        ######### actions given stimuli around transition
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

        num_transitions_averaged += 1
    average_reward_around_transition /= num_transitions_averaged

    return average_reward_around_transition, actionscount_to_stimulus

def plot_prob_actions_given_stimuli(actionscount_to_stimulus, units='steps', trans='O2V'):
    # normalize over the actions (last axis i.e. -1) to get probabilities
    # do not add a small amount to denominator to avoid divide by zero!
    # allowing nan so that irrelvant time steps are not plotted
    probability_action_given_stimulus = actionscount_to_stimulus \
                / np.sum(actionscount_to_stimulus,axis=-1)[:,:,np.newaxis] #\
                        #+ np.finfo(np.double).eps )

    # debug print
    #for stimulus_index in range(6):
    #    print(stimulus_index+1,probability_action_given_stimulus[stimulus_index,half_window-5:half_window+5,1])
    #    print(stimulus_index+1,actionscount_to_stimulus[stimulus_index,half_window-5:half_window+5,1],
    #            np.sum(actionscount_to_stimulus[stimulus_index,half_window-5:half_window+5,:],axis=-1))

    xvec = range(-half_window,half_window)
    fig, axes = plt.subplots(2,3)
    figall, axall = plt.subplots(1,1)
    #figall = plt.figure()
    #axall = figall.add_axes([0.1, 0.1, 0.9, 0.9])
    axall.plot([0,0],[0,1],',k',linestyle='--')
    colors = ['r','g','y','c','b','m']
    labels = ['+v','-v','/+v','/-v','+o','-o']
    for stimulus_index in range(6):
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
        axall.plot(xvec,probability_action_given_stimulus\
                            [stimulus_index,:,1], marker='.',
                            color=colors[stimulus_index],
                            label=labels[stimulus_index])
        axall.set_xlabel(units+' around '+trans+' transition')
        axall.set_ylabel('P(lick|stimulus)')
        axall.set_xlim([-half_window,half_window])

    axes[row,col].legend()
    fig.subplots_adjust(wspace=0.5,hspace=0.5)
    axall.legend()
    figall.tight_layout()

if __name__ == "__main__":
    # simulate the RL agent on the task
    exp_step, block_vector_exp_compare, \
        reward_vector_exp_compare, stimulus_vector_exp_compare, \
            action_vector_exp_compare = \
                beliefRL(belief_switching_rate)
    # obtain the mean reward and action given stimulus around O2V transition
    # no need to pass above variables as they are not modified, only analysed
    average_reward_around_o2v_transition, actionscount_to_stimulus = \
        process_transitions(O2V = True)

    ## obsolete - start
    #fig1 = plt.figure()
    #plt.plot(reward_vec)
    #smooth = 20 # time steps to smooth over
    #plt.plot(np.convolve(reward_vec,np.ones(smooth))/smooth)
    #plt.plot(block_vector)

    #fig2 = plt.figure()
    #plt.plot(cumulative_reward)
    ## obsolete - end

    fig3 = plt.figure()
    plt.plot(average_reward_around_o2v_transition)
    plt.plot([half_window,half_window],\
                [min(average_reward_around_o2v_transition),\
                    max(average_reward_around_o2v_transition)])
    plt.xlabel('time steps around olfactory to visual transition')
    plt.ylabel('average reward on time step')

    plot_prob_actions_given_stimuli(actionscount_to_stimulus)

    # obtain the mean reward and action given stimulus around V2O transition
    # no need to pass above variables as they are not modified, only analysed
    average_reward_around_o2v_transition, actionscount_to_stimulus = \
        process_transitions(O2V = False)

    plot_prob_actions_given_stimuli(actionscount_to_stimulus, trans='V2O')

    plt.show()
