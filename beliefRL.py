import gym
import numpy as np
import gym_tasks
import matplotlib as mpl
import matplotlib.pyplot as plt

# reproducible random number generation
np.random.seed(1)

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
steps = 500000

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

# context switching rate 
# this rate is applicable after both contextual tasks are learned,
#  but currently, we don't incorporate context learning,
#  so context prediction, detection and switching are applied from the start
belief_switching_rate = 0.2

# uncomment one of these environments,
#  with or without blank state at start of each trial
#env = gym.make('visual_olfactory_attention_switch-v0')
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
half_window = 22

def beliefRL(belief_switching_rate):
    # for history, assume earlier observation was 'end'
    previous_observation = end_observation
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
    reward_vector_exp_compare = []
    block_vector_exp_compare = []
    
    trial_num = 0

    # needed as I modify this global variable in the function
    global context_belief_probabilities

    for t in range(1,steps):
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

        observation, reward, done, info = env.step(action)
        #print(observation, reward, done, info)
        #env.render() # prints on sys.stdout

        # don't keep reward for end of trial steps,
        #  since reward for end of trial 'observation' is not experimentally recorded!
        # also keep rewards only after learning is done!
        if t>learning_time_steps and previous_observation!=end_observation:
            reward_vector_exp_compare.append(reward)
            block_vector_exp_compare.append(info['block'])

        reward_vector[t] = reward
        cumulative_reward[t] = cumulative_reward[t-1]+reward
        block_vector[t] = info['block']
        
        # IMPORTANT: states with history are encoded as string of observations
        #  separated by a separator string/character
        states_list = previous_state.split(separator)
        # drop earliest observation in history and add current observation to state
        state = separator.join(states_list[1:])+separator+str(observation)
        if state not in value_vector.keys():
            value_vector[state] = np.zeros(n_contexts)
            Q_array[state] = np.zeros((n_contexts,actions_length))
        
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
            ## is NOT needed since I don't update these above as enforced
            ##  via `if previous_observation != observations_length-1:`

        previous_observation = observation
        previous_state = state

    # calculate mean reward around transitions
    olfactory_to_visual_transitions = \
        np.where(np.diff(block_vector_exp_compare)==-1)[0]
    print("o2v transition at steps ",olfactory_to_visual_transitions)
    average_reward_around_o2v_transition = np.zeros(half_window*2)
    num_transitions_averaged = 0
    for transition in olfactory_to_visual_transitions:
        window_min = max((0,transition-half_window))
        window_max = min((transition+half_window,len(block_vector_exp_compare)))
        window_start = window_min-transition+half_window
        window_end = half_window+window_max-transition
        average_reward_around_o2v_transition[window_start:window_end] \
                += reward_vector_exp_compare[window_min:window_max]
        num_transitions_averaged += 1
    average_reward_around_o2v_transition /= num_transitions_averaged

    #print(value_vector)
    #print(Q_array)

    return average_reward_around_o2v_transition

if __name__ == "__main__":
    # call the task function and obtain the mean reward around transition
    average_reward_around_o2v_transition = beliefRL(belief_switching_rate)

    #fig1 = plt.figure()
    #plt.plot(reward_vec)
    #smooth = 20 # time steps to smooth over
    #plt.plot(np.convolve(reward_vec,np.ones(smooth))/smooth)
    #plt.plot(block_vector)

    #fig2 = plt.figure()
    #plt.plot(cumulative_reward)


    fig3 = plt.figure()
    plt.plot(average_reward_around_o2v_transition)
    plt.plot([half_window,half_window],\
                [min(average_reward_around_o2v_transition),\
                    max(average_reward_around_o2v_transition)])
    plt.xlabel('time steps around olfactory to visual transition')
    plt.ylabel('average reward on time step')

    plt.show()
