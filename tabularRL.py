import gym
import numpy as np
import gym_tasks
import matplotlib as mpl
import matplotlib.pyplot as plt

# if history, then store current and previous observations as state
# else just store current observation as state
# uncomment one of these
history = 0 # observation is state
#history = 1 # use previous and current observation as state
#history = 5 # n recent observations as state

# policy can be one of these
# uncomment one
#policy = 0 # random
policy = 1 # epsilon-greedy

alpha = 0.1 # TD learning rate
epsilon = 0.1 # exploration rate in epsilon-greedy policy

env = gym.make('visual_olfactory_attention_switch-v0')
observations_length = env.observation_space.n
actions_length = env.action_space.n

if history == 0:
    states_length = observations_length
elif history == 1:
    states_length = observations_length*observations_length

# for history, assume earlier observation was 'end' i.e. (observations_length-1)
previous_observation = observations_length-1
observation = env.reset()
#env.render() # prints on sys.stdout

# IMPORTANT: states with history == 1 are encoded as 
#  (current observation)*observations_length + previous observation
#  or a string of observations for history > 1
if history == 0: state = observation
elif history == 1: state = observation*observations_length + previous_observation
else: state = str(previous_observation)*history + str(observation)

# for the first iteration of the loop,
#  these states/observations are previous ones!
previous_observation = observation
previous_state = state

# not all combinations of observations exist in the task,
#  but we still allocate space for those, they'll just remain 0.
value_vector = np.zeros(states_length)
Q_array = np.zeros((states_length,actions_length))

steps = 1000000
reward_vec = np.zeros(steps)
cumulative_reward = np.zeros(steps)
block_vector = np.zeros(steps)

for t in range(1,steps):
    if policy == 0:
        action = env.action_space.sample()
    elif policy == 1:
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_array[previous_state,:])

    observation, reward, done, info = env.step(action)
    #print(observation, reward, done, info)
    #env.render() # prints on sys.stdout

    reward_vec[t] = reward
    cumulative_reward[t] = cumulative_reward[t-1]+reward
    block_vector[t] = info['block']
    
    # IMPORTANT: states with history are encoded as 
    #  (current observation)*observations_length + previous observation
    if history == 0: state = observation
    elif history == 1: state = observation*observations_length + previous_observation
    
    # values of previous state get updated, not current state
    # don't change values of 'end' state, change only if not end state
    if previous_observation != observations_length-1:
        value_prediction_error = reward + value_vector[state] \
                                        - value_vector[previous_state]
        value_vector[previous_state] += alpha * value_prediction_error
        Q_array[previous_state,action] += \
                alpha * (reward + value_vector[state] \
                                - Q_array[previous_state,action])

    if done:
        print("Finished after {} timesteps".format(t+1))
        # enforcing that end state has zero value and zero Q_array[end,:]
        # not actually needed since I don't change these above as
        #  enforced via `if previous_observation != observations_length-1:`
        if history == 0:
            value_vector[-1] = 0
            Q_array[-1,:] = 0            
        elif history == 1:
            # IMPORTANT: states with history are encoded as 
            #  (current observation)*observations_length + previous observation
            value_vector[-observations_length:] = 0
            Q_array[-observations_length,:] = 0

    previous_observation = observation
    previous_state = state

print(value_vector)
print(Q_array)

#fig1 = plt.figure()
#plt.plot(reward_vec)
#smooth = 20 # time steps to smooth over
#plt.plot(np.convolve(reward_vec,np.ones(smooth))/smooth)
#plt.plot(block_vector)

#fig2 = plt.figure()
#plt.plot(cumulative_reward)

olfactory_to_visual_transitions = np.where(np.diff(block_vector)==-1)[0]
half_window = 50
average_reward_around_o2v_transition = np.zeros(half_window*2)
for transition in olfactory_to_visual_transitions:
    window_min = max((0,transition-half_window))
    window_max = min((transition+half_window,steps))
    window_start = window_min-transition+half_window
    window_end = half_window+window_max-transition
    average_reward_around_o2v_transition[window_start:window_end] \
            += reward_vec[window_min:window_max]
average_reward_around_o2v_transition /= len(olfactory_to_visual_transitions)

fig3 = plt.figure()
plt.plot(average_reward_around_o2v_transition)
plt.plot([half_window,half_window],\
            [min(average_reward_around_o2v_transition),\
                max(average_reward_around_o2v_transition)])

plt.show()
