"""
Context-belief- and/or history-based agent

by Aditya Gilra, Sep-Oct-Nov 2021
"""

import gym
import numpy as np
import gym_tasks
import matplotlib as mpl
import matplotlib.pyplot as plt
from exp_data_analysis import get_exp_reward_around_transition

# reproducible random number generation
seed = 1
np.random.seed(seed)

# number of steps on each side of the transition to consider
half_window = 30

class BeliefHistoryTabularRL():
    # state is a concatenated string of previous history observations
    # separator is used to separate observations
    separator = '*'

    def __init__(self, env, policy=1, alpha=0.1, epsilon=0.1,
                exploration_decay=False, exploration_decay_time_steps=100000,
                learning_time_steps=100000, recording_time_steps=100000,
                history=0,
                beliefRL=True, belief_switching_rate=0.7,
                belief_exploration_add_factor=8):
        self.env = env

        # policy can be one of these
        #policy = 0 # random
        #policy = 1 # epsilon-greedy
        self.policy = policy
        self.epsilon = epsilon # exploration probability in epsilon-greedy policy

        self.alpha = alpha # TD learning rate in [0,1]

        # exploration_decay possibly make learning longer
        self.exploration_decay = exploration_decay
        # if exploration_decay is True,
        #  exploration decays to zero over this time scale
        self.exploration_decay_time_steps = exploration_decay_time_steps

        # learning only occurs till this time step
        self.learning_time_steps = learning_time_steps
        # recording starts after this time step
        self.recording_time_steps = recording_time_steps

        # if history, then store current and previous observations as state
        # else just store current observation as state
        # history can be 0,1,2,3,...
        #history = 0 # observation is state
        #history = 1 # use previous and current observation as state
        #history = 3 # n recent observations as state
        self.history = history

        self.observations_length = env.observation_space.n
        self.actions_length = env.action_space.n
        # end of trial is encoded by (observations_length-1)
        self.end_observation = self.observations_length-1
        # odor observations are encoded by two numbers before end
        self.odor_observations = (self.observations_length-3,self.observations_length-2)
        # visual observations are encoded by two numbers further before
        self.visual_observations = (self.observations_length-5,self.observations_length-4)

        self.beliefRL = beliefRL
        if beliefRL:
            # if beliefRL, then context switching rate is used
            # this rate is applicable after both contextual tasks are learned,
            #  but currently, we don't incorporate context learning,
            #  so context prediction, detection and switching are applied from the start
            self.belief_switching_rate = belief_switching_rate
            self.belief_exploration_add_factor = belief_exploration_add_factor

            # assume agent knows number of contexts before-hand (ideally, should learn!)
            self.n_contexts = 2
            self.context_belief_probabilities = np.zeros(self.n_contexts)
            self.context_belief_probabilities[0] = 1 # at start, agent assumes context 0
            self.context_prediction_error = np.zeros(self.n_contexts)
            self.true_context = np.array((0.5,0.5))

            # choose one of the options below, for assuming a context at each step:
            # if True, agent weights Q values of contexts by current context probabilities
            # if False, agent chooses a context for action, based on current context probabilities
            #self.weight_contexts_by_probabilities = True # NOT IMPLEMENTED CURRENTLY!
            self.weight_contexts_by_probabilities = False
        else:
            self.n_contexts = 1
            
        self.reset()

    def reset(self):
        self.t = 0
        # for history, assume earlier observation was 'end'
        self.previous_observation = self.end_observation
        self.env.seed(seed)
        self.observation = self.env.reset()
        #self.env.render() # prints on sys.stdout

        if self.history == 0:
            self.state = self.observation
        else:
            # if self.history>0, states are encoded as a string of observations
            self.state = (str(self.previous_observation)+self.separator)*self.history \
                                + str(self.observation)
        # debug print
        #print('initial state',self.state)

        # for the first iteration of the loop,
        #  these states/observations are previous ones!
        self.previous_observation = self.observation
        self.previous_state = self.state
        
        # not all combinations of observations exist in the task,
        #  but we still allocate space for those, they'll just remain 0.
        self.value_vector = {self.state: np.zeros(self.n_contexts)}
        self.Q_array = {self.state: np.zeros((self.n_contexts,self.actions_length))}

    def decide_action(self):
        ############# choose an action
        if self.policy == 0:
            # random policy
            action = self.env.action_space.sample()
        elif self.policy == 1:
            # Q-value based epsilon-greedy policy
            # no exploration after learning stops!
            if self.t>self.learning_time_steps:
                self.exploration_rate = 0.
            else:
                self.exploration_rate = self.epsilon
            if self.exploration_decay:
                # explore with a decreasing probability over exploration_decay_time_steps
                self.exploration_rate *= \
                        np.clip(1.-self.t/self.exploration_decay_time_steps,0,1)

            if self.beliefRL:
                # using context_prediction_error instead of context_belief_probabilities
                #  the former detects context change and affects exploration from first trial after transition
                #  the latter updates in first trial and affects exploration from second trial after transtion
                #context_uncertainty = 1.0 - np.abs(self.context_belief_probabilities[0]\
                #                                    -self.context_belief_probabilities[1])
                context_uncertainty = ( np.abs(self.context_prediction_error[0])\
                                                    + np.abs(self.context_prediction_error[1]) ) / 2.
                self.exploration_rate *= (1 + self.belief_exploration_add_factor*context_uncertainty)
                if self.weight_contexts_by_probabilities:
                    # agent weights Q values of contexts by current context probabilities
                    pass # to implement
                    if np.random.uniform() < exploration_rate:
                        action = self.env.action_space.sample()
                    else:
                        # to implement
                        pass
                else:
                    # agent chooses a context for action, based on current context probabilities
                    context_assumed_now = \
                        np.random.choice( range(self.n_contexts),\
                                            p=self.context_belief_probabilities )
                    if np.random.uniform() < self.exploration_rate:
                        action = self.env.action_space.sample()
                    else:
                        action = np.argmax(self.Q_array[self.previous_state][context_assumed_now,:])
            else:
                context_assumed_now = 0
                if np.random.uniform() < self.exploration_rate:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.Q_array[self.previous_state][0,:])
                    
        return action, context_assumed_now

    def train(self,steps):
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

        for self.t in range(1,steps):

            self.action, self.context_assumed_now = self.decide_action()

            ############### take the action in the environment / task
            self.observation, self.reward, self.done, self.info = self.env.step(self.action)
            #print(observation, reward, done, info)
            #env.render() # prints on sys.stdout

            ############### record reward, stimulus/observation and response/action
            # don't record for end of trial observations,
            #  since end of trial 'observation' is not experimentally recorded!
            # also record only after recording_time_steps (usually set same as learning_time_steps)
            # HARDCODED for no-blanks environment: assume blank observation is not present,
            #  so no need to filter out blanks similar to end_observation
            if self.t>self.recording_time_steps and self.previous_observation!=self.end_observation:
                reward_vector_exp_compare[exp_step] = self.reward
                # note that block number changes on the first time step of a new trial,
                block_vector_exp_compare[exp_step] = self.info['block']
                # HARDCODED for no-blanks environment: observation numbers to stimulus numbers mapping
                # observations are 0: 'rewarded' visual, 1: 'unrewarded' visual (in visual or olfactory block)
                #                   2: rewarded olfactory, 3: unrewarded olfactory 
                # stimuli are: 1 rewarded visual, 2 unrewarded visual, 3 irrelevant 'rewarded' visual,
                #               4 irrelevant 'unrewarded', 5 rewarded olfactory, 6 unrewarded olfactory
                # actions are 0 nolick, 1 lick
                if block_vector_exp_compare[exp_step] == 0: # visual block
                    stimulus_vector_exp_compare[exp_step] = \
                            self.previous_observation+1 # 0 and 1 mapped to 1 and 2
                else: # olfactory block
                    stimulus_vector_exp_compare[exp_step] = \
                            self.previous_observation+3 # 0 to 3 mapped to 3 to 6
                # this is the action to the previous_observation
                action_vector_exp_compare[exp_step] = self.action

                # increment the running index of the saved vectors
                exp_step += 1

            reward_vector[self.t] = self.reward
            cumulative_reward[self.t] = cumulative_reward[self.t-1]+self.reward
            block_vector[self.t] = self.info['block']        
            
            ################ observation processing to convert into state
            if self.history == 0:
                self.state = self.observation
            else:
                # states with history are encoded as string of observations
                #  separated by a separator string/character
                states_list = self.previous_state.split(self.separator)
                # drop earliest observation in history and add current observation to state
                self.state = self.separator.join(states_list[1:]) + \
                    self.separator+str(self.observation)
            ################ add in a new state if not previously encountered
            if self.state not in self.value_vector.keys():
                # debug print
                #print('new state encountered', self.state)
                self.value_vector[self.state] = np.zeros(self.n_contexts)
                self.Q_array[self.state] = np.zeros((self.n_contexts,self.actions_length))
            
            ################ update state and Q(state,action) values
            # values of previous state get updated, not current state
            # should not change values of 'end' state 
            #  (for a finite horizon MDP, value of end state = 0),
            # change only if previous observation is not end state
            # also learning happens only for learning_time_steps
            if self.previous_observation != self.end_observation and self.t<=self.learning_time_steps:
                # self.weight_contexts_by_probabilities == False is not implemented
                # below update works for self.beliefRL==False and self.beliefRL==True,
                #  though the update will be very different for the case:
                #  self.beliefRL==True and self.weight_contexts_by_probabilities == True

                # value updation
                value_prediction_error = \
                    self.reward + self.value_vector[self.state][self.context_assumed_now] \
                            - self.value_vector[self.previous_state][self.context_assumed_now]
                self.value_vector[self.previous_state][self.context_assumed_now] += \
                            self.alpha * value_prediction_error
                # Q-value updation
                self.Q_array[self.previous_state][self.context_assumed_now,self.action] += \
                        self.alpha * ( self.reward + self.value_vector[self.state][self.context_assumed_now] \
                                    - self.Q_array[self.previous_state][self.context_assumed_now,self.action] )
                # debug print
                #print(self.t,self.reward,value_prediction_error)

                if self.beliefRL:
                    if self.observation in self.odor_observations:
                        self.true_context = (0.,1.)
                    elif self.observation in self.visual_observations:
                        # to rectify as this is not quite correct,
                        #  as this could be irrelevant visual in odor block
                        self.true_context = (1.,0.)
                    
                    # context prediction error is computed at each observation
                    #  so that it can be used to modulate exploration
                    #  but context_belief_probabilities is
                    #  updated only at the end of a trial,
                    #  so as to update only once per trial
                    self.context_prediction_error = \
                        self.true_context - self.context_belief_probabilities

            ################ end of trial processing, including context belief update
            if self.done:
                if self.beliefRL:
                    # context prediction error and update context_belief_probabilities
                    #  we don't need to predict transitions from each state to the next.
                    #  just whether an olfactory cue is expected or not before trial ends
                    #  is enough to serve as a context prediction
                    #  and thence we compute context prediction error
                    if self.previous_observation in self.odor_observations:
                        self.true_context = (0.,1.)
                    else:
                        self.true_context = (1.,0.)
                    
                    #if weight_contexts_by_probabilities:
                    #    self.context_prediction_error = \
                    #        true_context - context_belief_probabilities
                    #else:
                    #    self.context_prediction_error = \
                    #        true_context - context_assumed_now
                    self.context_prediction_error = \
                        self.true_context - self.context_belief_probabilities

                    # update context belief by context prediction error
                    self.context_belief_probabilities += \
                            self.belief_switching_rate * self.context_prediction_error
                    self.context_belief_probabilities = \
                            np.clip(self.context_belief_probabilities,0.,1.)
                    # normalize context belief probabilities
                    self.context_belief_probabilities /= \
                            np.sum(self.context_belief_probabilities)
                
                trial_num += 1
                # info message
                if trial_num%10000==0:
                    print("Finished trial number {} after {} timesteps".format(trial_num,self.t+1))
                
                ## enforcing that end state has zero value and zero Q_array[end,:]
                ## is NOT needed since I don't update these as enforced above
                ##  via `if previous_observation != observations_length-1:`

            self.previous_observation = self.observation
            self.previous_state = self.state

        # debug print
        #print(self.value_vector)
        #print(self.Q_array)

        return exp_step, block_vector_exp_compare, reward_vector_exp_compare, \
                    stimulus_vector_exp_compare, action_vector_exp_compare

def process_transitions(exp_step, block_vector_exp_compare,
                        reward_vector_exp_compare, 
                        stimulus_vector_exp_compare,
                        action_vector_exp_compare,
                        O2V=True):
    # exp_step, at end of time loop above, equals end index saved in _exp_compare vectors
    # clip vector at exp_step, to avoid detecting a last spurious transtion in block_vector_exp_compare

    # processing after agent simulation has ended
    # calculate mean reward and mean action given stimulus, around transitions
    transitions = \
        np.where(np.diff(block_vector_exp_compare[:exp_step])==(-1 if O2V else 1))[0] + 1
    # note that block number changes on the first time step of a new trial,
    # debug print
    #print(('O2V' if O2V else 'V2O')+" transition at steps ",transitions)
    
    average_reward_around_transition = np.zeros(half_window*2+1)
    actionscount_to_stimulus = np.zeros((6,half_window*2+1,2)) # 6 stimuli, 2 actions
    num_transitions_averaged = 0
    for transition in transitions:
        # take edge effects into account when taking a window around transition
        # i.e. don't go beyond the start and end of saved data if transition is close to an edge
        window_min = max((0,transition-half_window))
        # exp_step, at end of time loop above, equals end index saved in _exp_compare vectors
        window_max = min((transition+half_window+1,exp_step))
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

    # normalize over the actions (last axis i.e. -1) to get probabilities
    # do not add a small amount to denominator to avoid divide by zero!
    # allowing nan so that irrelvant time steps are not plotted
    probability_action_given_stimulus = actionscount_to_stimulus \
                / np.sum(actionscount_to_stimulus,axis=-1)[:,:,np.newaxis] #\
                        #+ np.finfo(np.double).eps )

    return average_reward_around_transition, \
                actionscount_to_stimulus, \
                probability_action_given_stimulus

def plot_prob_actions_given_stimuli(probability_action_given_stimulus,
                                        exp_mean_probability_action_given_stimulus,
                                        detailed_plots, 
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
    figall, axall = plt.subplots(1,1)
    #figall = plt.figure()
    #axall = figall.add_axes([0.1, 0.1, 0.9, 0.9])
    axall.plot([0,0],[0,1],',k',linestyle='--')
    #colors = ['r','g','y','c','b','m']
    colors = ['b','r','b','r','g','y']
    labels = ['+v','-v','/+v','/-v','+o','-o']
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
        axall.plot(xvec,exp_mean_probability_action_given_stimulus\
                            [stimulus_index,:,1], marker='x',
                            color=colors[stimulus_index],
                            label=labels[stimulus_index])
        axall.plot(xvec,probability_action_given_stimulus\
                            [stimulus_index,:,1], marker='.',
                            color=colors[stimulus_index],
                            linestyle='dotted',
                            label=labels[stimulus_index])
        axall.set_xlabel(units+' around '+trans+' transition')
        axall.set_ylabel('P(lick|stimulus)')
        axall.set_xlim([-half_window,half_window])

    if detailed_plots:
        axes[row,col].legend()
        fig.subplots_adjust(wspace=0.5,hspace=0.5)
    axall.legend()
    figall.tight_layout()

def get_env_agent(agent_type='belief'):
    # use one of these environments,
    #  with or without blank state at start of each trial
    # OBSOLETE - with blanks environment won't work,
    #  as I've HARDCODED observations to stimuli encoding
    #env = gym.make('visual_olfactory_attention_switch-v0')
    # HARDCODED these observations to stimuli encoding
    env = gym.make('visual_olfactory_attention_switch_no_blank-v0',
                    lick_without_reward_factor=1.)

    ############# agent instatiation and agent parameters ###########
    # you need around 1,000,000 steps for enough transitions to average over
    # ensure steps is larger than the defaults of the agent for
    #  exploration_decay_time_steps, learning_time_steps,
    #  and recording_time_steps

    # instantiate the RL agent either with history or belief
    # choose one of the below:

    if agent_type=='basic' or agent_type=='history0_nobelief':
        # with history=0 and beliefRL=False, need to keep learning always on!
        #steps = 1000000
        steps = 2000000
        # 2-param fit using brute to minimize mse,
        #  ended without success, due to exceeding max func evals
        #epsilon, alpha = 0.21886517, 0.72834129
        # does not give a sudden peak/lick in /+V stimulus in V2O transition
        #epsilon, alpha = 0.2, 0.2
        # 2-param fit using brute to minimize mse,
        #  fitting nan-s to nan-s, terminated successfully
        epsilon, alpha = 0.21947144, 0.76400787
        agent = BeliefHistoryTabularRL(env,history=0,beliefRL=False,
                                        alpha=alpha,epsilon=epsilon,
                                        learning_time_steps=steps,
                                        recording_time_steps=steps//2)
    elif agent_type=='history1_nobelief':
        # with history=1 and beliefRL=False, no need to keep learning always on!
        #  learning stops after learning_time_steps,
        #  then 1 step correct performance switch!
        # history=1 takes a bit longer to learn than history=2!
        steps = 2000000
        agent = BeliefHistoryTabularRL(env,history=2,beliefRL=False,
                                        learning_time_steps=steps,
                                        recording_time_steps=steps//2)
    elif agent_type=='history2_nobelief':
        # with history=2 and beliefRL=False, no need to keep learning always on!
        #  learning stops after learning_time_steps,
        #  then 1 step correct performance switch!
        steps = 1000000
        agent = BeliefHistoryTabularRL(env,history=2,beliefRL=False,
                                        learning_time_steps=steps//2,
                                        recording_time_steps=steps//2)
    elif agent_type=='belief' or agent_type=='history0_belief':
        # no history, just belief in one of two contexts - two Q tables
        # keeping exploration & learning always on
        #  for noise and belief uncertainty driven exploration
        #steps = 500000
        steps = 2000000
        # obtained by 1-param fit using powell's minimize mse
        #belief_switching_rate = 0.76837728
        # obtained by 3-param fit using powell's minimize mse -- out of bounds - gives nans
        #belief_switching_rate, epsilon, alpha = 1.694975  , 0.48196603, 2.1
        # obtained by 3-param fit using brute minimize mse
        #belief_switching_rate, epsilon, alpha = 0.6, 0.1, 0.105
        # obtained by 3-param fit using brute minimize mse (alpha fixed at 0.1)
        #belief_switching_rate, epsilon, belief_exploration_add_factor, alpha \
        #            = 0.52291667, 0.10208333, 8.146875, 0.1
        # obtained by 3-param fit using brute minimize mse with nan-errors (alpha fixed at 0.1)
        belief_switching_rate, epsilon, belief_exploration_add_factor, alpha \
                    = 0.54162102, 0.09999742, 8.2049604, 0.1
        agent = BeliefHistoryTabularRL(env,history=0,beliefRL=True,
                                        belief_switching_rate=belief_switching_rate,
                                        alpha=alpha, epsilon=epsilon,
                                        belief_exploration_add_factor=belief_exploration_add_factor,
                                        learning_time_steps=steps,
                                        recording_time_steps=steps//2)

    return env, agent, steps

if __name__ == "__main__":

    ############## choose / uncomment one of the agents below! #################
    env, agent, steps = get_env_agent(agent_type='belief')
    #env, agent, steps = get_env_agent(agent_type='basic')
    
    # train the RL agent on the task
    exp_step, block_vector_exp_compare, \
        reward_vector_exp_compare, stimulus_vector_exp_compare, \
            action_vector_exp_compare = \
                agent.train(steps)

    detailed_plots = False

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
        probability_action_given_stimulus_o2v = \
            process_transitions(exp_step, block_vector_exp_compare,
                                reward_vector_exp_compare,
                                stimulus_vector_exp_compare,
                                action_vector_exp_compare,
                                O2V = True)

    if detailed_plots:
        fig3 = plt.figure()
        plt.plot(average_reward_around_o2v_transition)
        plt.plot([half_window,half_window],\
                    [min(average_reward_around_o2v_transition),\
                        max(average_reward_around_o2v_transition)])
        plt.xlabel('time steps around olfactory to visual transition')
        plt.ylabel('average reward on time step')

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

    plot_prob_actions_given_stimuli(probability_action_given_stimulus_o2v,
                                    mean_probability_action_given_stimulus_o2v,
                                        detailed_plots)

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

    plot_prob_actions_given_stimuli(probability_action_given_stimulus_v2o,
                                    mean_probability_action_given_stimulus_v2o,
                                        detailed_plots, trans='V2O')

    plt.show()
