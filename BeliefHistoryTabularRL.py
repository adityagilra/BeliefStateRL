import numpy as np

class BeliefHistoryTabularRL():
    # state is a concatenated string of previous history observations
    # separator is used to separate observations
    separator = '*'

    def __init__(self, env, policy=1, alpha=0.1, epsilon=0.1, seed=None,
                exploration_decay=False, exploration_decay_time_steps=100000,
                learning_time_steps=100000, recording_time_steps=100000,
                history=0,
                beliefRL=True, belief_switching_rate=0.7, ACC_off_factor=1.,
                exploration_is_modulated_by_context_prediction_error=False,
                exploration_add_factor_for_context_prediction_error=8):
        self.env = env
        self.seed = seed

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
            # ACC_off_factor should be between 0 and 1
            #  it is factor to reduce context prediction error as a proxy for ACC silencing
            #  setting is as 1 implies no ACC silencing
            self.ACC_off_factor = ACC_off_factor
            # whether exploration is modulated by context prediction error,
            #  and by how much additional factor
            self.exploration_is_modulated_by_context_prediction_error = \
                    exploration_is_modulated_by_context_prediction_error
            self.exploration_add_factor_for_context_prediction_error = \
                exploration_add_factor_for_context_prediction_error

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
        self.env.seed(self.seed)
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
            ############ Q-value based epsilon-greedy policy
            ###### Exploration rate
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
                ####### BeliefRL
                if self.exploration_is_modulated_by_context_prediction_error:
                    # using context_prediction_error to guide exploration instead of context_belief_probabilities
                    #  the former detects context change and affects exploration from first trial after transition
                    #  the latter updates in first trial and affects exploration from second trial after transtion
                    #context_uncertainty = 1.0 - np.abs(self.context_belief_probabilities[0]\
                    #                                    -self.context_belief_probabilities[1])
                    context_uncertainty = ( np.abs(self.context_prediction_error[0])\
                                                        + np.abs(self.context_prediction_error[1]) ) / 2.
                    self.exploration_rate *= \
                            (1 + self.exploration_add_factor_for_context_prediction_error*context_uncertainty)
                else:
                    pass
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
                ###### Not context-belief-based, just one context assumed
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

                    #if weight_contexts_by_probabilities:
                    #    self.context_prediction_error = \
                    #        true_context - context_belief_probabilities
                    #else:
                    #    self.context_prediction_error = \
                    #        true_context - context_assumed_now

                    self.context_prediction_error = \
                        self.true_context - self.context_belief_probabilities
                    
                    # ACC encoding of prediction error can be reduced by a factor
                    #  that serves as a proxy for ACC silencing
                    self.context_prediction_error *= self.ACC_off_factor

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
