import numpy as np
import sys

class BeliefHistoryTabularRL():
    # state is a concatenated string of previous history observations
    # separator is used to separate observations
    separator = '*'

    def __init__(self, env, policy=1, alpha=0.1, epsilon=0.1, seed=None,
                exploration_decay=False, exploration_decay_time_steps=100000,
                learning_during_testing=False,
                onlyexploration_nolearning_during_testing=False,
                history=0,
                beliefRL=True, belief_switching_rate=0.7,
                ACC_off_factor_visual=1., ACC_off_factor_odor=1.,
                weak_visual_factor = 1., context_sampling = True,
                context_error_noiseSD_factor = 0.,
                unrewarded_visual_exploration_rate = None,
                exploration_is_modulated_by_context_uncertainty=False,
                exploration_add_factor_for_context_uncertainty=8):
        self.env = env
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

        # policy can be one of these
        #policy = 0 # random
        #policy = 1 # epsilon-greedy
        self.policy = policy
        self.epsilon = epsilon # exploration probability in epsilon-greedy policy
        # to fit the high licking rate to unrewarded visual cue in visual block
        # can set it to None so that this is not used, and usual epsilon is used throughout
        self.unrewarded_visual_exploration_rate = unrewarded_visual_exploration_rate

        self.alpha = alpha # TD learning rate in [0,1]

        # exploration_decay possibly make learning longer
        self.exploration_decay = exploration_decay
        # if exploration_decay is True,
        #  exploration decays to zero over this time scale
        self.exploration_decay_time_steps = exploration_decay_time_steps

        self.learning_during_testing = learning_during_testing
        self.onlyexploration_nolearning_during_testing = onlyexploration_nolearning_during_testing

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
            self.ACC_off_factor_visual = ACC_off_factor_visual
            self.ACC_off_factor_odor = ACC_off_factor_odor
            #self.weak_visual_factor = weak_visual_factor # obsolete, not used now
            # whether exploration is modulated by context prediction error,
            #  and by how much additional factor
            self.exploration_is_modulated_by_context_uncertainty = \
                    exploration_is_modulated_by_context_uncertainty
            self.exploration_add_factor_for_context_uncertainty = \
                exploration_add_factor_for_context_uncertainty
            
            # whether to sample context from context belief probabilities
            #  or just choose the one with largest probability
            self.context_sampling = context_sampling
            # mismatch error (context prediction error) might be noisy in experiments,
            self.context_error_noiseSD_factor = context_error_noiseSD_factor

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
        self.rng = np.random.default_rng(self.seed)
        self.env.set_seed(self.seed)
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

    def decide_action(self, exploration_rate):
        ############# choose an action
        if self.policy == 0:
            # random policy
            # do not use environment's action_space.sample() as it'll use OpenAI Gym's RNG whose seed is not set
            #action = self.env.action_space.sample()
            action = self.rng.choice(range(self.actions_length))
        elif self.policy == 1:
            if self.beliefRL and not self.weight_contexts_by_probabilities:
                # agent chooses a context for action, based on current context belief / probabilities                
                if self.context_sampling:
                    # sample the current context from belief probabilities
                    #  could be a proxy for a neural decision making process
                    #  this sampling noise typically overshadows the noise added to context prediction error
                    context_assumed_now = \
                        self.rng.choice( range(self.n_contexts),\
                                            p=self.context_belief_probabilities )
                else:
                    # select the context with higher belief
                    context_assumed_now = np.argmax(self.context_belief_probabilities)

            if self.unrewarded_visual_exploration_rate is not None:
                # if unrewarded visual cue in visual block, then boost exploration independently
                if self.observation == self.visual_observations[1]:
                    if self.beliefRL:
                        if context_assumed_now == 0:
                            exploration_rate = self.unrewarded_visual_exploration_rate
                    else:
                        exploration_rate = self.unrewarded_visual_exploration_rate

            ############ Q-value based epsilon-greedy policy
            ###### Exploration rate
            # if exploration_rate passed in is zero,
            #  then below modulations have no effect as they are all multiplicative
            if self.exploration_decay:
                # explore with a decreasing probability over exploration_decay_time_steps
                exploration_rate *= \
                        np.clip(1.-self.t/self.exploration_decay_time_steps,0,1)

            if self.beliefRL:
                ####### BeliefRL
                if self.exploration_is_modulated_by_context_uncertainty:
                    # using context_prediction_error to guide exploration instead of context_belief_probabilities
                    #  the former detects context change and affects exploration from first trial after transition
                    #  the latter updates in first trial and affects exploration from second trial after transtion
                    #context_uncertainty = ( np.abs(self.context_prediction_error[0])\
                    #                                    + np.abs(self.context_prediction_error[1]) ) / 2.
                    # modified context error to be computed only at end of trial step, so will be zero when cues are presented,
                    # thus use context_belief_probabilities
                    context_uncertainty = 1.0 - np.abs(self.context_belief_probabilities[0]\
                                                        -self.context_belief_probabilities[1])
                    exploration_rate *= \
                            (1 + self.exploration_add_factor_for_context_uncertainty*context_uncertainty)

                if self.weight_contexts_by_probabilities:
                    # agent weights Q values of contexts by current context probabilities
                    pass # to implement
                    if self.rng.uniform() < exploration_rate:
                        # do not use environment's action_space.sample() as it'll use OpenAI Gym's RNG whose seed is not set
                        #action = self.env.action_space.sample()
                        action = self.rng.choice(range(self.actions_length))
                    else:
                        # to implement
                        pass
                else:
                    # agent chooses a context for action, based on current context belief / probabilities
                    
                    if self.context_sampling:
                        # sample the current context from belief probabilities
                        #  could be a proxy for a neural decision making process
                        #  this sampling noise typically overshadows the noise added to context prediction error
                        context_assumed_now = \
                            self.rng.choice( range(self.n_contexts),\
                                                p=self.context_belief_probabilities )
                    else:
                        # select the context with higher belief
                        context_assumed_now = np.argmax(self.context_belief_probabilities)
                    
                    if self.rng.uniform() < exploration_rate:
                        # do not use environment's action_space.sample() as it'll use OpenAI Gym's RNG whose seed is not set
                        #action = self.env.action_space.sample()
                        action = self.rng.choice(range(self.actions_length))
                    else:
                        action = np.argmax(self.Q_array[self.previous_state][context_assumed_now,:])
            else:
                ###### Not context-belief-based, just one context assumed
                context_assumed_now = 0
                if self.rng.uniform() < exploration_rate:
                    # do not use environment's action_space.sample() as it'll use OpenAI Gym's RNG whose seed is not set
                    #action = self.env.action_space.sample()
                    action = self.rng.choice(range(self.actions_length))
                else:
                    action = np.argmax(self.Q_array[self.previous_state][0,:])
                    
        return action, context_assumed_now

    def train(self,steps):
        # diagnostic print to check if agent uses the params sent in by fit.py (4 params only for BeliefStateRL
        #print('Training agent with params: belief_switching_rate, context_error_noiseSD_factor, epsilon, unrewarded_visual_exploration_rate =',
        #        self.belief_switching_rate, self.context_error_noiseSD_factor, self.epsilon, self.unrewarded_visual_exploration_rate)

        # learning occurs only till `learning_time_steps`
        # if learning_during_testing == True, then learning and exploration occurs till end of simulation `steps`
        # if onlyexploration_nolearning_during_testing == True, then learning till steps//2 and exploration till end
        if self.learning_during_testing and (not self.onlyexploration_nolearning_during_testing):
            learning_time_steps = steps
        else:
            learning_time_steps = steps//2
        # recording starts after this time step
        recording_time_steps = steps//2

        reward_vector = np.zeros(steps)
        cumulative_reward = np.zeros(steps)
        block_vector = np.zeros(steps)
        # lists where we don't keep steps (e.g. end of trial, blanks)
        reward_vector_exp_compare = np.zeros(steps)
        block_vector_exp_compare = np.zeros(steps)
        stimulus_vector_exp_compare = np.zeros(steps)
        action_vector_exp_compare = np.zeros(steps)
        context_record = np.zeros((steps,self.n_contexts))
        mismatch_error_record = np.zeros((steps,self.n_contexts))
        exp_step = 0 # step index as would be saved in experiment
        
        trial_num = 0

        for self.t in range(1,steps):

            # no exploration after learning stops! 
            #  unless overriden by onlyexploration_nolearning_during_testing
            if self.t>learning_time_steps and (not self.onlyexploration_nolearning_during_testing):
                exploration_rate = 0.
            else:
                exploration_rate = self.epsilon
            self.action, self.context_assumed_now = self.decide_action(exploration_rate)

            ############### take the action in the environment / task
            self.observation, self.reward, self.done, self.info = self.env.step(self.action)
            #print(observation, reward, done, info)
            #env.render() # prints on sys.stdout

            #################### calculate context prediction error
            if self.beliefRL:
                ###### context prediction error:
                # we don't need to predict transitions from each state to the next.
                # one way to compute is one step before end of trial:
                #  whether an olfactory cue is expected or not before trial ends
                #  is enough to serve as a context prediction
                #  and thence we compute context prediction error
                # but experimentally the context prediction error is observed at each step
                #  so we calculate a context prediction error at every step

                if self.weight_contexts_by_probabilities:
                    context_effective = self.context_belief_probabilities
                else:
                    #context_used = self.context_assumed_now
                    context_effective = self.context_belief_probabilities

                # Obsolete, hence commented out:
                """
                # Below method of detecting context doesn't match well the experimental p(lick|rewarded visual) in odor block,
                #  or if you get set weak_visual_factor low to match, then it makes the O2V transition slow:
                #  see my OneNote notes of 17-18 Feb 2022
                if self.previous_observation in self.odor_observations:
                    # definitely an olfactory context
                    self.context_prediction_error = np.array((0.,1.)) - context_effective
                elif self.previous_observation in self.visual_observations:
                    self.context_prediction_error = np.array((1.,0.)) - context_effective
                    # this could be irrelevant visual in odor block as well
                    #  thus we reduce the error signal by a factor if context assumed is olfactory
                    if self.context_assumed_now == 1: # odor block
                        self.context_prediction_error *= self.weak_visual_factor
                else:
                    # other cues like end_trial don't produce a context prediction error
                    self.context_prediction_error = np.array((0.,0.))
                """

                # last observation before end state to detect context
                if self.observation == self.end_observation:
                    if self.previous_observation in self.odor_observations:
                        # definitely odor trial
                        context_detected = np.array((0.,1.))
                    elif self.previous_observation in self.visual_observations:
                        # definitely visual trial
                        context_detected = np.array((1.,0.))
                    # since effective context is (0,1) or (1,0), context prediction error is (0,0) or (-1,1) or (1,-1)
                    self.context_prediction_error = context_detected - context_effective
                else:
                    self.context_prediction_error = np.array((0.,0.))

                # context prediction error (mismatch error) is noisy in experiments
                # additive noise
                #self.context_prediction_error += self.context_error_noiseSD*self.rng.normal()
                # multiplicative noise
                self.context_prediction_error *= (1.+self.context_error_noiseSD_factor*self.rng.normal())

                # ACC encoding of prediction error can be reduced by a factor
                #  that serves as a proxy for ACC silencing
                # NOTE: currently noise is also reduced by the same factor,
                #  move the noise addition line from before to after this, to not reduce noise by this factor
                if context_detected[0] == 0: # odor trial
                    self.context_prediction_error *= self.ACC_off_factor_odor
                else: # visual trial
                    self.context_prediction_error *= self.ACC_off_factor_visual

            ############### record reward, stimulus/observation, response/action, context and mismatch error
            # don't record for end of trial observations,
            #  since end of trial 'observation' is not experimentally recorded!
            # also record only after recording_time_steps (usually set same as learning_time_steps)
            # HARDCODED for no-blanks environment: assume blank observation is not present,
            #  so no need to filter out blanks similar to end_observation
            if self.t>recording_time_steps and self.previous_observation!=self.end_observation:
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
                
                if self.beliefRL:
                    # current context belief probabilities
                    context_record[exp_step,:] = self.context_belief_probabilities
                    # context prediction error saved here is calculated above using previous step's observation
                    #  as the action taken in this step was for the observation in the previous step
                    mismatch_error_record[exp_step,:] = self.context_prediction_error

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
            if self.previous_observation != self.end_observation and self.t<=learning_time_steps:
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

            #################### context belief updation
            if self.beliefRL:
                ###### update context belief by context prediction error
                ###### Ideally, I should be doing a Bayesian update with a model,
                ######  but this integration of context_error followed by clipping and normalization seems easier to implement bio-plausibly,
                ######  and will be similar to (or even possibly indistinguishable from) a Bayesian update
                ###### Also, the ACC represents a context prediction error which is easier to incorporate this way instead of a Bayesian model.
                self.context_belief_probabilities += \
                        self.belief_switching_rate * self.context_prediction_error

                # clip to avoid negative probabilities
                # if context_error_noiseSD is small, then probability that both elements of context belief are below 0 is low.
                #  can clip in this case, but might get a nan error in some runs.
                self.context_belief_probabilities = \
                        np.clip(self.context_belief_probabilities,0.,np.inf)
                """
                # avoid negative probabilities, by squeezing asymmetrically between (0,1)
                # do not clip between (0,1) as error+noise could drive both elements of context belief below 0
                # don't squeeze symmetrically around 0, else probabilities remain around 0.5 and are not driven to 0 to 1
                # anything below 0.5 is squeezed between 0.5 and 0, anything above is squeezed between 0.5 and 1:
                switch_sharpness = 1.
                self.context_belief_probabilities = \
                        ( np.tanh(switch_sharpness*(self.context_belief_probabilities-0.5)) +1.) / 2.
                """

                # normalize context belief probabilities after the update
                self.context_belief_probabilities /= \
                        np.sum(self.context_belief_probabilities)
                #print(self.t,self.context_belief_probabilities)

            ################ end of trial processing
            if self.done:
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
                    stimulus_vector_exp_compare, action_vector_exp_compare, context_record, mismatch_error_record
