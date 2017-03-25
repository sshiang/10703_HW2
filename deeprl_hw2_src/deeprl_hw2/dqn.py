"""Main DQN agent."""

import numpy as np
from copy import deepcopy
import keras.backend as K

from keras.layers import Lambda, Input
from keras.models import Model

from deeprl_hw2 import utils
from deeprl_hw2.objectives import *

import ipdb

class DQNAgent: 
    """Class implementing DQN.

    This is a basic outline of the functions/parameters you will need
    in order to implement the DQNAgnet. This is just to get you
    started. You may need to tweak the parameters, add new ones, etc.

    Feel free to change the functions and funciton parameters that the
    class provides.

    We have provided docstrings to go along with our suggested API.

    Parameters
    ----------
    q_network: keras.models.Model
      Your Q-network model.
    preprocessor: deeprl_hw2.core.Preprocessor
      The preprocessor class. See the associated classes for more
      details.
    memory: deeprl_hw2.core.Memory
      Your replay memory.
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    batch_size: int
      How many samples in each minibatch.
    """
    def __init__(self,
                 num_actions,
                 q_network,
                 preprocessor,
                 memory,
                 policy,
                 gamma,
                 target_update_freq,
                 warmup, #num_burn_in,
                 train_freq,
                 batch_size,
                 model_name, 
                 save_path):

        self.model = q_network
        self.preprocessor = preprocessor
        self.memory = memory
        self.policy = policy
        self.gamma = gamma
        self.tau = 1. / target_update_freq
        self.warmup = warmup
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.nb_actions = num_actions

        self.step = 0
        self.rand_policy = lambda: np.random.randint(0, num_actions)
        self.is_training = True
        self.model_name = model_name


	print("model name: " + self.model_name)

        self.compiled = False
        self.soft_update = True
        self.save_path = save_path


    def compile(self, optimizer, loss_func):
        """Setup all of the TF graph variables/ops.

        This is inspired by the compile method on the
        keras.models.Model class.

        This is a good place to create the target network, setup your
        loss function and any placeholders you might need.
        
        You should use the mean_huber_loss function as your
        loss_function. You can also experiment with MSE and other
        losses.

        The optimizer can be whatever class you want. We used the
        keras.optimizers.Optimizer class. Specifically the Adam
        optimizer.
        """
        
        self.compiled = False
        self.target_model = utils.clone_model(self.model)
        self.target_model.compile(optimizer='sgd', loss='mse')
        self.model.compile(optimizer='sgd', loss='mse')

        # build loss tensor
        def mask_loss(args):
            y_true, y_pred, mask = args
            if loss_func == 'huber_loss':
                loss = huber_loss(y_true, y_pred) # FIXME TODO: check max_grad in original paper
            else:
                raise RuntimeError('undefined loss_func:{}'.format(loss_func))
            loss *= mask
            return K.sum(loss, axis=-1)

        y_pred = self.model.output
        y_true = Input(name='y_true', shape=(self.nb_actions,))
        mask = Input(name='mask', shape=(self.nb_actions,))
        loss_out = Lambda(mask_loss, output_shape=(1,), name='loss')([y_true, y_pred, mask])
        
        # build optimizer
        if self.soft_update:
            updates = utils.get_soft_target_model_updates(self.target_model, self.model, self.tau)
        else:
            updates = utils.get_hard_target_model_updates(self.target_model, self.model)
        optimizer = utils.AdditionalUpdatesOptimizer(optimizer, updates)

        # build metrics TODO
        metrics = lambda y_true, y_pred: K.mean(K.max(y_pred, axis=-1))

        # build trainable model
        trainable_model = Model(input=[self.model.input, y_true, mask], output=[loss_out, y_pred])
        assert len(trainable_model.output_names) == 2
        combined_metrics = {trainable_model.output_names[1]: metrics}
        losses = [
            lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
            lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
        ]
        trainable_model.compile(optimizer=optimizer, loss=losses, metrics=combined_metrics)
        self.trainable_model = trainable_model
        # debug()

        # finish
        self.compiled = True

    def calc_q_values(self, states, model):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        
        if states.ndim < 4:
            states = states[None,...]
        q_values = model.predict_on_batch(states)
	assert q_values.shape == (len(states), self.nb_actions)
        return q_values


    def select_action(self, state, **kwargs):
        """Select the action based on the current state.

        You will probably want to vary your behavior here based on
        which stage of training your in. For example, if you're still
        collecting random samples you might want to use a
        UniformRandomPolicy.

        If you're testing, you might want to use a GreedyEpsilonPolicy
        with a low epsilon.

        If you're training, you might want to use the
        LinearDecayGreedyEpsilonPolicy.

        This would also be a good place to call
        process_state_for_network in your preprocessor.

        Returns
        --------
        selected action
        """
        process_state = self.preprocessor.process_state_for_network(state)
        if self.step < self.warmup:
            action = self.rand_policy()
        else:
            meta = {
                'is_training': self.is_training
            }
            q_values = self.calc_q_values(process_state, self.model)
            action = self.policy.select_action(q_values, **meta)

        return action


    def update_policy(self):
        """Update your policy.

        Behavior may differ based on what stage of training your
        in. If you're in training mode then you should check if you
        should update your network parameters based on the current
        step and the value you set for train_freq.

        Inside, you'll want to sample a minibatch, calculate the
        target values, update your network, and then update your
        target values.

        You might want to return the loss and other metrics as an
        output. They can help you monitor how training is going.
        """
        
        assert self.is_training


        # prepare batch
        batch = self.memory.sample(self.batch_size, self.model_name)
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = \
            self.preprocessor.process_batch(batch)

        # calculate q
	if self.model_name == 'ddqn' : #use_ddqn:
            """ q^{DDQN}_i = Q^{'}(s_{i+1},argmax_a Q(s_{i+1},a))"""
            
            # random switch online_model and target_model? FIXME
            m1, m2 = (self.model, self.target_model) if np.random.rand() < 0.5 else (self.target_model,self.model)
            
            # cal actions based on online_model(m1)
            online_q = self.calc_q_values(next_state_batch, m1)
            assert online_q.shape == (self.batch_size, self.nb_actions)
            actions = np.argmax(online_q, axis=1)
            
            # cal target q based on target_model(m2)
            target_q = self.calc_q_values(next_state_batch, m2)
            assert target_q.shape == (self.batch_size, self.nb_actions)
            
            # cal q_batch
            q_batch  = target_q[range(self.batch_size), actions]
        else:
            """ q^{DQN}_i = max_a Q^{'}(s_{i+1},a) """

            # cal target q based on target_model
            target_q = self.calc_q_values(next_state_batch, self.target_model)
            assert target_q.shape == (self.batch_size, self.nb_actions)

            # cal q_batch
            q_batch = np.max(target_q,axis=1).flatten()
        assert q_batch.shape == (self.batch_size,)

        # calculate target
        """ y_i = r_i + \gamma * q_i """
        targets = reward_batch + terminal_batch*self.gamma*q_batch
        assert targets.shape == (self.batch_size,)

        # prepare for q-update
        targets_batch = np.zeros((self.batch_size, self.nb_actions))
        targets_batch[range(self.batch_size), action_batch] = targets
        targets_batch = targets_batch.astype('float32')
        
        masks_batch = np.zeros((self.batch_size, self.nb_actions))
        masks_batch[range(self.batch_size), action_batch] = 1.
        masks_batch = masks_batch.astype('float32')
        
        # q-update
        metrics = self.trainable_model.train_on_batch(
            [state_batch, targets_batch, masks_batch],
            [np.zeros((self.batch_size,)), targets_batch]
        )

        return metrics


    def fit(self, env, num_iterations, validate_steps, 
        max_episode_length=None, validate_episodes=20, debug=False):
        """Fit your model to the provided environment.

        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.

        You should probably also periodically save your network
        weights and any other useful info.

        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.

        Parameters
        ----------
        env: gym.Env
          This is your Atari environment. You should wrap the
          environment using the wrap_atari_env function in the
          utils.py
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        validate_steps: int
          How many steps to run a validation test
        validate_episodes: int
          How many episodes to run for a single validation test
        """
        assert self.compiled

        self.is_training = True
        self.step = 0
        episode = 0
        observation = None
        validate_rewards = np.array([]).reshape(validate_episodes,0)
        while self.step < num_iterations:

            # reset if it is the start of episode
            if observation is None:
                self.reset()
                observation = deepcopy(env.reset())
                episode_steps = 0
                episode_reward = 0.

                # TODO random start (keras-rl has this option)
                
            assert (observation is not None and episode_steps is not None and episode_reward is not None)

            # basic operation, action ,reward, blablabla ...
            action = self.select_action(observation)
            observation2, r, done, info = env.step(action)
            reward = self.preprocessor.process_reward(r)
            episode_reward += reward

            if max_episode_length and episode_steps >= max_episode_length -1:
                done = True

            # add replay memory and update policy            
            self.memory.append(
                self.preprocessor.process_state_for_memory(observation), 
                action, reward, done
            )
            if self.step > self.warmup and self.step % self.train_freq == 0:
                self.update_policy()

            # evaluate
            if validate_steps > 0 and self.step % validate_steps == 0:
                validate_reward = self.evaluate(env, validate_episodes, restore=True, debug=debug)
                validate_rewards = np.hstack([validate_rewards, validate_reward])
                utils.plot_reward(
                    '{}/validate_reward'.format(self.save_path),
                    validate_steps, self.step, validate_rewards,
                )
                if debug:
                    utils.prYellow('[Evaluate] Step_{:07d}: mean_reward:{}'.format(
                        self.step, np.mean(validate_reward)
                    ))

            if self.step % (num_iterations/3) == 0:
                self.save_weights(
                    '{}/weights-{}.h5f'.format(self.save_path, self.step),
                    overwrite=True,
                )

            # update 
            episode_steps += 1
            self.step += 1

            observation = deepcopy(observation2)
            if done: # end of episode
                if debug: utils.prGreen('#{}: episode_reward:{} steps:{}'.format(episode,episode_reward,self.step))
                # # We are in a terminal state but the agent hasn't yet seen it. We therefore
                # # perform one more forward-backward call and simply ignore the action before
                # # resetting the environment. We need to pass in `terminal=False` here since
                # # the *next* state, that is the state of the newly reset environment, is
                # # always non-terminal by convention.
                self.memory.append(
                    self.preprocessor.process_state_for_memory(observation),
                    self.select_action(observation),
                    0., False
                )
                if self.step > self.warmup and self.step % self.train_freq == 0:
                    self.update_policy()

                observation = None
                episode_steps = None
                episode_reward = None
                episode += 1


    def evaluate(self, env, num_episodes, max_episode_length=None, visualize=False, restore=False, debug=False):
        """Test your agent with a provided environment.
        
        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """
        assert self.compiled

        # store current value and restore in the end of function 
        if restore:
            temp_is_training = self.is_training
            temp_history = self.preprocessor.get_history()

        self.is_training = False
        observation = None
        episode_rewards = []
        for episode in range(num_episodes):

            # reset at the start of episode
            self.reset()
            observation = deepcopy(env.reset())
            episode_steps = 0
            episode_reward = 0.

            # TODO random start (keras-rl has this option)
                
            assert observation is not None

            # start episode
            done = False
            while not done:
                # basic operation, action ,reward, blablabla ...
                action = self.select_action(observation)
                observation, r, done, info = env.step(action)
                reward = self.preprocessor.process_reward(r)
                if max_episode_length and episode_steps >= max_episode_length -1:
                    done = True
                
                if visualize:
                    env.render(mode='human')

                # update
                observation = deepcopy(observation)
                episode_reward += reward
                episode_steps += 1

            if debug: utils.prYellow('[Evaluate] #Episode{}: episode_reward:{}'.format(episode,episode_reward))
            episode_rewards.append(episode_reward)

        if restore:
            self.is_training = temp_is_training
            self.preprocessor.reset(temp_history)

        return np.array(episode_rewards).reshape(-1,1)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)
        self.target_model.set_weights(self.model.get_weights())

    def save_weights(self, filepath, overwrite=False):
        self.model.save_weights(filepath, overwrite=overwrite)

    def reset(self):
        self.preprocessor.reset()
        self.model.reset_states()
        self.target_model.reset_states()
