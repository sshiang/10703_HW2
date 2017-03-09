"""Main DQN agent."""

import numpy as np
from deeprl_hw2 import utils

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
                 batch_size):

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
        self.rand_policy = lambda: np.random.randint(0, self.num_actions)
        self.training = True


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
        pass

    def calc_q_values(self, states):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        if not isinstance(states, list):
            states = [states]
        batch = self.preprocessor.process_batch(states)
        q_values = self.model.predict_on_batch(batch)
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

        # select action
        process_state = self.preprocessor.process_state_for_network(state) 
        if self.step < self.warmup:
            action = self.rand_policy()
        else:
            q_values = self.calc_q_values(process_state)
            action = self.policy.select_action(q_values,**kwargs)

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
        
        assert self.training
        if self.step < self.warmup:
            return

        batch = self.memory.sample(self.batch_size)

        # TODO
        # self.tau


    def fit(self, env, num_iterations, max_episode_length=None):
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
        """
        
        self.training = True
        self.step = 0
        observation = None
        while self.step < num_iterations:

            # reset if it is the start of episode
            if observation is None:
                observation = self.env.reset()
                episode_steps = 0
                episode_reward = 0.
            assert (observation is not None and episode_steps is not None and episode_reward is not None)

            # basic operation, action ,reward, blablabla ...
            action = self.select_action(observation)
            observation2, r, done, info = self.env.step(action)
            reward = self.preprocessor.process_reward(r)
            episode_reward += reward

            if max_episode_length and episode_steps >= max_episode_length -1:
                done = True

            # add replay memory and update policy            
            self.memory.append(
              self.preprocessor.process_state_for_memory(observation), 
              action, reward, done
            )
            if self.step % self.train_freq == 0:
                self.update_policy()

            episode_steps += 1
            self.step += 1

            if done: # end of episode
                # # We are in a terminal state but the agent hasn't yet seen it. We therefore
                # # perform one more forward-backward call and simply ignore the action before
                # # resetting the environment. We need to pass in `terminal=False` here since
                # # the *next* state, that is the state of the newly reset environment, is
                # # always non-terminal by convention.
                # self.forward(observation)
                # self.backward(0., terminal=False)

                observation = None
                episode_steps = None
                episode_reward = None
            else:
                observation = observation2
            

    def evaluate(self, env, num_episodes, max_episode_length=None):
        """Test your agent with a provided environment.
        
        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """
        pass

        self.training = False
        # TODO