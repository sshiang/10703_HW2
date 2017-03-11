"""Suggested Preprocessors."""

import numpy as np
from PIL import Image

from deeprl_hw2 import utils
from deeprl_hw2.core import Preprocessor


class HistoryPreprocessor(Preprocessor):
    """Keeps the last k states.

    Useful for domains where you need velocities, but the state
    contains only positions.

    When the environment starts, this will just fill the initial
    sequence values with zeros k times.

    Parameters
    ----------
    history_length: int
      Number of previous states to prepend to state being processed.

    """

    def __init__(self, history_length=1):
        self.history_length = history_length
        self.reset()

    def process_state_for_network(self, state):
        """You only want history when you're deciding the current action to take."""

        if self.history is None:
            self.history = np.dstack((state, state, state, state))
        else:
            self.history = np.dstack((self.history[:,:,1:], state))

        assert self.history.shape[-1] == self.history_length
        return self.history

    def reset(self):
        """Reset the history sequence.

        Useful when you start a new episode.
        """
        self.history = None

    def get_config(self):
        return {'history_length': self.history_length}


class AtariPreprocessor(Preprocessor):
    """Converts images to greyscale and downscales.

    Based on the preprocessing step described in:

    @article{mnih15_human_level_contr_throug_deep_reinf_learn,
    author =	 {Volodymyr Mnih and Koray Kavukcuoglu and David
                  Silver and Andrei A. Rusu and Joel Veness and Marc
                  G. Bellemare and Alex Graves and Martin Riedmiller
                  and Andreas K. Fidjeland and Georg Ostrovski and
                  Stig Petersen and Charles Beattie and Amir Sadik and
                  Ioannis Antonoglou and Helen King and Dharshan
                  Kumaran and Daan Wierstra and Shane Legg and Demis
                  Hassabis},
    title =	 {Human-Level Control Through Deep Reinforcement
                  Learning},
    journal =	 {Nature},
    volume =	 518,
    number =	 7540,
    pages =	 {529-533},
    year =	 2015,
    doi =        {10.1038/nature14236},
    url =	 {http://dx.doi.org/10.1038/nature14236},
    }

    You may also want to max over frames to remove flickering. Some
    games require this (based on animations and the limited sprite
    drawing capabilities of the original Atari).

    Parameters
    ----------
    new_size: 2 element tuple
      The size that each image in the state should be scaled to. e.g
      (84, 84) will make each image in the output have shape (84, 84).
    """

    def __init__(self, new_size):
        self.new_size = new_size

    def _process_state(self, state, convert_type):
        """
        convert_type: str ('unit8' or 'float32')
        """
        assert state.ndim == 3  # (height, width, channel)

        img = Image.fromarray(state)
        img = img.resize(self.new_size).convert('L')  # resize and convert to grayscale
        
        processed_observation = np.array(img)
        assert processed_observation.shape == self.new_size

        return processed_observation.astype(convert_type) 
               

    def process_state_for_memory(self, state):
        """Scale, convert to greyscale and store as uint8.

        We don't want to save floating point numbers in the replay
        memory. We get the same resolution as uint8, but use a quarter
        to an eigth of the bytes (depending on float32 or float64)

        We recommend using the Python Image Library (PIL) to do the
        image conversions.
        """
        return self._process_state(state, 'uint8')

    def process_state_for_network(self, state):
        """Scale, convert to greyscale and store as float32.

        Basically same as process state for memory, but this time
        outputs float32 images.
        """
        return self._process_state(state, 'float32')

    def process_batch(self, samples):
        """The batches from replay memory will be uint8, convert to float32.

        Same as process_state_for_network but works on a batch of
        samples from the replay memory. Meaning you need to convert
        both state and next state values.
        """
        # state, action, reward, next_state, is_terminal
        # processed_batch = [Sample(
        #     sample.state.astype('float32'),
        #     sample.action,
        #     sample.reward,
        #     sample.next_state.astype('float32'),
        #     sample.is_terminal,
        # ) for sample in samples]

        state_batch = np.array([ 
            s.state.astype('float32') for s in samples
        ])
        action_batch = np.array([ s.action for s in samples])
        reward_batch = np.array([ s.reward for s in samples])
        terminal_batch = np.array([ 0. if s.is_terminal else 1. for s in samples])
        next_state_batch = np.array([ 
            s.next_state.astype('float32') for s in samples
        ])

        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch

    def process_reward(self, reward):
        """Clip reward between -1 and 1."""
        return np.clip(reward, -1., 1.)


class PreprocessorSequence(Preprocessor):
    """You may find it useful to stack multiple prepcrocesosrs (such as the History and the AtariPreprocessor).

    You can easily do this by just having a class that calls each preprocessor in succession.

    For example, if you call the process_state_for_network and you
    have a sequence of AtariPreproccessor followed by
    HistoryPreprocessor. This class could implement a
    process_state_for_network that does something like the following:

    state = atari.process_state_for_network(state)
    return history.process_state_for_network(state)
    """
    def __init__(self, preprocessors):
        # FIXME
        self.atari = preprocessors[0]
        self.history = preprocessors[1]

    def process_state_for_network(self, state):
        state = self.atari.process_state_for_network(state)
        return self.history.process_state_for_network(state)

    def process_state_for_memory(self, state):
        return self.atari.process_state_for_memory(state)

    def process_batch(self, samples):
        return self.atari.process_batch(samples)

    def process_reward(self, reward):
        return self.atari.process_reward(reward)

    def reset(self):
        self.history.reset()