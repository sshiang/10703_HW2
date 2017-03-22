
import random
import numpy as np
from collections import deque, namedtuple

from deeprl_hw2 import utils
from deeprl_hw2.core import ReplayMemory, Sample

from ipdb import set_trace as debug

class RingBuffer(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.clear()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v

    def clear(self):
        self.start = 0
        self.length = 0
        self.data = [None for _ in range(self.maxlen)]

class SequentialMemory(ReplayMemory):

    def __init__(self, *args):
        super(SequentialMemory, self).__init__(*args)
        self.actions = RingBuffer(self.max_size)
        self.rewards = RingBuffer(self.max_size)
        self.states  = RingBuffer(self.max_size)
        self.terminals = RingBuffer(self.max_size)

    def _sample_batch_indexes(self, low, high, size):
        if high - low >= size:
            try:
                r = xrange(low, high) 
            except NameError:
                r = range(low, high) 
            batch_idxs = random.sample(r, size)
        else:
            # warnings.warn('Not enough entries to sample without replacement. Consider increasing your warm-up phase to avoid oversampling!')
            batch_idxs = np.random.random_integers(low, high - 1, size=size)
        return batch_idxs

    def append(self, state, action, reward, terminal):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(terminal)

    @property
    def nb_entries(self):
        return len(self.states)
    
    def sample(self, batch_size, indexes=None):
        
        # create batch indexes
        if indexes is None:
            indexes = self._sample_batch_indexes(0, self.nb_entries - 1, size=batch_size)
        assert (np.min(indexes) >= 1 and np.max(indexes) < self.nb_entries and len(indexes) == batch_size)

        # create samples
        samples = []
        for idx in indexes:
            # if self.states[idx] is the terminal state, resample it
            terminal0 = self.terminals[idx - 2] if idx >= 2 else False
            while terminal0:
                idx = self._sample_batch_indexes(1, self.nb_entries, size=1)[0]
                terminal0 = self.terminals[idx - 2] if idx >= 2 else False
            assert 1 <= idx < self.nb_entries

            # fill state0 to window_length size, if encounter terminate, add zero instead
            state0 = [self.states[idx - 1]]
            for offset in range(0, self.window_length - 1):
                current_idx = idx - 2 - offset
                current_terminal = self.terminals[current_idx - 1] if current_idx - 1 > 0 else False
                if current_idx < 0 or current_terminal:
                    break
                state0.insert(0, self.states[current_idx])
            while len(state0) < self.window_length:
                state0.insert(0, np.zeros(state0[0].shape))

            # fill state1
            state1 = [np.copy(x) for x in state0[1:]]
            state1.append(self.states[idx])
            # debug()
            assert len(state0) == self.window_length
            assert len(state1) == len(state0)

            state0 = np.transpose(np.array(state0), (1,2,0))
            state1 = np.transpose(np.array(state1), (1,2,0))

            # fill others
            action = self.actions[idx - 1]
            reward = self.rewards[idx - 1]
            terminal1 = self.terminals[idx - 1]

            # build sample
            samples.append(
                Sample(state0, action, reward, state1, terminal1)
            )

        assert len(samples) == batch_size
        return samples


    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.terminals.clear()
