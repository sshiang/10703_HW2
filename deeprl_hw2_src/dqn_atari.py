#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os
import random
import time

import gym
from gym import wrappers

import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute)
from keras.models import (Model, Sequential)
from keras.optimizers import Adam
from keras.layers.core import Lambda
from keras import backend as K

import deeprl_hw2 as tfrl
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.objectives import mean_huber_loss
from deeprl_hw2.preprocessors import *
from deeprl_hw2.policy import *
from deeprl_hw2.memory import SequentialMemory
from deeprl_hw2.utils import (prRed, prGreen, prYellow)

# from ipdb import set_trace as debug

def create_model(window, input_shape, num_actions,
                 model_name='dqn'):  # noqa: D103
    """Create the Q-network model.

    Use Keras to construct a keras.models.Model instance (you can also
    use the SequentialModel class).

    We highly recommend that you use tf.name_scope as discussed in
    class when creating the model and the layers. This will make it
    far easier to understnad your network architecture if you are
    logging with tensorboard.

    Parameters
    ----------
    window: int
      Each input to the network is a sequence of frames. This value
      defines how many frames are in the sequence.
    input_shape: tuple(int, int)
      The expected input image size.
    num_actions: int
      Number of possible actions. Defined by the gym environment.
    model_name: str
      Useful when debugging. Makes the model show up nicer in tensorboard.

    Returns
    -------
    keras.models.Model
      The Q-model.
    """

    with tf.name_scope(model_name):
        
        # Linear model
        if model_name == "linear" or model_name == "dlinear" or model_name == "naive":
            S = Input(shape=input_shape + (window,))
            H = Flatten()(S)
            Q = Dense(num_actions, activation='linear')(H)

        # Deep model
        elif model_name =='duel_dqn' or model_name == 'dqn' or model_name == 'ddqn' :
            S = Input(shape=input_shape + (window,))
            H = Convolution2D(32, 8, 8, activation='relu', subsample=(4, 4))(S)
            H = Convolution2D(64, 4, 4, activation='relu', subsample=(2, 2))(H)
            H = Convolution2D(64, 3, 3, activation='relu', subsample=(1, 1))(H)
            H = Flatten()(H)

            if model_name == 'duel_dqn':
                V = Dense(512, activation='relu')(H)
                V = Dense(1, activation='linear')(V)

                A = Dense(512, activation='relu')(H)
                A = Dense(num_actions, activation='linear')(A)

                # Q = V + A - K.mean(A)
                Q = Lambda(lambda x: x[0] + x[1] - K.mean(x[1], keepdims=True))([V,A])

            else:
                Q = Dense(512, activation='relu')(H)
                Q = Dense(num_actions, activation='linear')(Q)

        else:
            raise RuntimeError('undefined model_name:{}'.format(model_name))

        model = Model(input=S, output=Q)

    print(model.summary())
    return model


def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    # os.makedirs(parent_dir, exist_ok=True)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    # os.makedirs(parent_dir, exist_ok=True)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    return parent_dir


def main():  # noqa: D103
    parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')
    parser.add_argument('--env', default='Enduro-v0', help='Atari env name')
    parser.add_argument(
        '-o', '--output', default='atari-v0', help='Directory to save data to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')

    # hyper-parameters
    parser.add_argument('--gamma',default=0.99, type=float, help='discount factor')
    parser.add_argument('--lr',default=0.0001, type=float, help='learning rate')
    parser.add_argument('--eps',default=0.05, type=float, help='epsilon')
    parser.add_argument('--training_steps',default=5000000, type=int, help='')
    parser.add_argument('--input_shape', default=84, type=int, help='Input size')
    parser.add_argument('--window', default=4, type=int, help='Window size')
    parser.add_argument('--rb_size',default=1000000, type=int, help='replay buffer size')
    parser.add_argument('--target_update_freq',default=10000, type=int, help='q target interval')
    parser.add_argument('--warmup',default=200, type=int, help='fill replay buffer')
    parser.add_argument('--train_freq',default=4, type=int, help='train_freq')
    parser.add_argument('--batch_size',default=32, type=int, help='batch_size')
    parser.add_argument('--episode_len',default=100000, type=int, help='max episode length')

    # policy options
    parser.add_argument(
        '--policy', default='greedy_eps_decay', type=str, help='Options: uniform_random/greedy/greedy_eps/greedy_eps_decay')
    parser.add_argument('--eps_min',default=0.0, type=float, help='epsilon min value')
    parser.add_argument('--eps_steps',default=10000, type=int, help='epsilon decay steps')

    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--record', default='', type=str)
    parser.add_argument('--model', default='dqn', type=str, help='Options: naive/linear/dqn/ddqn/duel_dqn')
    parser.add_argument('--validate_steps',default=10000, type=int, help='steps to run validate test')

    args = parser.parse_args()
    args.input_shape = tuple((args.input_shape,args.input_shape))


    if args.mode == "train":
        args.output = get_output_folder('{}-{}'.format(args.output,args.model), args.env)
    prGreen(args.output)

    # Create Env, Agent, and compile it
    env = gym.make(args.env)
    num_actions = env.action_space.n

    if args.debug: prGreen('initial q network')
    q_network = create_model(args.window, args.input_shape, num_actions, model_name=args.model) 

    if args.debug: prGreen('initial preprocessor')
    preprocessor = PreprocessorSequence([
        AtariPreprocessor(args.input_shape), 
        HistoryPreprocessor(args.window),
    ])

    if args.debug: prGreen('initial sequential memory')
    memory = SequentialMemory(args.rb_size, args.window)

    if args.debug: prGreen('initial policy')
    policy = {
        'uniform_random':UniformRandomPolicy(num_actions),
        'greedy':GreedyPolicy(),
        'greedy_eps':GreedyEpsilonPolicy(args.eps),
        'greedy_eps_decay':LinearDecayGreedyEpsilonPolicy(args.eps, args.eps_min, args.eps_steps),
    }.get(args.policy)

    if args.debug: prGreen('initial agent')
    agent = DQNAgent(
        num_actions,
        q_network,
        preprocessor,
        memory,
        policy,
        args.gamma,
        args.target_update_freq,
        args.warmup if args.mode == 'train' else 0,
        args.train_freq,
        args.batch_size,
        args.model,
        args.output
    )

    optimizer = Adam(args.lr)

    if args.debug: prGreen('compile ...')
    agent.compile(optimizer, 'huber_loss')

    # Start experiment
    if args.mode == 'train':    
        if args.debug: prGreen('fit ...')
        agent.fit(env,args.training_steps, args.validate_steps, debug=args.debug) # args.episode_len
        agent.save_weights(
            '{}/weights.h5f'.format(args.output),
            overwrite=True,
        )
    elif args.mode == 'test':
        if args.debug: prGreen('evaluate ...')
        agent.load_weights('{}/weights.h5f'.format(args.output))
        rewards = agent.evaluate(env,100, debug=args.debug, record=args.record)
        prGreen('Evaluate Summary: mean:{} std:{}'.format(np.mean(rewards), np.std(rewards)))
    
    elif args.mode =='record':

        for frame in [0,1666666, 3333332, 4999998]:
            env = gym.make(args.env)
            env = wrappers.Monitor(env, args.record+str(frame), force=True)

            if args.debug: prGreen('record frame {}...'.format(frame))
            agent.load_weights('{}/weights-{}.h5f'.format(args.output, frame))
            rewards = agent.evaluate(env,1,debug=args.debug)
            prGreen('Evaluate Summary: mean{}, std:{}'.format(np.mean(rewards), np.std(rewards)))
            
            env.close()

    else:
        raise RuntimeError('un-supported mode:{}'.format(args.mode))

if __name__ == '__main__':
    main()
