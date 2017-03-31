# 10703_HW2

For training: 
python dqn_atari.py --model dqn --debug --output [model_name]

For testing:
python dqn_atari.py --model dqn --debug --output [model_name] --mode test

Parameters: 
--model [model] (default: dqn)
          naive: linear network without experience replay
          linear: linear network
          dlinear: double linear network
          dqn: deep Q network
          ddqn: double deep Q network
          duel_dqn: dueling deep Q network
--env = [env] (default: Enduro-v0)
