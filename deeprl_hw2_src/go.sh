
# Double DQN
./dqn_atari.py --debug --model ddqn --output atari-v0-ddqn/Enduro-v0-run4 --mode record --record record/ddqn

# Duel DQN
./dqn_atari.py --debug --model duel_dqn --output enduro-v0-duel-duel_dqn/Enduro-v0-run2 --mode record --record record/duel

# DQN
./dqn_atari.py --debug --model dqn --output enduro-v0-dqn-dqn/Enduro-v0-run1 --mode record --record record/dqn


# Double linear
./dqn_atari.py --debug --model dlinear --output dlinear-dlinear/Enduro-v0-run1 --mode record --record record/dlinear

# Linear
./dqn_atari.py --debug --model linear --output linear-linear/Enduro-v0-run4 --mode record --record record/linear

# Naive
./dqn_atari.py --debug --model naive --output naive-naive/Enduro-v0-run2 --mode record --record record/naive