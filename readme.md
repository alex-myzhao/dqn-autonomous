# DQN for Autonomous

## Deep Q-Learning
> Unlike policy gradient methods, which attempt to learn functions which directly map an observation to an action, Q-Learning attempts to learn the estimated future return of being in a given state, and taking a specific action there.

## How to use

These steps are tested on macOS Mojave 10.14.4 and Miniconda 4.6.14.

### Install required packages
To create a conda environment including required packages:

```bash
make
```

### Offline training

```bash
make offline
```

### Online learning

1. First, open the simulator `term2_sim.app`.
2. Select the faster speed and lowest quality.
3. Open "project 4: PID controller" in the simulator.
4. Run the python script to connect to the server and start training.

```bash
make online
```

## Reference

- [Simple Reinforcement Learning with TensorFlow](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0)
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
- [A Walk-through of AlexNet](https://medium.com/@smallfishbigsea/a-walk-through-of-alexnet-6cbd137a5637)
- [Deep Q-Learning with Keras and Gym](https://keon.io/deep-q-learning/)
