# Reinforcement Learning using PyTorch
The networks for the RL algorithms are coded entirely in PyTorch, including choosing the optimizers and calculation of the loss functions. <br /><br />


This library has been heavily inspired by: <br />
https://github.com/higgsfield/RL-Adventure <br />
https://github.com/higgsfield/RL-Adventure-2 <br />
https://deeplearningcourses.com/c/artificial-intelligence-reinforcement-learning-in-python <br />
https://deeplearningcourses.com/c/deep-reinforcement-learning-in-python <br />
https://deeplearningcourses.com/c/cutting-edge-artificial-intelligence <br />
https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code <br />
https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code <br />

Based from the above inspiration, I have created agents using DQN, Double DQN, Dueling DQN, Double Dueling DQN, PPO, REINFORCE, ActorCritic, DDPG, TD3 and SAC.
The intent is for this library to work with OpenAI Gym style environments <br />

Setting up the library is simple as follows <br />
1. Navigate to the root of the torch_rl directory on your local machine on the terminal<br />
2. Run python setup.py sdist <br />
3. Run python -m pip install dist/torch-rl-4.0.tar.gz <br />
