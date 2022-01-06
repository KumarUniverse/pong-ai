# pong-ai
A Pong game played by two AIs.

![Pong GUI](https://github.com/KumarUniverse/pong-ai/blob/main/img/how_pong_ai_sees_the_game.png)

AI1 is a Q-learning agent and AI2 is the near-perfect opponent.
Compared to previously related work which train Pong RL agents by combining Q-learning with deep learning in an algorithm known as Deep Q-Networks, this implementation takes advantage of known environment constraints of the custom-made Pong environment to train the agent using one-step Q-learning alone.

This work highlights that it is possible to use one-step Q-learning, a model-free, off-policy reinforcement learning algorithm typically relegated to solving simple maze world environments, in combination with a POMDP and a novel technique called state distillation to train a Q-agent to play Pong and converge to the optimal policy.

For more info, please refer to the project's [thesis paper](https://github.com/KumarUniverse/pong-ai/blob/main/Akash-Kumar-Masters-Thesis-Playing-Pong-Using-Q-Learning.pdf).