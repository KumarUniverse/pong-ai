import gym
import time

# Env RBG dimensions: 210 x 160 x 3
env = gym.make('Pong-ram-v0') # Pong-v0 Pong-ram-v0 CartPole-v0
observation = env.reset()
print(f"type of array: {type(observation)}")  # numpy.ndarray
print(f"dimensions: {observation.shape}")
print(observation)
# Add another for loop here to control the number of simulation episodes.
for t in range(1000):  # render the env for 1000 timestamps
    time.sleep(0.1)
    env.render()
    action = env.action_space.sample() # take a random action
    observation, reward, done, info = env.step(action)
    if done:
        print(f"Episode finished after {t+1} timesteps.")
        break
env.close()