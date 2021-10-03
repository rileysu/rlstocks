import gym

from stable_baselines3 import DDPG

from trading_environment import TradingEnvironment

env = TradingEnvironment()

model = DDPG("MlpPolicy", env, verbose=2, learning_rate=0.001)
model.learn(total_timesteps=500)

obs = env.reset()
for i in range(6000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()
      break

env.close()
