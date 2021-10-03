import gym

from stable_baselines3 import DDPG

from trading_environment import TradingEnvironment

env = TradingEnvironment()

model = DDPG("MlpPolicy", env, verbose=2)
model.learn(total_timesteps=500)

obs = env.reset()
for i in range(2187):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    print(action)
    if done:
      obs = env.reset()
      break

env.close()