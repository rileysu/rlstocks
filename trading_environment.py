import gym
import pandas
import numpy

def convert_to_relative(prices):
    relative_prices = [0.0]
    for i in range(1, len(prices)):
        relative_prices.append(1.0 - (prices[i] / prices[i-1]))

    return numpy.array(relative_prices)

class TradingEnvironment(gym.Env):
    # Action Kinds = [Sell < -0.33, Hold otherwise, Buy > 0.33]
    # Quantity = Percentage of total money
    # Action Format = [ActionKind, Quantity]
    action_space = gym.spaces.Box(low=numpy.array([-1.0, 0.0]), high=numpy.array([1.0, 1.0]), shape=(2,))
    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(16,))
    # Reward range default of -inf -> +inf is suitable so we don't need to do anything

    def __init__(self, init_balance_usd=20.0, data_csv='data/gemini_BTCUSD_1hr.csv'):
        self.data = convert_to_relative(pandas.read_csv(data_csv)['Close'].to_numpy())
        self.init_balance_usd = init_balance_usd

        self.reset()

    def _get_observation(self):
        return self.data[self.curr_pos:self.curr_pos+16]

    def _get_current_price(self):
        return self.data[self.curr_pos+16]

    def _execute_action(self, action, curr_price):
        kind = action[0]
        quantity = action[1]

        if kind > 0.33: #Buy
            self.curr_balance_btc += curr_price / quantity * self.curr_balance_usd
            self.curr_balance_usd -= quantity * self.curr_balance_usd
        elif kind < -0.33: #Sell
            self.curr_balance_btc -= quantity * self.curr_balance_btc
            self.curr_balance_usd += curr_price * quantity * self.curr_balance_btc

        #Otherwise nothing

    def step(self, action):
        observation = self._get_observation()

        self._execute_action(action, self._get_current_price())

        reward = self.curr_balance_usd

        self.curr_pos += 1
        self.done = self.curr_pos + 16 >= len(self.data) or (self.curr_balance_btc <= 0.0 and self.curr_balance_usd <= 0.0)
        done = self.done

        info = {}

        return observation, reward, done, info

    def reset(self):
        self.curr_pos = 0
        self.curr_balance_btc = 0.0
        self.curr_balance_usd = self.init_balance_usd
        self.done = False

        observation = self._get_observation()

        return observation

    def render(self):
        print('Balance BTC: ', self.curr_balance_btc)
        print('Balance USD: ', self.curr_balance_usd)