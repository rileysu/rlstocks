import gym
import pandas
import numpy

def convert_to_relative(prices):
    relative_prices = [0.0]
    for i in range(1, len(prices)):
        relative_prices.append(1.0 - (prices[i] / prices[i-1]))

    relative_prices = numpy.array(relative_prices)

    mean = numpy.mean(relative_prices)
    std = numpy.std(relative_prices)

    return (relative_prices - mean) / std

class TradingEnvironment(gym.Env):
    # Action Kinds = [Sell < -0.33, Hold otherwise, Buy > 0.33]
    # Quantity = Percentage of total money
    # Action Format = [ActionKind, Quantity]
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))
    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(16,))
    # Reward range default of -inf -> +inf is suitable so we don't need to do anything

    def __init__(self, init_balance_usd=20.0, data_csv='data/gemini_BTCUSD_1hr.csv'):
        self.data = numpy.flip(pandas.read_csv(data_csv)['Close'].to_numpy())
        self.relative_data = convert_to_relative(self.data)
        self.init_balance_usd = init_balance_usd

        self.reset()

    def _get_observation(self):
        return self.relative_data[self.curr_pos:self.curr_pos+16]

    def _get_current_price(self):
        return self.data[self.curr_pos+16]

    def _execute_action(self, action, curr_price):
        quantity = action[0]

        if quantity > 0.0: #Buy
            self.curr_balance_btc += (quantity * self.curr_balance_usd) / curr_price
            self.curr_balance_usd -= quantity * self.curr_balance_usd
        elif quantity < 0.0: #Sell
            self.curr_balance_btc -= -quantity * self.curr_balance_btc
            self.curr_balance_usd += curr_price * -quantity * self.curr_balance_btc

        #Otherwise nothing

    def step(self, action):
        observation = self._get_observation()

        self._execute_action(action, self._get_current_price())

        reward = self.curr_balance_usd

        self.curr_pos += 1
        self.done = (self.curr_pos + 16) >= (len(self.data) - 1) or ((self.curr_balance_btc * self._get_current_price() + self.curr_balance_usd) <= 0.50)
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
        print('Balance Value: ', (self.curr_balance_btc * self._get_current_price() + self.curr_balance_usd))
