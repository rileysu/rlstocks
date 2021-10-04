import pandas
import numpy
from sklearn.preprocessing import MinMaxScaler

def generate_relative_frame(prices):
    ref_price = prices[0]
    relative_prices = [0.0]

    for price in prices[1:]:
        if ref_price != 0:
            relative_prices.append(price / ref_price  - 1.0)
        else:
            quit('encountered zero ref value')

    relative_prices = numpy.array(relative_prices)

    return relative_prices

def generate_dataset(file, window_length=16):
    data = numpy.flip(pandas.read_csv(file)['Close'].replace(numpy.nan, 0.0).to_numpy())
    samples = []
    labels = []

    for i in range(len(data) - window_length - 1):
        frame = generate_relative_frame(data[i:i+window_length+1])
        samples.append(frame[:window_length])
        labels.append(1.0 if frame[-1] > 0 else 0.0)

    samples_scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    samples, labels = numpy.array(samples), numpy.array(labels).reshape(-1, 1)

    samples_scaler.fit(samples)

    return samples_scaler.transform(samples), labels

    

