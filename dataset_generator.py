import pandas
import numpy
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

def generate_ohlcv(data, index):
    return numpy.array([
        data.loc[index]['Open'],
        data.loc[index]['High'],
        data.loc[index]['Low'],
        data.loc[index]['Close'],
        data.loc[index]['Volume'],
    ])

def generate_relative_frame(data, start_index, window_length):
    ref_ohlcv = generate_ohlcv(data, start_index)
    outs = [
        numpy.array([
            0.0,
            0.0,
            0.0,
            0.0,
            ref_ohlcv[-1]
        ])
    ]

    for index in range(start_index + 1, start_index + window_length + 1):
        curr_ohlcv = generate_ohlcv(data, index)

        curr_ohlc = curr_ohlcv[:-1]
        curr_volume = curr_ohlc[-1]

        rel_ohlc = (ref_ohlcv[:-1] / curr_ohlc - 1.0)

        outs.append(numpy.concatenate([rel_ohlc, numpy.array([curr_volume])]))

    return numpy.array(outs)

def generate_dataset(file, window_length=16):
    data = pandas.read_csv(file)[::-1]
    samples = []
    labels = []

    for index in tqdm(range(len(data) - window_length - 1)):
        frame = generate_relative_frame(data, index, window_length)
        samples.append(frame[:-1])
        labels.append(1.0 if frame[-1][-2] > 0 else 0.0) #Close price

    old_samples = numpy.array(samples)

    samples_scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    samples, labels = numpy.array(samples).reshape(-1, 5), numpy.array(labels)

    samples_scaler.fit(samples)

    return samples_scaler.transform(samples).reshape(-1, window_length, 5), labels

    

