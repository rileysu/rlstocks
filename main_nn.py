import keras
import numpy

from dataset_generator import generate_dataset

model = keras.Sequential([
    keras.layers.TimeDistributed(keras.Sequential([
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
    ])),
    keras.layers.LSTM(1),
    keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_x, train_y = generate_dataset('data/gemini_BTCUSD_1hr.csv')

print(train_x.shape)

val_x, val_y = generate_dataset('data/gemini_ETHUSD_1hr.csv')

model.fit(train_x, train_y, batch_size=16, epochs=2)
model.evaluate(val_x, val_y)