import keras

from dataset_generator import generate_dataset

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_x, train_y = generate_dataset('data/gemini_BTCUSD_1hr.csv')
val_x, val_y = generate_dataset('data/gemini_ETHUSD_1hr.csv')

model.fit(train_x, train_y, batch_size=16, epochs=1)
model.evaluate(val_x, val_y)