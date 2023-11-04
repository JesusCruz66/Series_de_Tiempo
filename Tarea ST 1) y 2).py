from matplotlib import pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import os

fname = os.path.join("jena_climate_2009_2016.csv")
with open(fname) as f:
   data = f.read()
lines = data.split("\n")
header = lines[0].split(",")
lines = lines[1:-1]


temperature = np.zeros((len(lines),))
raw_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
  values = [float(x) for x in line.split(",")[1:]]
  temperature[i] = values[1]
  raw_data[i, :] = values[:]

num_total_samples = len(lines)
num_train_samples = int(0.5 * num_total_samples)
num_val_samples = int(0.25 * num_total_samples)+1
num_test_samples = int(0.25 * num_total_samples)
delay = 10

train_dataset = keras.utils.timeseries_dataset_from_array(
    data=temperature[:-delay],
    targets=temperature[delay:],
    sampling_rate=1,
    sequence_length=120,
    shuffle=True,
    batch_size=100,
    end_index=num_train_samples)

val_dataset = keras.utils.timeseries_dataset_from_array(
    data=temperature[:-delay],
    targets=temperature[delay:],
    sampling_rate=1,
    sequence_length=120,
    shuffle=True,
    batch_size=100,
    start_index=num_train_samples,
    end_index=num_train_samples + num_val_samples)

test_dataset = keras.utils.timeseries_dataset_from_array(
    data=temperature[:-delay],
    targets=temperature[delay:],
    sampling_rate=1,
    sequence_length=120,
    shuffle=True,
    batch_size=100,
    start_index=num_train_samples + num_val_samples)



model = keras.Sequential()

model.add(layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(120, 1)))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

model.fit(train_dataset, validation_data=val_dataset, epochs=10)


predictions = model.predict(test_dataset)

real_temperatures = temperature[num_train_samples + num_val_samples + delay:]

#time = range(num_test_samples)
time = range(len(real_temperatures))

plt.plot(time, real_temperatures, label='Real Temperature', color='blue')

timee = range(len(predictions))

plt.plot(timee, predictions, label='Predicted Temperature', color='orange')

plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend()
plt.show()