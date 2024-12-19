import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import Callback

# generate random data for the dataset
num_samples = 1000

data = {
    'size': np.random.uniform(-5, 5, num_samples),

    'weight': np.random.uniform(-5, 5, num_samples),

    'sweetness': np.random.uniform(-5, 5, num_samples),


    'softness': np.random.uniform(-5, 5, num_samples),

    'harvest_time': np.random.uniform(-5, 5, num_samples),



    'ripeness': np.random.uniform(-5, 5, num_samples),



    'acidity': np.random.uniform(-5, 5, num_samples),
    'quality': np.random.randint(2, size=num_samples)  # binary quality label
}

# convert the dictionary to a DataFrame


df = pd.DataFrame(data)

# data partitioning
X_features = df.drop('quality', axis=1)
y_label = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X_features, y_label, test_size=0.2, random_state=42)

# normalize the data


X_train_normalized = X_train / np.max(X_train)
X_test_normalized = X_test / np.max(X_test)


# define the neural network model
model = Sequential([


    Flatten(input_shape=(7,)),  # flatten layer added

    Dense(64, activation='relu'),  # fully connected layer with 64 neurons and ReLU activation

    Dense(32, activation='relu'),  # fully connected layer with 32 neurons and ReLU activation
    Dense(1, activation='sigmoid')  # output layer with 1 neuron and sigmoid activation for binary classification

])

# define custom callback for magnitude-based pruning


class MagnitudePruning(Callback):
    def __init__(self, pruning_percent):

        super(MagnitudePruning, self).__init__()

        self.pruning_percent = pruning_percent

    def on_epoch_end(self, epoch, logs=None):

        if epoch > 0:  # prune after the first epoch

            for layer in self.model.layers:


                if isinstance(layer, Dense):
                    weights = layer.get_weights()[0]
                    weights_abs = np.abs(weights)
                    threshold = np.percentile(weights_abs, self.pruning_percent)

                    pruned_weights = np.where(weights_abs <= threshold, 0, weights)
                    layer.set_weights([pruned_weights, layer.get_weights()[1]])


# range of pruning thresholds to test
pruning_thresholds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]



# lists to store results
num_params_list = []  # number of parameters in the model
accuracy_list = []  # classification accuracies for different pruning thresholds

# iterate over thresholds to evaluate the effect of pruning on model performance
for threshold in pruning_thresholds:
    # compile the model with binary cross-entropy loss and accuracy metric
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # define pruning callback instance
    pruning_callback = MagnitudePruning(pruning_percent=threshold)



    # train the model with pruning callback
    history = model.fit(X_train_normalized, y_train, epochs=100, batch_size=1,
                        validation_data=(X_test_normalized, y_test), callbacks=[pruning_callback], verbose=0)

    # evaluate the pruned model on the test set
    test_loss, test_accuracy = model.evaluate(X_test_normalized, y_test, verbose=2)
    print('\nTest accuracy:', test_accuracy)

    # store the classification accuracy and number of parameters for plotting
    accuracy_list.append(test_accuracy)


    num_params = sum([np.prod(layer.get_weights()[0].shape) for layer in model.layers if isinstance(layer, Dense)])
    num_params_list.append(num_params)

# plot training history

plt.plot(history.history['accuracy'], label='accuracy')  # plot training accuracy



plt.plot(history.history['val_accuracy'], label='val_accuracy')  # plot validation accuracy
plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.ylim([0, 1])

plt.legend(loc='lower right')
plt.show()

# plotting
plt.figure(figsize=(10, 5))

# plot number of parameters vs pruning threshold
plt.subplot(1, 2, 1)
plt.plot(pruning_thresholds, num_params_list, marker='o')

plt.title('Number of Parameters vs Pruning Threshold')
plt.xlabel('Pruning Threshold (%)')
plt.ylabel('Number of Parameters')

# plot classification accuracy vs pruning threshold

plt.subplot(1, 2, 2)
plt.plot(pruning_thresholds, accuracy_list, marker='o')


plt.title('Classification Accuracy vs Pruning Threshold')
plt.xlabel('Pruning Threshold (%)')
plt.ylabel('Classification Accuracy')

plt.tight_layout()
plt.show()
