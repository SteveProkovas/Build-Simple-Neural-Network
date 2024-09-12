import numpy as np
from tensorflow import keras

# Generate some example data
# Inputs (x1, x2, x3)
X = np.array([[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0],
              [0, 1, 1],
              [1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]])

# Outputs (target)
y = np.array([[0], [0], [0], [1], [0], [1], [1], [1]])

# Build the model
model = keras.Sequential([
    keras.layers.Dense(4, activation='relu', input_shape=(3,)),  # Hidden layer with 4 neurons
    keras.layers.Dense(1, activation='sigmoid')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=1000, verbose=0)

# Make predictions
predictions = model.predict(X)
print(predictions)
