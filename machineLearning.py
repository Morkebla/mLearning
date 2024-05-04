# Import necessary libraries and functions.
import pandas as pd  # Import the pandas library and alias it as pd.
import tensorflow as tf  # Import the TensorFlow library and alias it as tf.
from sklearn.model_selection import train_test_split  # Import the train_test_split function from sklearn.model_selection.

# Load the dataset from the CSV file 'cancer.csv' into a pandas DataFrame.
dataset = pd.read_csv('cancer.csv')

# Create a sequential model using TensorFlow's Keras API.
model = tf.keras.models.Sequential()

# Prepare the input features (x) and target variable (y) from the dataset.
x = dataset.drop(columns=["diagnosis(1=m, 0=b)"])  # Features (remove the diagnosis column).
y = dataset["diagnosis(1=m, 0=b)"]  # Target variable (diagnosis column).

# Split the dataset into training and testing sets using train_test_split function.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Add layers to the model: input layer, hidden layers, and output layer.
model.add(tf.keras.layers.Dense(256, input_shape=(x_train.shape[1],), activation='sigmoid' ))  # Input layer.
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))  # Hidden layer.
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # Output layer.

# Compile the model: specify optimizer, loss function, and evaluation metrics.
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model on the training data for a specified number of epochs.
model.fit(x_train, y_train, epochs=600)

model.evaluate(x_test, y_test)

# Make predictions on the test data
predictions = model.predict(x_test)

# Convert the predictions to binary labels (0 or 1)
binary_predictions = [1 if pred > 0.5 else 0 for pred in predictions]

# Display the predictions on the console
for i in range(len(x_test)):
    print(f"Actual: {y_test.iloc[i]}, Predicted: {binary_predictions[i]}")  