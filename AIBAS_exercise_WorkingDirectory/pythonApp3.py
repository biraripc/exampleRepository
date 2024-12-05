
# from pybrain.tools.shortcuts import buildNetwork
# from pybrain.supervised.trainers import BackpropTrainer
# from pybrain.datasets import SupervisedDataSet
# import numpy as np

# # Function to preprocess and load data
# def load_data(file_path):
#     data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
#     inputs = data[:, :-1]
#     targets = data[:, -1:]
#     return inputs, targets

# # Function to create training and testing datasets
# def prepare_datasets(inputs, targets, split_ratio=0.8):
#     dataset_size = len(inputs)
#     split_index = int(dataset_size * split_ratio)
#     train_inputs, test_inputs = inputs[:split_index], inputs[split_index:]
#     train_targets, test_targets = targets[:split_index], targets[split_index:]

#     train_ds = SupervisedDataSet(inputs.shape[1], 1)
#     test_ds = SupervisedDataSet(inputs.shape[1], 1)

#     for i in range(len(train_inputs)):
#         train_ds.addSample(train_inputs[i], train_targets[i])
#     for i in range(len(test_inputs)):
#         test_ds.addSample(test_inputs[i], test_targets[i])

#     return train_ds, test_ds

# # Function to create a feedforward network
# def create_network(input_size, hidden_size, output_size):
#     return buildNetwork(input_size, hidden_size, output_size, bias=True)

# # Function to train the model
# def train_model(network, train_data, learningrate=0.01, max_epochs=10):
#     trainer = BackpropTrainer(network, train_data, learningrate=learningrate)
#     for epoch in range(max_epochs):
#         error = trainer.train()
#         print(f"Epoch {epoch + 1}/{max_epochs}, Training Error: {error}")

# # Main function
# if __name__ == "__main__":
#     # Load data
#     file_path = "dataset03.csv" 
#     inputs, targets = load_data(file_path)

#     # Prepare datasets
#     train_data, test_data = prepare_datasets(inputs, targets)

#     # Create network
#     input_size = train_data.indim
#     hidden_size = 5 
#     output_size = train_data.outdim
#     ann = create_network(input_size, hidden_size, output_size)

#     # Train the model
#     print("Training the ANN model...")
#     train_model(ann, train_data, learningrate=0.005, max_epochs=100)

import pickle
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
import numpy as np


# Function to preprocess and load data
def load_data(file_path):
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    inputs = data[:, :-1]
    targets = data[:, -1:]
    return inputs, targets


# Function to create training and testing datasets
def prepare_datasets(inputs, targets, split_ratio=0.8):
    dataset_size = len(inputs)
    split_index = int(dataset_size * split_ratio)
    train_inputs, test_inputs = inputs[:split_index], inputs[split_index:]
    train_targets, test_targets = targets[:split_index], targets[split_index:]

    train_ds = SupervisedDataSet(inputs.shape[1], 1)
    test_ds = SupervisedDataSet(inputs.shape[1], 1)

    for i in range(len(train_inputs)):
        train_ds.addSample(train_inputs[i], train_targets[i])
    for i in range(len(test_inputs)):
        test_ds.addSample(test_inputs[i], test_targets[i])

    return train_ds, test_ds


# Function to create a feedforward network
def create_network(input_size, hidden_size, output_size):
    return buildNetwork(input_size, hidden_size, output_size, bias=True)


# Function to train the model
def train_model(network, train_data, learningrate=0.01, max_epochs=10):
    trainer = BackpropTrainer(network, train_data, learningrate=learningrate)
    for epoch in range(max_epochs):
        error = trainer.train()
        print(f"Epoch {epoch + 1}/{max_epochs}, Training Error: {error}")


# Function to save the trained model
def save_model(model, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {file_name}")


# Function to load the trained model
def load_model(file_name):
    with open(file_name, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded from {file_name}")
    return model


# Function to activate models and compare outputs
def compare_models(original_model, loaded_model, test_samples):
    print("\nComparing activations from the original and loaded models:")
    for i, sample in enumerate(test_samples):
        original_output = original_model.activate(sample)
        loaded_output = loaded_model.activate(sample)
        print(f"Sample {i + 1}:")
        print(f"  Original Model Output: {original_output}")
        print(f"  Loaded Model Output: {loaded_output}")
        print()


# Main function
if __name__ == "__main__":
    # Load data
    file_path = "dataset03.csv"  # Update the path to your dataset
    inputs, targets = load_data(file_path)

    # Prepare datasets
    train_data, test_data = prepare_datasets(inputs, targets)

    # Create network
    input_size = train_data.indim
    hidden_size = 5  # Adjustable parameter
    output_size = train_data.outdim
    ann = create_network(input_size, hidden_size, output_size)

    # Train the model
    print("Training the ANN model...")
    train_model(ann, train_data, learningrate=0.005, max_epochs=50)

    # Save the trained model
    model_file = "UE_05_App3_ANN_Model.xml"
    save_model(ann, model_file)

    # Load the trained model
    loaded_ann = load_model(model_file)

    # Compare model activations
    test_samples = [inputs[0], inputs[1]]  # Choose two samples for activation
    compare_models(ann, loaded_ann, test_samples)


