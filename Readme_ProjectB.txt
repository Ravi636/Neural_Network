Neural Network Pruning Project
This project implements neural network pruning using TensorFlow/Keras. The goal is to explore the impact of pruning thresholds on model performance and sparsity.

Overview
The project comprises several components:

Data Generation: Random data simulating characteristics of fruits is generated for training and testing the neural network.
Model Architecture: A simple feedforward neural network is constructed using TensorFlow/Keras, consisting of input, hidden, and output layers.
Magnitude-Based Pruning: A custom callback is implemented to perform magnitude-based pruning on the weights of the neural network during training.
Evaluation: The pruned models are evaluated on a test dataset to analyze classification accuracy and sparsity.
Visualization: The results are visualized using matplotlib, showing the number of parameters vs. pruning threshold and classification accuracy vs. pruning threshold.
Requirements
To run the code, you'll need:

Python 3.x
Libraries: numpy, matplotlib, pandas, scikit-learn, TensorFlow/Keras
Usage
Clone this repository to your local machine.
Install the required libraries using pip install -r requirements.txt.
Run the main script neural_network_pruning.py.
View the results in the generated plots.
File Structure
neural_network_pruning.py: Main script containing the implementation.
README.md: This file, providing an overview and instructions.
requirements.txt: List of required libraries.