Decision Tree Classifier
This Python code implements a decision tree classifier from scratch using the ID3 algorithm. The decision tree classifier is built based on the concepts of entropy and information gain, aiming to predict the outcome of an outdoor activity based on various weather conditions.

Functions:

Node class: This class represents a node in the decision tree. It contains attributes for the feature to split on (feature), the label if it's a leaf node (label), and the child nodes (children). The purpose of this class is to organize the structure of the decision tree.

get_unique_values(data, feature_index): 
This function retrieves the unique values of a specific feature in the dataset. It takes the dataset data and the index of the feature feature_index as input and returns a set of unique values for that feature.

split_data(data, feature_index, value): 
This function splits the dataset based on a specific feature value. It takes the dataset data, the index of the feature feature_index, and the value to split on value as input, and returns a subset of the dataset where the specified feature has the given value.

entropy(data): 
This function calculates the entropy of the dataset. Entropy is a measure of disorder or randomness in the dataset. It takes the dataset data as input and computes the entropy using Shannon's entropy formula. The purpose of this function is to quantify the uncertainty in the dataset.

information_gain(data, feature_index): 
This function calculates the information gain for a specific feature. Information gain measures the reduction in entropy achieved by splitting the dataset based on the values of that feature. It takes the dataset data and the index of the feature feature_index as input and returns the information gain.

find_best_split(data, features): 
This function finds the best feature to split on based on the highest information gain. It iterates through all features and calculates their information gain, then returns the feature with the highest information gain. The purpose is to determine the optimal feature for splitting the dataset.

build_decision_tree(data, features): 
This recursive function builds the decision tree using the ID3 algorithm. It takes the dataset data and the list of features features as input and returns the constructed decision tree. The function recursively splits the dataset based on the best features and continues until all instances belong to the same class or no features are left.

print_decision_tree(node, depth=0, parent_branch=""): 
This function prints the decision tree in a readable format. It takes a node of the decision tree node, the current depth of the node depth, and the parent branch information parent_branch as input. The function recursively traverses the tree and prints each node along with its children, indicating the feature split and outcome counts.




Usage
The code consists of two main parts:

1.Feature Ranking: In this section, the importance of each weather condition feature (Outlook, Temperature, Humidity, and Wind) in predicting the outdoor activity outcome is analyzed. The code calculates the information gain for each feature and ranks them accordingly.

2.Decision Tree Construction: Here, the decision tree is built recursively using the ID3 algorithm. The dataset is split based on the feature with the highest information gain at each node, and the process continues until the tree is fully constructed.

To run the code, simply execute the Python script. Ensure that the necessary Python libraries are installed, particularly math for mathematical calculations.

bash
python decision_tree_classifier.py

Dataset
The dataset used in this code is provided directly within the script as a list of lists. Each sublist represents an instance, where the first four elements denote the weather conditions (Outlook, Temperature, Humidity, and Wind), and the last element represents the outcome of the outdoor activity (Yes or No).

References
Medium - Decision Tree for Classification: Entropy and Information Gain
Medium - Decision Trees
GeeksforGeeks - Difference between Feature Selection and Feature Extraction
Medium - Entropy & Conditional Entropy