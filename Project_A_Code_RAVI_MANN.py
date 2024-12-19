import math



class Node:
    def __init__(self, feature=None, label=None):
        self.feature = feature  # Feature to split on
        
        self.label = label      # Label if it's a leaf node
        self.children = {}      # Child nodes



# Provided dataset
dataset = [
    ["Sunny", "Hot", "High", "Weak", "No"],
    ["Sunny", "Hot", "High", "Strong", "No"],
    ["Overcast", "Hot", "High", "Weak", "Yes"],
    ["Rain", "Mild", "High", "Weak", "Yes"],
    ["Rain", "Cool", "Normal", "Weak", "Yes"],
    ["Rain", "Cool", "Normal", "Strong", "No"],
    ["Overcast", "Cool", "Normal", "Strong", "Yes"],
    ["Sunny", "Mild", "High", "Weak", "No"],
    ["Sunny", "Cool", "Normal", "Weak", "Yes"],
    ["Rain", "Mild", "Normal", "Weak", "Yes"],
    ["Sunny", "Mild", "Normal", "Strong", "Yes"],
    ["Overcast", "Mild", "High", "Strong", "Yes"],
    ["Overcast", "Hot", "Normal", "Weak", "Yes"],
    ["Rain", "Mild", "High", "Strong", "No"],
]

# Function to get unique values of a feature in the dataset
def get_unique_values(data, feature_index):
    return set([instance[feature_index] for instance in data])

# Function to split data based on a feature value
def split_data(data, feature_index, value):
    return [instance for instance in data if instance[feature_index] == value]

# Function to calculate entropy
def entropy(data):
    
    # Counting total instances in the dataset
    total_instances = len(data)
    
    # Creating an empty dictionary to store label counts
    counts = {}
    
    # Iterating through each instance in the dataset
    for instance in data:
        
        # Extracting the label of the instance
        label = instance[-1]
        
        # Checking if the label is already in the dictionary
        if label not in counts:
            # If not, initializing its count to 0
            counts[label] = 0
        # Incrementing the count of the label
        
        counts[label] += 1
        
    # Initializing entropy value
    entropy_val = 0
    
    # Iterating through each label count
    for label in counts:
        # Calculating probability of the label
        
        prob = counts[label] / total_instances
        # Calculating entropy contribution of the label
        entropy_val -= prob * math.log2(prob)
        
        
    # Returning the calculated entropy value
    return entropy_val


# Function to calculate information gain
def information_gain(data, feature_index):
    
    # Calculating total entropy of the dataset
    total_entropy = entropy(data)
    
    # Getting unique values of the specified feature
    unique_values = get_unique_values(data, feature_index)
    
    # Initializing weighted entropy
    total_weighted_entropy = 0
    
    # Calculating weighted entropy for each unique value of the feature
    for value in unique_values:
        # Splitting the data based on the current feature value
        subset = split_data(data, feature_index, value)
        # Calculating the weighted entropy for this subset
        total_weighted_entropy += (len(subset) / len(data)) * entropy(subset)
        
        
    # Calculating and returning the information gain
    return total_entropy - total_weighted_entropy



# Feature names
feature_names = ['Outlook', 'Temperature', 'Humidity', 'Wind']

# Calculate and store information gain for each feature
info_gains = [(feature_names[i], information_gain(dataset, i)) for i in range(len(feature_names))]
ranked_features = sorted(info_gains, key=lambda x: x[1], reverse=True)



# Print ranked features
print("Ranked Features based on Information Gain:\n")
for i, (feature_name, gain) in enumerate(ranked_features):
    print(f"Rank {i+1}: Feature {feature_name} - Information Gain: {gain}")



# Function to find the best feature to split on
def find_best_split(data, features):
    
    
    best_gain = -1
    best_feature = None
    
    for feature_index in range(len(features)):
        gain = information_gain(data, feature_index)
        if gain > best_gain:
            
            best_gain = gain
            best_feature = features[feature_index]
            
            
    return best_feature

# Function to get unique values of a feature in the dataset
def get_unique_values(data, feature_index):
    return set([instance[feature_index] for instance in data])



# Function to split data based on a feature value
def split_data(data, feature_index, value):
    return [instance for instance in data if instance[feature_index] == value]


# Recursive function to build decision tree
def build_decision_tree(data, features):
    # Base case: If all instances have the same label
    labels = [instance[-1] for instance in data]
    if len(set(labels)) == 1:
        return labels[0]
    
    
    # Base case: If there are no features left to split on
    if len(features) == 0:
        return max(set(labels), key=labels.count)
    
    
    # Find the best feature to split on
    best_feature = find_best_split(data, features)
    tree = {best_feature: {}}
    feature_values = get_unique_values(data, features.index(best_feature))
    
    
    
    # Remove the selected feature from the list of features
    features_copy = features[:]
    features_copy.remove(best_feature)
    
    
    
    # Recursively build subtrees
    for value in feature_values:
        subset = split_data(data, features.index(best_feature), value)
        if len(subset) == 0:
            tree[best_feature][value] = max(set(labels), key=labels.count)
        else:
            tree[best_feature][value] = build_decision_tree(subset, features_copy)
    
    
    return tree

def print_decision_tree(node, depth=0, parent_branch=""):
    indentation = '  ' * depth
    if isinstance(node, str):
        print(indentation + parent_branch + 'Predict:', node)
        
    else:
        feature = list(node.keys())[0]
        print(indentation + parent_branch + 'Feature:', feature)
        children = node[feature]
        
        for value, child_node in children.items():
            subset_labels = [instance[-1] for instance in dataset if instance[feature_names.index(feature)] == value]
            yes_count = subset_labels.count('Yes')
            no_count = subset_labels.count('No')
            
            print(indentation + ' ' * len(parent_branch) + ' |')
            print(indentation + ' ' * len(parent_branch) + f' +-- Value: {value}')
            print(indentation + ' ' * len(parent_branch) + f' ' * len(' +-- Value: ') + f' +-- Outcome Split: Yes={yes_count}, No={no_count}')
            print_decision_tree(child_node, depth + 1, parent_branch=" " * len(parent_branch) + ' |')

# Feature names
feature_names = ['Outlook', 'Temperature', 'Humidity', 'Wind']


# Building the decision tree
decision_tree = build_decision_tree(dataset, feature_names)

print_decision_tree(decision_tree)
