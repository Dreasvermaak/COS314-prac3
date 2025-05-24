import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import random
import sys

def main():
    if len(sys.argv) < 4:
        print("Usage: python mlp_script.py <seed> <train_file_path> <test_file_path>")
        return

    seed = int(sys.argv[1])
    train_file = sys.argv[2]
    test_file = sys.argv[3]

    print("Using seed: {}".format(seed))
    print("Training file: {}".format(train_file))
    print("Test file: {}".format(test_file))

    random.seed(seed)
    np.random.seed(seed)

    try:
        print("Loading data...")
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)

        print("Training data shape: {}".format(train_data.shape))
        print("Test data shape: {}".format(test_data.shape))

    except Exception as e:
        print("Error loading data: {}".format(e))
        return

    X_train = train_data.drop('Output', axis=1)
    y_train = train_data['Output']
    X_test = test_data.drop('Output', axis=1)
    y_test = test_data['Output']

    print("Training class distribution: {}".format(np.bincount(y_train)))
    print("Test class distribution: {}".format(np.bincount(y_test)))

    print("Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nMLP Configuration:")
    hidden_layers = (10, 5)
    activation = 'relu'
    solver = 'adam'
    alpha = 0.0001
    batch_size = 'auto'
    learning_rate = 'adaptive'
    max_iter = 1000

    print("Hidden layer sizes: {}".format(hidden_layers))
    print("Activation function: {}".format(activation))
    print("Solver: {}".format(solver))
    print("Alpha (L2 penalty): {}".format(alpha))
    print("Batch size: {}".format(batch_size))
    print("Learning rate: {}".format(learning_rate))
    print("Maximum iterations: {}".format(max_iter))

    print("\nTraining MLP classifier...")
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        solver=solver,
        alpha=alpha,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=seed
    )

    mlp.fit(X_train_scaled, y_train)

    print("Making predictions...")
    y_pred = mlp.predict(X_test_scaled)

    with open("mlp_predictions.txt", "w") as f:
        for pred in y_pred:
            f.write(str(pred) + "\n")

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print("\n===== MLP Classification Results =====")
    print("Accuracy: {:.4f}".format(accuracy))
    print("F1 Score: {:.4f}".format(f1))
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)

    with open("mlp_results.txt", "w") as f:
        f.write("===== MLP Classification Results =====\n")
        f.write("Seed: {}\n".format(seed))
        f.write("Training file: {}\n".format(train_file))
        f.write("Test file: {}\n".format(test_file))
        f.write("\nMLP Configuration:\n")
        f.write("Hidden layer sizes: {}\n".format(hidden_layers))
        f.write("Activation function: {}\n".format(activation))
        f.write("Solver: {}\n".format(solver))
        f.write("Alpha (L2 penalty): {}\n".format(alpha))
        f.write("Batch size: {}\n".format(batch_size))
        f.write("Learning rate: {}\n".format(learning_rate))
        f.write("Maximum iterations: {}\n".format(max_iter))
        f.write("\nAccuracy: {:.4f}\n".format(accuracy))
        f.write("F1 Score: {:.4f}\n".format(f1))
        f.write("\nConfusion Matrix:\n")
        f.write(str(conf_matrix))
        f.write("\n\nClassification Report:\n")
        f.write(class_report)

    print("Results saved to mlp_results.txt")
    print("Predictions saved to mlp_predictions.txt")

    return accuracy, f1

if __name__ == "__main__":
    main()
