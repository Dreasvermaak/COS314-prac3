# Multi-Layer Perceptron Implementation

This implementation uses a Java program to interface with Python's scikit-learn library for running an MLP classifier on Bitcoin price data.

## Requirements

- Java Development Kit (JDK) 11 or higher
- Python 3.6 or higher
- Python libraries:
  - numpy
  - pandas
  - scikit-learn

## How to Install Requirements

1. Make sure Java is installed on your system. You can check by running:
   ```
   java -version
   ```

2. Make sure Python is installed on your system. You can check by running:
   ```
   python --version
   ```

3. Install the required Python libraries:
   ```
   pip install numpy pandas scikit-learn
   ```

## How to Compile

Compile the Java file using the following command:

```
javac MLPClassifier.java
```

## How to Run

You can run the program in two ways:

### Method 1: With command-line arguments

```
java MLPClassifier <seed> <train_file_path> <test_file_path>
```

Example:
```
java MLPClassifier 42 BTC_train.csv BTC_test.csv
```

### Method 2: With interactive prompts

Simply run:
```
java MLPClassifier
```

The program will prompt you to enter:
1. A seed value for reproducibility
2. The path to the training data file
3. The path to the testing data file

## How It Works

1. The Java program (`MLPClassifier.java`) creates a Python script (`mlp_script.py`) that implements the MLP classifier using scikit-learn.
2. It then executes this Python script with the provided parameters (seed, training file, and test file).
3. The Python script trains an MLP classifier on the Bitcoin price data and makes predictions.
4. The results (accuracy, F1 score, confusion matrix, etc.) are printed to the console and saved to a file called `mlp_results.txt`.

## MLP Configuration

The MLP classifier is configured with the following parameters:
- Hidden layer sizes: (10, 5) (two hidden layers with 10 and 5 neurons)
- Activation function: ReLU
- Solver: Adam optimizer
- Alpha (L2 penalty): 0.0001
- Batch size: auto
- Learning rate: adaptive
- Maximum iterations: 1000

## Creating a JAR File

To create an executable JAR file:

```
jar cfe MLPClassifier.jar MLPClassifier MLPClassifier.class
```

Then run it using:

```
java -jar MLPClassifier.jar <seed> <train_file_path> <test_file_path>
```

## Notes

- The program expects the CSV files to have a column named 'Output' containing the target values (0 or 1).
- The features are standardized (mean=0, std=1) before training the model.
- The random seed ensures reproducibility of the results.