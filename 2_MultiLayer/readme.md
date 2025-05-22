
##java MLPClassifier <seed> <train_file_path> <test_file_path>
---------------
#First run 
---------------
dreas@DESKTOP-V5A0C20:/mnt/c/Users/Client/Documents/GitHub/COS314-prac3/2_MultiLayer$ java MLPClassifier 42 BTC_train.csv BTC_test.csv     
Using seed: 42
Training file: BTC_train.csv
Test file: BTC_test.csv

Checking Python environment...
Found Python: Python 3.8.10
Python script created: mlp_script.py
All required libraries are installed

Running MLP classifier using Python libraries...
Using seed: 42
Training file: BTC_train.csv
Test file: BTC_test.csv
Loading data...
Training data shape: (998, 6)
Test data shape: (263, 6)
Training class distribution: [470 528]
Test class distribution: [130 133]
Standardizing features...

MLP Configuration:
Hidden layer sizes: (10, 5)
Activation function: relu
Solver: adam
Alpha (L2 penalty): 0.0001
Batch size: auto
Learning rate: adaptive
Maximum iterations: 1000

Training MLP classifier...
/home/dreas/.local/lib/python3.8/site-packages/sklearn/neural_network/_multilayer_perceptron.py:691: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (1000) reached and the optimization hasn't converged yet.
  warnings.warn(
Making predictions...

===== MLP Classification Results =====
Accuracy: 0.9544
F1 Score: 0.9544

Confusion Matrix:
[[122   8]
 [  4 129]]

Classification Report:
              precision    recall  f1-score   support

           0       0.97      0.94      0.95       130
           1       0.94      0.97      0.96       133

    accuracy                           0.95       263
   macro avg       0.95      0.95      0.95       263
weighted avg       0.95      0.95      0.95       263

Results saved to mlp_results.txt

MLP classification completed successfully.

Results from MLP classification:
----------------------------
===== MLP Classification Results =====
Seed: 42
Training file: BTC_train.csv
Test file: BTC_test.csv

MLP Configuration:
Hidden layer sizes: (10, 5)
Activation function: relu
Solver: adam
Alpha (L2 penalty): 0.0001
Batch size: auto
Learning rate: adaptive
Maximum iterations: 1000

Accuracy: 0.9544
F1 Score: 0.9544

Confusion Matrix:
[[122   8]
 [  4 129]]

Classification Report:
              precision    recall  f1-score   support

           0       0.97      0.94      0.95       130
           1       0.94      0.97      0.96       133

    accuracy                           0.95       263
   macro avg       0.95      0.95      0.95       263
weighted avg       0.95      0.95      0.95       263
dreas@DESKTOP-V5A0C20:/mnt/c/Users/Client/Documents/GitHub/COS314-prac3/2_MultiLayer$ 

---------------------------------------------------------------------------------------------------------------

---------------
#second run 
---------------

dreas@DESKTOP-V5A0C20:/mnt/c/Users/Client/Documents/GitHub/COS314-prac3/2_MultiLayer$ java MLPClassifier 123 BTC_train.csv BTC_test.csv    
Using seed: 123
Training file: BTC_train.csv
Test file: BTC_test.csv

Checking Python environment...
Found Python: Python 3.8.10
Python script created: mlp_script.py
All required libraries are installed

Running MLP classifier using Python libraries...
Using seed: 123
Training file: BTC_train.csv
Test file: BTC_test.csv
Loading data...
Training data shape: (998, 6)
Test data shape: (263, 6)
Training class distribution: [470 528]
Test class distribution: [130 133]
Standardizing features...

MLP Configuration:
Hidden layer sizes: (10, 5)
Activation function: relu
Solver: adam
Alpha (L2 penalty): 0.0001
Batch size: auto
Learning rate: adaptive
Maximum iterations: 1000

Training MLP classifier...
Making predictions...
/home/dreas/.local/lib/python3.8/site-packages/sklearn/metrics/_classification.py:1471: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/home/dreas/.local/lib/python3.8/site-packages/sklearn/metrics/_classification.py:1471: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/home/dreas/.local/lib/python3.8/site-packages/sklearn/metrics/_classification.py:1471: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))

===== MLP Classification Results =====
Accuracy: 0.5057
F1 Score: 0.3397

Confusion Matrix:
[[  0 130]
 [  0 133]]

Classification Report:
              precision    recall  f1-score   support

           0       0.00      0.00      0.00       130
           1       0.51      1.00      0.67       133

    accuracy                           0.51       263
   macro avg       0.25      0.50      0.34       263
weighted avg       0.26      0.51      0.34       263

Results saved to mlp_results.txt

MLP classification completed successfully.

Results from MLP classification:
----------------------------
===== MLP Classification Results =====
Seed: 123
Training file: BTC_train.csv
Test file: BTC_test.csv

MLP Configuration:
Hidden layer sizes: (10, 5)
Activation function: relu
Solver: adam
Alpha (L2 penalty): 0.0001
Batch size: auto
Learning rate: adaptive
Maximum iterations: 1000

Accuracy: 0.5057
F1 Score: 0.3397

Confusion Matrix:
[[  0 130]
 [  0 133]]

Classification Report:
              precision    recall  f1-score   support

           0       0.00      0.00      0.00       130
           1       0.51      1.00      0.67       133

    accuracy                           0.51       263
   macro avg       0.25      0.50      0.34       263
weighted avg       0.26      0.51      0.34       263
dreas@DESKTOP-V5A0C20:/mnt/c/Users/Client/Documents/GitHub/COS314-prac3/2_MultiLayer$ 