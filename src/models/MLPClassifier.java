package models;
import java.io.*;
import java.nio.file.*;
import java.util.*;

public class MLPClassifier {
    private long seed;
    private String trainingDataPath;
    
    public MLPClassifier(long seed) {
        this.seed = seed;
        System.out.println("MLPClassifier initialized with seed: " + seed);
    }

    public void train(Object data) {
        try {
            // Convert Weka Instances to CSV file
            this.trainingDataPath = "mlp_train_data.csv";
            writeInstancesToCSV(data, this.trainingDataPath);
            System.out.println("MLPClassifier training data prepared: " + this.trainingDataPath);
        } catch (Exception e) {
            System.err.println("Error preparing training data: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private void writeInstancesToCSV(Object data, String filename) throws Exception {
        // Use reflection to handle Weka Instances without importing weka classes
        Class<?> instancesClass = data.getClass();
        
        // Get number of instances and attributes
        int numInstances = (Integer) instancesClass.getMethod("numInstances").invoke(data);
        int numAttributes = (Integer) instancesClass.getMethod("numAttributes").invoke(data);
        
        PrintWriter writer = new PrintWriter(filename);
        
        // Write header
        StringBuilder header = new StringBuilder();
        for (int i = 0; i < numAttributes; i++) {
            Object attribute = instancesClass.getMethod("attribute", int.class).invoke(data, i);
            String attrName = (String) attribute.getClass().getMethod("name").invoke(attribute);
            header.append(attrName);
            if (i < numAttributes - 1) header.append(",");
        }
        writer.println(header.toString());
        
        // Write data
        for (int i = 0; i < numInstances; i++) {
            Object instance = instancesClass.getMethod("instance", int.class).invoke(data, i);
            StringBuilder row = new StringBuilder();
            for (int j = 0; j < numAttributes; j++) {
                double value = (Double) instance.getClass().getMethod("value", int.class).invoke(instance, j);
                row.append(value);
                if (j < numAttributes - 1) row.append(",");
            }
            writer.println(row.toString());
        }
        writer.close();
    }
    
    private static String checkPythonEnvironment() {
        String[] pythonCommands = {"python", "python3", "py"};
        
        for (String cmd : pythonCommands) {
            try {
                ProcessBuilder pb = new ProcessBuilder(cmd, "--version");
                Process process = pb.start();
                BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
                String line = reader.readLine();
                int exitCode = process.waitFor();
                
                if (exitCode == 0 && line != null && line.toLowerCase().contains("python")) {
                    System.out.println("Found Python: " + line);
                    return cmd;
                }
            } catch (Exception e) {
                // Continue trying other commands
            }
        }
        
        return null;
    }
    
    private static boolean checkRequiredLibraries(String pythonCommand) {
        String checkLibrariesScript = 
            "try:\n" +
            "    import numpy\n" +
            "    import pandas\n" +
            "    from sklearn.neural_network import MLPClassifier\n" +
            "    print('All required libraries are installed')\n" +
            "    exit(0)\n" +
            "except ImportError as e:\n" +
            "    print('Missing library:', e)\n" +
            "    exit(1)\n";
        
        try {
            Files.write(Paths.get("check_libraries.py"), checkLibrariesScript.getBytes());
            
            ProcessBuilder pb = new ProcessBuilder(pythonCommand, "check_libraries.py");
            Process process = pb.start();
            
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
            
            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println(line);
            }
            
            while ((line = errorReader.readLine()) != null) {
                System.out.println("Error: " + line);
            }
            
            int exitCode = process.waitFor();
            return exitCode == 0;
            
        } catch (Exception e) {
            System.out.println("Error checking libraries: " + e.getMessage());
            return false;
        }
    }
    
    private static void installRequiredLibraries(String pythonCommand) {
        try {
            System.out.println("Attempting to install required Python libraries...");
            
            String[] pipCommands = {pythonCommand + " -m pip", "pip", "pip3"};
            boolean installed = false;
            
            for (String pipCmd : pipCommands) {
                if (installed) break;
                
                try {
                    System.out.println("Trying to install with: " + pipCmd);
                    ProcessBuilder pb = new ProcessBuilder();
                    
                    String[] cmdParts;
                    if (pipCmd.contains(" ")) {
                        cmdParts = pipCmd.split(" ");
                    } else {
                        cmdParts = new String[]{pipCmd};
                    }
                    
                    List<String> command = new ArrayList<>();
                    Collections.addAll(command, cmdParts);
                    Collections.addAll(command, "install", "numpy", "pandas", "scikit-learn");
                    
                    pb.command(command);
                    pb.inheritIO();
                    
                    Process process = pb.start();
                    int exitCode = process.waitFor();
                    
                    if (exitCode == 0) {
                        System.out.println("Libraries installed successfully!");
                        installed = true;
                    }
                } catch (Exception e) {
                    System.out.println("Error with " + pipCmd + ": " + e.getMessage());
                }
            }
            
            if (!installed) {
                System.out.println("\nFailed to install libraries automatically.");
                System.out.println("Please install the required libraries manually using:");
                System.out.println(pythonCommand + " -m pip install numpy pandas scikit-learn");
                System.out.println("Then run this program again.");
                System.exit(1);
            }
            
        } catch (Exception e) {
            System.out.println("Error installing libraries: " + e.getMessage());
            System.exit(1);
        }
    }
    
    private static void createPythonScript() throws IOException {
        String pythonCode = """
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
                
                print("\\nMLP Configuration:")
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
                
                print("\\nTraining MLP classifier...")
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
                        f.write(str(pred) + "\\n")
                
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                conf_matrix = confusion_matrix(y_test, y_pred)
                class_report = classification_report(y_test, y_pred)
                
                print("\\n===== MLP Classification Results =====")
                print("Accuracy: {:.4f}".format(accuracy))
                print("F1 Score: {:.4f}".format(f1))
                print("\\nConfusion Matrix:")
                print(conf_matrix)
                print("\\nClassification Report:")
                print(class_report)
                
                with open("mlp_results.txt", "w") as f:
                    f.write("===== MLP Classification Results =====\\n")
                    f.write("Seed: {}\\n".format(seed))
                    f.write("Training file: {}\\n".format(train_file))
                    f.write("Test file: {}\\n".format(test_file))
                    f.write("\\nMLP Configuration:\\n")
                    f.write("Hidden layer sizes: {}\\n".format(hidden_layers))
                    f.write("Activation function: {}\\n".format(activation))
                    f.write("Solver: {}\\n".format(solver))
                    f.write("Alpha (L2 penalty): {}\\n".format(alpha))
                    f.write("Batch size: {}\\n".format(batch_size))
                    f.write("Learning rate: {}\\n".format(learning_rate))
                    f.write("Maximum iterations: {}\\n".format(max_iter))
                    f.write("\\nAccuracy: {:.4f}\\n".format(accuracy))
                    f.write("F1 Score: {:.4f}\\n".format(f1))
                    f.write("\\nConfusion Matrix:\\n")
                    f.write(str(conf_matrix))
                    f.write("\\n\\nClassification Report:\\n")
                    f.write(class_report)
                
                print("Results saved to mlp_results.txt")
                print("Predictions saved to mlp_predictions.txt")
                
                return accuracy, f1
            
            if __name__ == "__main__":
                main()
            """;
        
        Files.write(Paths.get("mlp_script.py"), pythonCode.getBytes());
        System.out.println("Python script created: mlp_script.py");
    }

    public double[] predict(Object data) {
        try {
            if (this.trainingDataPath == null) {
                throw new RuntimeException("Model not trained yet. Call train() first.");
            }
            
            String testCsv = "mlp_test_data.csv";
            writeInstancesToCSV(data, testCsv);

            createPythonScript();
            String pythonCmd = checkPythonEnvironment();
            if (pythonCmd == null) throw new RuntimeException("Python not found");
            if (!checkRequiredLibraries(pythonCmd)) installRequiredLibraries(pythonCmd);

            ProcessBuilder pb = new ProcessBuilder(pythonCmd, "mlp_script.py", 
                                                  String.valueOf(this.seed), 
                                                  this.trainingDataPath, 
                                                  testCsv);
            pb.inheritIO();
            Process process = pb.start();
            int exitCode = process.waitFor();
            if (exitCode != 0) throw new RuntimeException("Python MLP script failed");

            List<Double> predictionsList = new ArrayList<>();
            BufferedReader reader = new BufferedReader(new FileReader("mlp_predictions.txt"));
            String line;
            while ((line = reader.readLine()) != null) {
                predictionsList.add(Double.parseDouble(line.trim()));
            }
            reader.close();
            
            double[] predictions = new double[predictionsList.size()];
            for (int i = 0; i < predictionsList.size(); i++) {
                predictions[i] = predictionsList.get(i);
            }
            
            System.out.println("MLPClassifier predictions loaded: " + predictions.length + " predictions");
            return predictions;
            
        } catch (Exception e) {
            System.err.println("Error in MLPClassifier.predict: " + e.getMessage());
            e.printStackTrace();
            return new double[0];
        }
    }
}