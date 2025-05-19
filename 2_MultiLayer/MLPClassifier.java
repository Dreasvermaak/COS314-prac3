import java.io.*;
import java.nio.file.*;
import java.util.*;

public class MLPClassifier {
    
    public static void main(String[] args) {
        try {
            // Get seed and filepaths from command line or user input
            Scanner scanner = new Scanner(System.in);
            int seed;
            String trainFile;
            String testFile;
            
            if (args.length < 3) {
                System.out.print("Enter seed for reproducibility: ");
                seed = scanner.nextInt();
                scanner.nextLine(); // Consume newline
                
                System.out.print("Enter path to training data file: ");
                trainFile = scanner.nextLine();
                
                System.out.print("Enter path to test data file: ");
                testFile = scanner.nextLine();
            } else {
                seed = Integer.parseInt(args[0]);
                trainFile = args[1];
                testFile = args[2];
            }
            
            System.out.println("Using seed: " + seed);
            System.out.println("Training file: " + trainFile);
            System.out.println("Test file: " + testFile);
            
            // Check if files exist
            if (!Files.exists(Paths.get(trainFile)) || !Files.exists(Paths.get(testFile))) {
                System.out.println("Error: One or both file paths are invalid.");
                return;
            }
            
            // Check Python and required libraries
            System.out.println("\nChecking Python environment...");
            String pythonCommand = checkPythonEnvironment();
            if (pythonCommand == null) {
                System.out.println("Error: Python is not installed or not in PATH.");
                System.out.println("Please install Python and try again.");
                return;
            }
            
            // Create the Python script file
            createPythonScript();
            
            // Check if required libraries are installed
            boolean librariesInstalled = checkRequiredLibraries(pythonCommand);
            if (!librariesInstalled) {
                System.out.println("\nInstalling required Python libraries...");
                installRequiredLibraries(pythonCommand);
            }
            
            // Run the Python script with the provided parameters
            System.out.println("\nRunning MLP classifier using Python libraries...");
            ProcessBuilder pb = new ProcessBuilder(pythonCommand, "mlp_script.py", 
                                                   String.valueOf(seed), trainFile, testFile);
            pb.inheritIO(); // Redirect Python output to Java console
            Process process = pb.start();
            int exitCode = process.waitFor();
            
            if (exitCode == 0) {
                System.out.println("\nMLP classification completed successfully.");
                
                // Read and display the results
                if (Files.exists(Paths.get("mlp_results.txt"))) {
                    System.out.println("\nResults from MLP classification:");
                    System.out.println("----------------------------");
                    Files.lines(Paths.get("mlp_results.txt")).forEach(System.out::println);
                }
            } else {
                System.out.println("\nError running MLP classifier. Exit code: " + exitCode);
            }
            
        } catch (Exception e) {
            e.printStackTrace();
        }
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
            "try:\\n" +
            "    import numpy\\n" +
            "    import pandas\\n" +
            "    from sklearn.neural_network import MLPClassifier\\n" +
            "    print('All required libraries are installed')\\n" +
            "    exit(0)\\n" +
            "except ImportError as e:\\n" +
            "    print('Missing library:', e)\\n" +
            "    exit(1)\\n";
        
        try {
            // Create the check script
            Files.write(Paths.get("check_libraries.py"), checkLibrariesScript.getBytes());
            
            // Run the check script
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
            
            // Try pip first
            String[] pipCommands = {pythonCommand + " -m pip", "pip", "pip3"};
            boolean installed = false;
            
            for (String pipCmd : pipCommands) {
                if (installed) break;
                
                try {
                    System.out.println("Trying to install with: " + pipCmd);
                    ProcessBuilder pb = new ProcessBuilder();
                    
                    // Split the command into parts
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
                # Get seed and filepaths from command line arguments
                if len(sys.argv) < 4:
                    print("Usage: python mlp_script.py <seed> <train_file_path> <test_file_path>")
                    return
                
                seed = int(sys.argv[1])
                train_file = sys.argv[2]
                test_file = sys.argv[3]
                
                print("Using seed: {}".format(seed))
                print("Training file: {}".format(train_file))
                print("Test file: {}".format(test_file))
                
                # Set random seed for reproducibility
                random.seed(seed)
                np.random.seed(seed)
                
                # Load data
                try:
                    print("Loading data...")
                    train_data = pd.read_csv(train_file)
                    test_data = pd.read_csv(test_file)
                    
                    print("Training data shape: {}".format(train_data.shape))
                    print("Test data shape: {}".format(test_data.shape))
                    
                except Exception as e:
                    print("Error loading data: {}".format(e))
                    return
                
                # Separate features and target
                X_train = train_data.drop('Output', axis=1)
                y_train = train_data['Output']
                X_test = test_data.drop('Output', axis=1)
                y_test = test_data['Output']
                
                # Print class distribution
                print("Training class distribution: {}".format(np.bincount(y_train)))
                print("Test class distribution: {}".format(np.bincount(y_test)))
                
                # Standardize the features
                print("Standardizing features...")
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Create and train the MLP classifier
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
                
                # Calculate metrics
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
                
                # Save the results to a file
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
                
                return accuracy, f1
            
            if __name__ == "__main__":
                main()
            """;
        
        Files.write(Paths.get("mlp_script.py"), pythonCode.getBytes());
        System.out.println("Python script created: mlp_script.py");
    
        
        Files.write(Paths.get("mlp_script.py"), pythonCode.getBytes());
        System.out.println("Python script created: mlp_script.py");
    }
}