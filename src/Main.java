import java.util.Scanner;
import models.*;
import utils.*;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        // Get seed value and filepath
        System.out.print("Enter seed value: ");
        long seed = scanner.nextLong();
        scanner.nextLine(); // Consume newline
        
        System.out.print("Enter training data filepath: ");
        String trainingPath = scanner.nextLine();
        
        System.out.print("Enter test data filepath: ");
        String testPath = scanner.nextLine();
        
        // Load data
        DataLoader dataLoader = new DataLoader();
        Object trainingData = dataLoader.loadData(trainingPath);
        Object testData = dataLoader.loadData(testPath);
        
        // Check if data loading was successful
        if (trainingData == null) {
            System.err.println("Failed to load training data. Exiting.");
            scanner.close();
            return;
        }
        
        if (testData == null) {
            System.err.println("Failed to load test data. Exiting.");
            scanner.close();
            return;
        }
        
        // Initialize models with the same seed
        //GeneticProgramming gp = new GeneticProgramming(seed);
        //MultiLayerPerceptron mlp = new MultiLayerPerceptron(seed);
        DecisionTree dt = new DecisionTree(seed);
        
        // Train models
        //System.out.println("Training Genetic Programming model...");
        //gp.train(trainingData);
        
        //System.out.println("Training Multi-Layer Perceptron model...");
        //mlp.train(trainingData);
        
        System.out.println("Training Decision Tree model...");
        dt.train(trainingData);
        
        // Evaluate models
        Evaluator evaluator = new Evaluator();
        
        // Results for GP
        //double[] gpTrainResults = evaluator.evaluate(gp, trainingData);
        //double[] gpTestResults = evaluator.evaluate(gp, testData);
        
        // Results for MLP
        // double[] mlpTrainResults = evaluator.evaluate(mlp, trainingData);
       // double[] mlpTestResults = evaluator.evaluate(mlp, testData);
        
        // Results for DT
        double[] dtTrainResults = evaluator.evaluate(dt, trainingData);
        double[] dtTestResults = evaluator.evaluate(dt, testData);
        
        // Display results
        System.out.println("\n---- Results ----");
        System.out.println("Decision Tree Training - Accuracy: " + String.format("%.4f", dtTrainResults[0]) + 
                          ", F1: " + String.format("%.4f", dtTrainResults[1]));
        System.out.println("Decision Tree Testing - Accuracy: " + String.format("%.4f", dtTestResults[0]) + 
                          ", F1: " + String.format("%.4f", dtTestResults[1]));
        
        // ResultsTable resultsTable = new ResultsTable();
        // resultsTable.addResult("Genetic Programming", seed, gpTrainResults[0], gpTrainResults[1], 
        //                       gpTestResults[0], gpTestResults[1]);
        // resultsTable.addResult("MLP", seed, mlpTrainResults[0], mlpTrainResults[1], 
        //                       mlpTestResults[0], mlpTestResults[1]);
        // resultsTable.addResult("Decision Tree", seed, dtTrainResults[0], dtTrainResults[1], 
        //                       dtTestResults[0], dtTestResults[1]);
        
        //resultsTable.display();
        
        // Run statistical tests
        //System.out.println("\nWilcoxon Signed-Rank Test Results (GP vs MLP):");
        //evaluator.wilcoxonTest(gpTestResults, mlpTestResults);
        
        scanner.close();
    }
}