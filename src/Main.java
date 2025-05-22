import java.util.Scanner;
import models.*;
import utils.*;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        // Get seed value and filepath
        long seed = 12345;
        System.out.print("seed value: " + seed );
        // long seed = scanner.nextLong();
        // scanner.nextLine(); // Consume newline
        
        String trainingPath = "Euro_USD_Stock/BTC_train.csv";
        System.out.print("Training data filepath: " + trainingPath );
        // String trainingPath = scanner.nextLine();
        
        String testPath = "Euro_USD_Stock/BTC_test.csv";
        System.out.print("Test data filepath: nter "+ testPath );
        // String testPath = scanner.nextLine();
        
        // Load data
        DataLoader dataLoader = new DataLoader();
        Object trainingData = dataLoader.loadData(trainingPath);
        Object testData = dataLoader.loadData(testPath);
        
        // Initialize models with the same seed
        GeneticProgramming gp = new GeneticProgramming(seed);
        MultiLayerPerceptron mlp = new MultiLayerPerceptron(seed);
        DecisionTree dt = new DecisionTree(seed);
        
        // Train models
        System.out.println("Training Genetic Programming model...");
        gp.train(trainingData);
        
        System.out.println("Training Multi-Layer Perceptron model...");
        mlp.train(trainingData);
        
        System.out.println("Training Decision Tree model...");
        dt.train(trainingData);
        
        // Evaluate models
        Evaluator evaluator = new Evaluator();
        
        // Results for GP
        double[] gpTrainResults = evaluator.evaluate(gp, trainingData);
        double[] gpTestResults = evaluator.evaluate(gp, testData);
        
        // Results for MLP
        double[] mlpTrainResults = evaluator.evaluate(mlp, trainingData);
        double[] mlpTestResults = evaluator.evaluate(mlp, testData);
        
        // Results for DT
        double[] dtTrainResults = evaluator.evaluate(dt, trainingData);
        double[] dtTestResults = evaluator.evaluate(dt, testData);
        
        // Display results
        ResultsTable resultsTable = new ResultsTable();
        resultsTable.addResult("Genetic Programming", seed, gpTrainResults[0], gpTrainResults[1], 
                               gpTestResults[0], gpTestResults[1]);
        resultsTable.addResult("MLP", seed, mlpTrainResults[0], mlpTrainResults[1], 
                               mlpTestResults[0], mlpTestResults[1]);
        resultsTable.addResult("Decision Tree", seed, dtTrainResults[0], dtTrainResults[1], 
                               dtTestResults[0], dtTestResults[1]);
        
        resultsTable.display();
        
        // Run statistical tests
        System.out.println("\nWilcoxon Signed-Rank Test Results (GP vs MLP):");
        evaluator.wilcoxonTest(gpTestResults, mlpTestResults);
        
        scanner.close();
    }
}