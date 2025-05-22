package utils;

import weka.core.Instances;
import java.util.Arrays;

public class Evaluator {
    
    // Evaluate a model and return [accuracy, F1Score]
    public double[] evaluate(Object model, Object data) {
        try {
            Instances instances = (Instances)data;
            double[] predictions = null;
            
            // Get predictions based on model type
            if (model instanceof models.GeneticProgramming) {
                predictions = ((models.GeneticProgramming)model).predict(data);
            } else if (model instanceof models.MLPClassifier) {
                predictions = ((models.MLPClassifier)model).predict(data);
            } else if (model instanceof models.DecisionTree) {
                predictions = ((models.DecisionTree)model).predict(data);
            }
            
            // Calculate metrics
            int correct = 0;
            int truePositives = 0;
            int falsePositives = 0;
            int falseNegatives = 0;
            
            for (int i = 0; i < instances.numInstances(); i++) {
                double actual = instances.instance(i).classValue();
                double predicted = predictions[i];
                
                // Accuracy
                if (predicted == actual) {
                    correct++;
                }
                
                // For F1 Score (assuming binary classification)
                if (predicted == 1.0 && actual == 1.0) {
                    truePositives++;
                } else if (predicted == 1.0 && actual == 0.0) {
                    falsePositives++;
                } else if (predicted == 0.0 && actual == 1.0) {
                    falseNegatives++;
                }
            }
            
            double accuracy = (double) correct / instances.numInstances();
            
            // Calculate F1 Score
            double precision = (truePositives == 0) ? 0 : 
                              (double) truePositives / (truePositives + falsePositives);
            double recall = (truePositives == 0) ? 0 : 
                           (double) truePositives / (truePositives + falseNegatives);
            double f1Score = (precision + recall == 0) ? 0 : 
                            2 * (precision * recall) / (precision + recall);
            
            return new double[] {accuracy, f1Score};
        } catch (Exception e) {
            System.err.println("Error in evaluation: " + e.getMessage());
            e.printStackTrace();
            return new double[] {0.0, 0.0};
        }
    }
    
    // Implement Wilcoxon signed-rank test between two models
    public void wilcoxonTest(double[] resultsModel1, double[] resultsModel2) {
        // Basic implementation of Wilcoxon signed-rank test
        // Note: For a proper test, you'd need multiple runs with different seeds
        
        System.out.println("Model 1 results: " + Arrays.toString(resultsModel1));
        System.out.println("Model 2 results: " + Arrays.toString(resultsModel2));
        
        // Calculate differences
        double diff = resultsModel1[0] - resultsModel2[0]; // Using accuracy
        
        if (diff > 0) {
            System.out.println("Model 1 performs better by " + diff);
        } else if (diff < 0) {
            System.out.println("Model 2 performs better by " + Math.abs(diff));
        } else {
            System.out.println("Both models perform equally");
        }
        
        // Note: For a proper Wilcoxon test, you need more samples
        System.out.println("Note: A proper Wilcoxon test requires multiple runs. " +
                         "Please refer to the report for the full statistical analysis.");
    }
}
