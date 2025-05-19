package utils;

import java.util.ArrayList;
import java.util.List;

public class ResultsTable {
    private List<String> models;
    private List<Long> seeds;
    private List<Double> trainAccuracies;
    private List<Double> trainF1Scores;
    private List<Double> testAccuracies;
    private List<Double> testF1Scores;
    
    public ResultsTable() {
        models = new ArrayList<>();
        seeds = new ArrayList<>();
        trainAccuracies = new ArrayList<>();
        trainF1Scores = new ArrayList<>();
        testAccuracies = new ArrayList<>();
        testF1Scores = new ArrayList<>();
    }
    
    public void addResult(String model, long seed, 
                         double trainAccuracy, double trainF1, 
                         double testAccuracy, double testF1) {
        models.add(model);
        seeds.add(seed);
        trainAccuracies.add(trainAccuracy);
        trainF1Scores.add(trainF1);
        testAccuracies.add(testAccuracy);
        testF1Scores.add(testF1);
    }
    
    public void display() {
        System.out.println("\n---- Results Table ----");
        System.out.printf("%-20s %-10s %-15s %-15s %-15s %-15s\n", 
                         "Model", "Seed", "Train Acc", "Train F1", "Test Acc", "Test F1");
        System.out.println("--------------------------------------------------------------------------------");
        
        for (int i = 0; i < models.size(); i++) {
            System.out.printf("%-20s %-10d %-15.4f %-15.4f %-15.4f %-15.4f\n", 
                             models.get(i), seeds.get(i), 
                             trainAccuracies.get(i), trainF1Scores.get(i),
                             testAccuracies.get(i), testF1Scores.get(i));
        }
    }
    
    // Method to export results to a file for the report
    public void exportToFile(String filename) {
        try {
            java.io.PrintWriter writer = new java.io.PrintWriter(filename);
            writer.println("Model,Seed,Train Accuracy,Train F1,Test Accuracy,Test F1");
            
            for (int i = 0; i < models.size(); i++) {
                writer.printf("%s,%d,%.4f,%.4f,%.4f,%.4f\n",
                              models.get(i), seeds.get(i),
                              trainAccuracies.get(i), trainF1Scores.get(i),
                              testAccuracies.get(i), testF1Scores.get(i));
            }
            
            writer.close();
            System.out.println("Results exported to " + filename);
        } catch (Exception e) {
            System.err.println("Error exporting results: " + e.getMessage());
        }
    }
}