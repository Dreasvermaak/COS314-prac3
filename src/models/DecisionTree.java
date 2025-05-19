package models;

import weka.classifiers.trees.J48;
import weka.core.Instances;

public class DecisionTree {
    private long seed;
    private J48 j48;
    
    public DecisionTree(long seed) {
        this.seed = seed;
        this.j48 = new J48();
        
        // Configure J48
        try {
            j48.setOptions(weka.core.Utils.splitOptions("-C 0.25 -M 2"));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    public void train(Object trainingData) {
        try {
            Instances data = (Instances)trainingData;
            j48.buildClassifier(data);
            System.out.println("J48 model built successfully");
        } catch (Exception e) {
            System.err.println("Error training J48: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    public double[] predict(Object testData) {
        try {
            Instances data = (Instances)testData;
            int numInstances = data.numInstances();
            double[] predictions = new double[numInstances];
            
            for (int i = 0; i < numInstances; i++) {
                predictions[i] = j48.classifyInstance(data.instance(i));
            }
            
            return predictions;
        } catch (Exception e) {
            System.err.println("Error making predictions with J48: " + e.getMessage());
            e.printStackTrace();
            return new double[0];
        }
    }
    
    public String getTreeStructure() {
        return j48.toString();
    }
}