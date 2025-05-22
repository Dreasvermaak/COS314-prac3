package models;

import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import java.util.Random;

public class MultiLayerPerceptron {
    private long seed;
    private MultilayerPerceptron mlp;
    
    public MultiLayerPerceptron(long seed) {
        this.seed = seed;
        this.mlp = new MultilayerPerceptron();
        
        // Configure MLP
        try {
            mlp.setLearningRate(0.3);
            mlp.setMomentum(0.2);
            mlp.setTrainingTime(500);
            mlp.setHiddenLayers("a"); // a = (attribs + classes) / 2
            mlp.setSeed((int)seed);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    public void train(Object trainingData) {
        try {
            Instances data = (Instances)trainingData;
            mlp.buildClassifier(data);
            System.out.println("MLP model built successfully");
        } catch (Exception e) {
            System.err.println("Error training MLP: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    public double[] predict(Object testData) {
        try {
            Instances data = (Instances)testData;
            int numInstances = data.numInstances();
            double[] predictions = new double[numInstances];
            
            for (int i = 0; i < numInstances; i++) {
                predictions[i] = mlp.classifyInstance(data.instance(i));
            }
            
            return predictions;
        } catch (Exception e) {
            System.err.println("Error making predictions with MLP: " + e.getMessage());
            e.printStackTrace();
            return new double[0];
        }
    }
}