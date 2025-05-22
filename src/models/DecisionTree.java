package models;

import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Discretize;

public class DecisionTree {
    private long seed;
    private J48 j48;
    private Instances convertedTrainingData;
    private Discretize discretizer;
    
    public DecisionTree(long seed) {
        this.seed = seed;
        this.j48 = new J48();
        
        // Configure J48 for optimal performance with discretized data
        try {
            String options = "-U -M 2";  // Unpruned tree, min 2 instances per leaf
            j48.setOptions(weka.core.Utils.splitOptions(options));
        } catch (Exception e) {
            System.err.println("Error configuring J48: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    public void train(Object trainingData) {
        try {
            Instances data = (Instances)trainingData;
            
            // Apply discretization to handle normalized continuous data
            discretizer = new Discretize();
            discretizer.setBins(20); // 20 bins for optimal granularity
            discretizer.setUseEqualFrequency(true); // Equal frequency binning
            discretizer.setInputFormat(data);
            Instances discretizedData = Filter.useFilter(data, discretizer);
            
            // Convert the class attribute from numeric to nominal
            NumericToNominal convert = new NumericToNominal();
            String[] options = new String[2];
            options[0] = "-R";
            options[1] = Integer.toString(discretizedData.classIndex()+1);
            convert.setOptions(options);
            convert.setInputFormat(discretizedData);
            Instances convertedData = Filter.useFilter(discretizedData, convert);
            
            // Store the converted data for prediction
            this.convertedTrainingData = new Instances(convertedData);
            
            // Build the classifier
            j48.buildClassifier(convertedData);
            
            System.out.println("J48 model built successfully");
            System.out.println("Decision Tree Size: " + j48.measureTreeSize());
            System.out.println("Number of Leaves: " + j48.measureNumLeaves());
            
        } catch (Exception e) {
            System.err.println("Error training J48: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    public double[] predict(Object testData) {
        try {
            Instances data = (Instances)testData;
            
            // Apply the same discretization to test data
            Instances discretizedData = Filter.useFilter(data, discretizer);
            
            // Convert test data to match the training data format
            NumericToNominal convert = new NumericToNominal();
            String[] options = new String[2];
            options[0] = "-R";
            options[1] = Integer.toString(discretizedData.classIndex()+1);
            convert.setOptions(options);
            convert.setInputFormat(discretizedData);
            Instances convertedData = Filter.useFilter(discretizedData, convert);
            
            int numInstances = convertedData.numInstances();
            double[] predictions = new double[numInstances];
            
            for (int i = 0; i < numInstances; i++) {
                predictions[i] = j48.classifyInstance(convertedData.instance(i));
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