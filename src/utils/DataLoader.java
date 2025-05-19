package utils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class DataLoader {
    public static class Dataset {
        public double[][] features;
        public double[] labels;
        public String[] featureNames;
        
        public Dataset(double[][] features, double[] labels, String[] featureNames) {
            this.features = features;
            this.labels = labels;
            this.featureNames = featureNames;
        }
        
        public int numInstances() {
            return labels.length;
        }
        
        public int numFeatures() {
            return featureNames.length;
        }
    }
    
    public Object loadData(String filepath) {
        List<double[]> featuresList = new ArrayList<>();
        List<Double> labelsList = new ArrayList<>();
        String[] featureNames = null;
        
        try (BufferedReader reader = new BufferedReader(new FileReader(filepath))) {
            String line = reader.readLine();
            if (line != null) {
                // Parse header
                String[] header = line.split(",");
                featureNames = new String[header.length - 1]; // Exclude label column
                for (int i = 0; i < header.length - 1; i++) {
                    featureNames[i] = header[i];
                }
                
                // Parse data
                while ((line = reader.readLine()) != null) {
                    String[] values = line.split(",");
                    double[] features = new double[values.length - 1];
                    for (int i = 0; i < values.length - 1; i++) {
                        features[i] = Double.parseDouble(values[i]);
                    }
                    featuresList.add(features);
                    labelsList.add(Double.parseDouble(values[values.length - 1]));
                }
            }
            
            // Convert lists to arrays
            double[][] featuresArray = new double[featuresList.size()][];
            for (int i = 0; i < featuresList.size(); i++) {
                featuresArray[i] = featuresList.get(i);
            }
            
            double[] labelsArray = new double[labelsList.size()];
            for (int i = 0; i < labelsList.size(); i++) {
                labelsArray[i] = labelsList.get(i);
            }
            
            return new Dataset(featuresArray, labelsArray, featureNames);
        } catch (IOException e) {
            System.err.println("Error loading data: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }
}