package utils;

import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ArffLoader;
import java.io.File;
import java.io.FileInputStream;

public class DataLoader {
    
    public Object loadData(String filepath) {
        try {
            File file = new File(filepath);
            
            // Check if file exists
            if (!file.exists()) {
                System.err.println("File does not exist: " + filepath);
                return null;
            }
            
            System.out.println("Attempting to load file: " + file.getAbsolutePath());
            
            if (filepath.toLowerCase().endsWith(".csv")) {
                CSVLoader loader = new CSVLoader();
                
                // Try different approaches for CSV loading
                try {
                    // Method 1: Use FileInputStream
                    FileInputStream fis = new FileInputStream(file);
                    loader.setSource(fis);
                    Instances data = loader.getDataSet();
                    fis.close();
                    
                    // Set class index to last attribute
                    if (data.classIndex() == -1) {
                        data.setClassIndex(data.numAttributes() - 1);
                    }
                    
                    System.out.println("Successfully loaded CSV data: " + data.numInstances() + " instances, " + 
                                     data.numAttributes() + " attributes");
                    System.out.println("Class attribute: " + data.classAttribute().name());
                    return data;
                    
                } catch (Exception e1) {
                    System.out.println("Method 1 failed, trying Method 2...");
                    
                    // Method 2: Use setFile
                    try {
                        loader = new CSVLoader();
                        loader.setFile(file);
                        Instances data = loader.getDataSet();
                        
                        // Set class index to last attribute
                        if (data.classIndex() == -1) {
                            data.setClassIndex(data.numAttributes() - 1);
                        }
                        
                        System.out.println("Successfully loaded CSV data: " + data.numInstances() + " instances, " + 
                                         data.numAttributes() + " attributes");
                        System.out.println("Class attribute: " + data.classAttribute().name());
                        return data;
                        
                    } catch (Exception e2) {
                        System.err.println("Both CSV loading methods failed.");
                        System.err.println("Method 1 error: " + e1.getMessage());
                        System.err.println("Method 2 error: " + e2.getMessage());
                        return null;
                    }
                }
                
            } else if (filepath.toLowerCase().endsWith(".arff")) {
                ArffLoader loader = new ArffLoader();
                loader.setFile(file);
                Instances data = loader.getDataSet();
                
                // Set class index to last attribute
                if (data.classIndex() == -1) {
                    data.setClassIndex(data.numAttributes() - 1);
                }
                
                System.out.println("Successfully loaded ARFF data: " + data.numInstances() + " instances, " + 
                                 data.numAttributes() + " attributes");
                return data;
            } else {
                System.err.println("Unsupported file format: " + filepath);
                return null;
            }
        } catch (Exception e) {
            System.err.println("Error loading data from " + filepath + ": " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }
}