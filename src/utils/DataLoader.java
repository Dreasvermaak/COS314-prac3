package utils;

import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ArffLoader;
import java.io.File;

public class DataLoader {
    
    public Object loadData(String filepath) {
        try {
            File file = new File(filepath);
            
            if (filepath.toLowerCase().endsWith(".csv")) {
                CSVLoader loader = new CSVLoader();
                loader.setSource(file);
                Instances data = loader.getDataSet();
                
                // Set class index to last attribute
                if (data.classIndex() == -1) {
                    data.setClassIndex(data.numAttributes() - 1);
                }
                
                return data;
            } else if (filepath.toLowerCase().endsWith(".arff")) {
                ArffLoader loader = new ArffLoader();
                loader.setFile(file);
                Instances data = loader.getDataSet();
                
                // Set class index to last attribute
                if (data.classIndex() == -1) {
                    data.setClassIndex(data.numAttributes() - 1);
                }
                
                return data;
            } else {
                System.err.println("Unsupported file format: " + filepath);
                return null;
            }
        } catch (Exception e) {
            System.err.println("Error loading data: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }
}