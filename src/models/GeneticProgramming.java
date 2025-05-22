package models;

import weka.core.Instances;
import weka.core.Instance;
import java.util.*;

public class GeneticProgramming {
    private Random random;
    private long seed;
    private int populationSize = 50;
    private int maxGenerations = 30;
    private int maxDepth = 4;
    private double crossoverRate = 0.7;
    private double mutationRate = 0.2;
    private double elitismRate = 0.05;
    
    private Individual bestIndividual;
    private Instances trainingData;
    
    private enum NodeType {
        FEATURE, CONSTANT, ADD, SUB, MUL, GT, LT
    }
    
    private static class Node {
        NodeType type;
        double value;
        int featureIndex;
        Node left, right;
        
        Node(NodeType type) {
            this.type = type;
        }
        
        Node(NodeType type, double value) {
            this.type = type;
            this.value = value;
        }
        
        Node(NodeType type, int featureIndex) {
            this.type = type;
            this.featureIndex = featureIndex;
        }
        
        public Node copy() {
            Node copy = new Node(this.type);
            copy.value = this.value;
            copy.featureIndex = this.featureIndex;
            if (this.left != null) copy.left = this.left.copy();
            if (this.right != null) copy.right = this.right.copy();
            return copy;
        }
        
        public int getDepth() {
            if (left == null && right == null) return 1;
            int leftDepth = (left != null) ? left.getDepth() : 0;
            int rightDepth = (right != null) ? right.getDepth() : 0;
            return 1 + Math.max(leftDepth, rightDepth);
        }
    }
    
    private static class Individual {
        Node root;
        double fitness;
        boolean evaluated;
        
        Individual(Node root) {
            this.root = root;
            this.fitness = 0.0;
            this.evaluated = false;
        }
        
        public Individual copy() {
            Individual copy = new Individual(this.root.copy());
            copy.fitness = this.fitness;
            copy.evaluated = this.evaluated;
            return copy;
        }
    }
    
    public GeneticProgramming(long seed) {
        this.seed = seed;
        this.random = new Random(seed);
        System.out.println("GP initialized with seed: " + seed);
    }
    
    public void train(Object data) {
        this.trainingData = (Instances) data;
        
        System.out.println("GP Training on " + trainingData.numInstances() + " instances");
        System.out.println("Features: " + (trainingData.numAttributes() - 1));
        
        // Check class balance
        int class0 = 0, class1 = 0;
        for (int i = 0; i < trainingData.numInstances(); i++) {
            double label = trainingData.instance(i).classValue();
            if (label == 0.0) class0++;
            else class1++;
        }
        System.out.println("Class distribution: 0=" + class0 + ", 1=" + class1);
        
        // Initialize population with diversity
        List<Individual> population = initializeDiversePopulation();
        
        double lastBestFitness = 0.0;
        int stagnationCount = 0;
        
        // Evolution loop
        for (int generation = 0; generation < maxGenerations; generation++) {
            // Evaluate fitness
            evaluatePopulation(population);
            
            // Sort by fitness (higher is better)
            population.sort((a, b) -> Double.compare(b.fitness, a.fitness));
            
            // Track best individual
            if (bestIndividual == null || population.get(0).fitness > bestIndividual.fitness) {
                bestIndividual = population.get(0).copy();
            }
            
            // Check for stagnation
            if (Math.abs(population.get(0).fitness - lastBestFitness) < 0.001) {
                stagnationCount++;
            } else {
                stagnationCount = 0;
            }
            lastBestFitness = population.get(0).fitness;
            
            // Print progress
            if (generation % 5 == 0 || generation == maxGenerations - 1) {
                System.out.printf("Gen %d: Best=%.4f, Avg=%.4f, Worst=%.4f\n", 
                    generation, population.get(0).fitness, 
                    population.stream().mapToDouble(i -> i.fitness).average().orElse(0.0),
                    population.get(population.size()-1).fitness);
            }
            
            // Early stopping if stagnant
            if (stagnationCount > 10) {
                System.out.println("Early stopping due to stagnation at generation " + generation);
                break;
            }
            
            // Prevent perfect overfitting
            if (population.get(0).fitness > 0.95) {
                System.out.println("Warning: Very high training fitness detected, adding diversity");
                // Add random individuals to prevent overfitting
                for (int i = population.size() - 10; i < population.size(); i++) {
                    population.set(i, new Individual(createRandomTree(0)));
                }
            }
            
            // Create next generation
            population = createNextGeneration(population);
        }
        
        System.out.printf("GP training completed. Final best fitness: %.4f\n", bestIndividual.fitness);
    }
    
    private List<Individual> initializeDiversePopulation() {
        List<Individual> population = new ArrayList<>();
        
        // Create diverse initial population
        for (int i = 0; i < populationSize; i++) {
            Node root;
            if (i % 4 == 0) {
                // Simple feature comparisons
                root = createSimpleComparison();
            } else if (i % 4 == 1) {
                // Feature combinations
                root = createFeatureCombination();
            } else {
                // Random trees
                root = createRandomTree(0);
            }
            population.add(new Individual(root));
        }
        
        return population;
    }
    
    private Node createSimpleComparison() {
        NodeType[] comparisons = {NodeType.GT, NodeType.LT};
        NodeType comp = comparisons[random.nextInt(comparisons.length)];
        
        Node node = new Node(comp);
        node.left = new Node(NodeType.FEATURE, random.nextInt(trainingData.numAttributes() - 1));
        node.right = new Node(NodeType.CONSTANT, random.nextGaussian() * 100);
        
        return node;
    }
    
    private Node createFeatureCombination() {
        NodeType[] ops = {NodeType.ADD, NodeType.SUB, NodeType.MUL};
        NodeType op = ops[random.nextInt(ops.length)];
        
        Node opNode = new Node(op);
        opNode.left = new Node(NodeType.FEATURE, random.nextInt(trainingData.numAttributes() - 1));
        opNode.right = new Node(NodeType.FEATURE, random.nextInt(trainingData.numAttributes() - 1));
        
        Node compNode = new Node(random.nextBoolean() ? NodeType.GT : NodeType.LT);
        compNode.left = opNode;
        compNode.right = new Node(NodeType.CONSTANT, random.nextGaussian() * 50);
        
        return compNode;
    }
    
    private Node createRandomTree(int depth) {
        if (depth >= maxDepth || (depth > 1 && random.nextDouble() < 0.4)) {
            // Terminal node
            if (random.nextBoolean()) {
                return new Node(NodeType.FEATURE, random.nextInt(trainingData.numAttributes() - 1));
            } else {
                return new Node(NodeType.CONSTANT, random.nextGaussian() * 100);
            }
        } else {
            // Function node
            NodeType[] functions = {NodeType.ADD, NodeType.SUB, NodeType.MUL, NodeType.GT, NodeType.LT};
            NodeType function = functions[random.nextInt(functions.length)];
            
            Node node = new Node(function);
            node.left = createRandomTree(depth + 1);
            node.right = createRandomTree(depth + 1);
            
            return node;
        }
    }
    
    private void evaluatePopulation(List<Individual> population) {
        for (Individual individual : population) {
            if (!individual.evaluated) {
                individual.fitness = calculateFitness(individual);
                individual.evaluated = true;
            }
        }
    }
    
    private double calculateFitness(Individual individual) {
        if (individual.root == null) return 0.0;
        
        int correct = 0;
        int total = trainingData.numInstances();
        
        for (int i = 0; i < total; i++) {
            Instance inst = trainingData.instance(i);
            double[] features = new double[trainingData.numAttributes() - 1];
            for (int j = 0; j < trainingData.numAttributes() - 1; j++) {
                features[j] = inst.value(j);
            }
            double actualLabel = inst.classValue();
            
            try {
                double rawOutput = evaluateTree(individual.root, features);
                double predictedLabel = rawOutput > 0 ? 1.0 : 0.0;
                
                if (predictedLabel == actualLabel) {
                    correct++;
                }
                
            } catch (Exception e) {
                return 0.0; // Penalize invalid trees
            }
        }
        
        double accuracy = (double) correct / total;
        
        // Penalize overly complex trees to prevent overfitting
        int treeSize = getTreeSize(individual.root);
        double complexityPenalty = Math.min(treeSize / 50.0, 0.1);
        
        // Final fitness with complexity penalty
        double fitness = accuracy - complexityPenalty;
        
        return Math.max(0.0, fitness);
    }
    
    private int getTreeSize(Node node) {
        if (node == null) return 0;
        return 1 + getTreeSize(node.left) + getTreeSize(node.right);
    }
    
    private double evaluateTree(Node node, double[] features) {
        if (node == null) return 0.0;
        
        switch (node.type) {
            case FEATURE:
                if (node.featureIndex >= 0 && node.featureIndex < features.length) {
                    double val = features[node.featureIndex];
                    if (Double.isNaN(val) || Double.isInfinite(val)) return 0.0;
                    return val;
                }
                return 0.0;
                
            case CONSTANT:
                if (Double.isNaN(node.value) || Double.isInfinite(node.value)) return 0.0;
                return node.value;
                
            case ADD:
                double addResult = evaluateTree(node.left, features) + evaluateTree(node.right, features);
                return Double.isFinite(addResult) ? addResult : 0.0;
                
            case SUB:
                double subResult = evaluateTree(node.left, features) - evaluateTree(node.right, features);
                return Double.isFinite(subResult) ? subResult : 0.0;
                
            case MUL:
                double left = evaluateTree(node.left, features);
                double right = evaluateTree(node.right, features);
                double mulResult = left * right;
                return Double.isFinite(mulResult) ? mulResult : 0.0;
                
            case GT:
                return evaluateTree(node.left, features) > evaluateTree(node.right, features) ? 1.0 : -1.0;
                
            case LT:
                return evaluateTree(node.left, features) < evaluateTree(node.right, features) ? 1.0 : -1.0;
                
            default:
                return 0.0;
        }
    }
    
    private List<Individual> createNextGeneration(List<Individual> population) {
        List<Individual> nextGeneration = new ArrayList<>();
        
        // Elitism
        int eliteCount = Math.max(1, (int) (populationSize * elitismRate));
        for (int i = 0; i < eliteCount; i++) {
            nextGeneration.add(population.get(i).copy());
        }
        
        // Fill rest with crossover and mutation
        while (nextGeneration.size() < populationSize) {
            if (random.nextDouble() < crossoverRate && nextGeneration.size() < populationSize - 1) {
                Individual parent1 = tournamentSelection(population);
                Individual parent2 = tournamentSelection(population);
                Individual[] offspring = crossover(parent1, parent2);
                
                if (random.nextDouble() < mutationRate) {
                    mutate(offspring[0]);
                }
                if (random.nextDouble() < mutationRate) {
                    mutate(offspring[1]);
                }
                
                nextGeneration.add(offspring[0]);
                if (nextGeneration.size() < populationSize) {
                    nextGeneration.add(offspring[1]);
                }
            } else {
                Individual parent = tournamentSelection(population);
                Individual offspring = parent.copy();
                mutate(offspring);
                nextGeneration.add(offspring);
            }
        }
        
        return nextGeneration;
    }
    
    private Individual tournamentSelection(List<Individual> population) {
        int tournamentSize = 3;
        Individual best = population.get(random.nextInt(population.size()));
        
        for (int i = 1; i < tournamentSize; i++) {
            Individual candidate = population.get(random.nextInt(population.size()));
            if (candidate.fitness > best.fitness) {
                best = candidate;
            }
        }
        
        return best;
    }
    
    private Individual[] crossover(Individual parent1, Individual parent2) {
        Individual offspring1 = parent1.copy();
        Individual offspring2 = parent2.copy();
        
        // Mark as not evaluated since they're modified
        offspring1.evaluated = false;
        offspring2.evaluated = false;
        
        List<Node> nodes1 = new ArrayList<>();
        List<Node> nodes2 = new ArrayList<>();
        collectNodes(offspring1.root, nodes1);
        collectNodes(offspring2.root, nodes2);
        
        if (!nodes1.isEmpty() && !nodes2.isEmpty()) {
            Node crossPoint1 = nodes1.get(random.nextInt(nodes1.size()));
            Node crossPoint2 = nodes2.get(random.nextInt(nodes2.size()));
            
            // Limit tree growth
            if (crossPoint1.getDepth() + crossPoint2.getDepth() < maxDepth * 2) {
                Node temp = crossPoint1.copy();
                replaceNode(offspring1.root, crossPoint1, crossPoint2.copy());
                replaceNode(offspring2.root, crossPoint2, temp);
            }
        }
        
        return new Individual[]{offspring1, offspring2};
    }
    
    private void collectNodes(Node node, List<Node> nodes) {
        if (node == null) return;
        nodes.add(node);
        collectNodes(node.left, nodes);
        collectNodes(node.right, nodes);
    }
    
    private boolean replaceNode(Node root, Node target, Node replacement) {
        if (root == null) return false;
        
        if (root.left == target) {
            root.left = replacement;
            return true;
        }
        if (root.right == target) {
            root.right = replacement;
            return true;
        }
        
        return replaceNode(root.left, target, replacement) || 
               replaceNode(root.right, target, replacement);
    }
    
    private void mutate(Individual individual) {
        individual.evaluated = false; // Mark as needing re-evaluation
        
        List<Node> nodes = new ArrayList<>();
        collectNodes(individual.root, nodes);
        
        if (!nodes.isEmpty()) {
            Node mutationPoint = nodes.get(random.nextInt(nodes.size()));
            
            if (random.nextDouble() < 0.5) {
                // Point mutation
                if (mutationPoint.type == NodeType.CONSTANT) {
                    mutationPoint.value += random.nextGaussian() * 10;
                } else if (mutationPoint.type == NodeType.FEATURE) {
                    mutationPoint.featureIndex = random.nextInt(trainingData.numAttributes() - 1);
                }
            } else {
                // Subtree mutation
                Node newSubtree = createRandomTree(0);
                replaceNode(individual.root, mutationPoint, newSubtree);
            }
        }
    }
    
    public double[] predict(Object data) {
        Instances testData = (Instances) data;
        double[] predictions = new double[testData.numInstances()];
        if (bestIndividual == null) {
            System.err.println("Model not trained yet!");
            return predictions;
        }
        for (int i = 0; i < testData.numInstances(); i++) {
            Instance inst = testData.instance(i);
            double[] features = new double[testData.numAttributes() - 1];
            for (int j = 0; j < testData.numAttributes() - 1; j++) {
                features[j] = inst.value(j);
            }
            try {
                double rawPrediction = evaluateTree(bestIndividual.root, features);
                predictions[i] = rawPrediction > 0 ? 1.0 : 0.0;
            } catch (Exception e) {
                predictions[i] = 0.0;
            }
        }
        return predictions;
    }
}