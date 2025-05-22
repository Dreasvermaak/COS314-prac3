import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class GeneticProgramming {
    // Core GP parameters
    private long seed;
    private Random random;
    private int populationSize = 500;
    private int maxGenerations = 50;
    private double crossoverRate = 0.9;
    private double mutationRate = 0.1;
    private double reproductionRate = 0.0; // Calculated as 1 - crossoverRate - mutationRate
    private int tournamentSize = 7;
    private int maxInitialDepth = 6;
    private int maxDepth = 17;
    private double elitismPercentage = 0.05;
    
    // Program representation
    private TreeNode[] population;
    private double[] fitness;
    private TreeNode bestProgram;
    private double bestFitness;
    
    // Dataset metadata
    private int numFeatures;
    private String[] featureNames;
    
    // Function and terminal sets
    private enum FunctionType {ADD, SUBTRACT, MULTIPLY, DIVIDE, GT, LT, IF, AND, OR, NOT}
    private static final FunctionType[] FUNCTION_SET = FunctionType.values();
    
    public GeneticProgramming(long seed) {
        this.seed = seed;
        this.random = new Random(seed);
        
        // Calculate reproduction rate based on crossover and mutation rates
        this.reproductionRate = 1.0 - crossoverRate - mutationRate;
        if (this.reproductionRate < 0) {
            this.reproductionRate = 0.0;
            this.crossoverRate = 1.0 - mutationRate;
        }
    }
    
    // The TreeNode inner class remains the same as before, but we'll update the evaluate method
    private class TreeNode {
        private FunctionType function; // Null if terminal
        private double value;          // For constant terminals
        private int featureIndex;      // For feature terminals
        private boolean isConstant;    // Whether this terminal is a constant or feature
        private List<TreeNode> children;
        
        // Constructor for function nodes
        public TreeNode(FunctionType function) {
            this.function = function;
            this.children = new ArrayList<>();
            this.isConstant = false;
            
            // Add appropriate number of children based on function arity
            if (function != null) {
                int arity = getArity(function);
                for (int i = 0; i < arity; i++) {
                    this.children.add(null);
                }
            }
        }
        
        // Constructor for feature terminals
        public TreeNode(int featureIndex) {
            this.function = null;
            this.featureIndex = featureIndex;
            this.isConstant = false;
            this.children = new ArrayList<>();
        }
        
        // Constructor for constant terminals
        public TreeNode(double value) {
            this.function = null;
            this.value = value;
            this.isConstant = true;
            this.children = new ArrayList<>();
        }
        
        // Deep copy constructor
        public TreeNode(TreeNode other) {
            this.function = other.function;
            this.value = other.value;
            this.featureIndex = other.featureIndex;
            this.isConstant = other.isConstant;
            this.children = new ArrayList<>();
            
            for (TreeNode child : other.children) {
                if (child != null) {
                    this.children.add(new TreeNode(child));
                } else {
                    this.children.add(null);
                }
            }
        }
        
        // Check if this is a terminal node (no function)
        public boolean isTerminal() {
            return function == null;
        }
        
        // Check if this is a function node
        public boolean isFunction() {
            return function != null;
        }
        
        // Get the number of nodes in the tree (including this one)
        public int size() {
            int count = 1; // Count this node
            for (TreeNode child : children) {
                if (child != null) {
                    count += child.size();
                }
            }
            return count;
        }
        
        // Get the depth of the tree (1 for single node)
        public int depth() {
            if (children.isEmpty()) {
                return 1;
            }
            
            int maxChildDepth = 0;
            for (TreeNode child : children) {
                if (child != null) {
                    int childDepth = child.depth();
                    if (childDepth > maxChildDepth) {
                        maxChildDepth = childDepth;
                    }
                }
            }
            
            return maxChildDepth + 1;
        }
        
        // Updated to work with the new Dataset class
        public double evaluate(double[] features) {
            // Terminal node - return value
            if (isTerminal()) {
                if (isConstant) {
                    return value;
                } else {
                    return features[featureIndex];
                }
            }
            
            // Function node - evaluate children and apply function
            switch (function) {
                case ADD:
                    return children.get(0).evaluate(features) + children.get(1).evaluate(features);
                case SUBTRACT:
                    return children.get(0).evaluate(features) - children.get(1).evaluate(features);
                case MULTIPLY:
                    return children.get(0).evaluate(features) * children.get(1).evaluate(features);
                case DIVIDE:
                    double divisor = children.get(1).evaluate(features);
                    if (Math.abs(divisor) < 0.001) {
                        return 1.0; // Protected division
                    }
                    return children.get(0).evaluate(features) / divisor;
                case GT:
                    return children.get(0).evaluate(features) > children.get(1).evaluate(features) ? 1.0 : 0.0;
                case LT:
                    return children.get(0).evaluate(features) < children.get(1).evaluate(features) ? 1.0 : 0.0;
                case IF:
                    return children.get(0).evaluate(features) > 0 ? 
                           children.get(1).evaluate(features) : 
                           children.get(2).evaluate(features);
                case AND:
                    return (children.get(0).evaluate(features) > 0 && 
                            children.get(1).evaluate(features) > 0) ? 1.0 : 0.0;
                case OR:
                    return (children.get(0).evaluate(features) > 0 || 
                            children.get(1).evaluate(features) > 0) ? 1.0 : 0.0;
                case NOT:
                    return children.get(0).evaluate(features) > 0 ? 0.0 : 1.0;
                default:
                    throw new IllegalStateException("Unknown function type: " + function);
            }
        }
        
        @Override
        public String toString() {
            if (isTerminal()) {
                if (isConstant) {
                    return String.format("%.2f", value);
                } else {
                    return featureNames != null && featureIndex < featureNames.length ? 
                           featureNames[featureIndex] : "x" + featureIndex;
                }
            }
            
            StringBuilder sb = new StringBuilder();
            
            switch (function) {
                case ADD:
                    sb.append("(").append(children.get(0)).append(" + ").append(children.get(1)).append(")");
                    break;
                case SUBTRACT:
                    sb.append("(").append(children.get(0)).append(" - ").append(children.get(1)).append(")");
                    break;
                case MULTIPLY:
                    sb.append("(").append(children.get(0)).append(" * ").append(children.get(1)).append(")");
                    break;
                case DIVIDE:
                    sb.append("(").append(children.get(0)).append(" / ").append(children.get(1)).append(")");
                    break;
                case GT:
                    sb.append("(").append(children.get(0)).append(" > ").append(children.get(1)).append(")");
                    break;
                case LT:
                    sb.append("(").append(children.get(0)).append(" < ").append(children.get(1)).append(")");
                    break;
                case IF:
                    sb.append("if(").append(children.get(0)).append(") then ")
                      .append(children.get(1)).append(" else ").append(children.get(2));
                    break;
                case AND:
                    sb.append("(").append(children.get(0)).append(" AND ").append(children.get(1)).append(")");
                    break;
                case OR:
                    sb.append("(").append(children.get(0)).append(" OR ").append(children.get(1)).append(")");
                    break;
                case NOT:
                    sb.append("NOT(").append(children.get(0)).append(")");
                    break;
                default:
                    sb.append(function).append("(");
                    for (int i = 0; i < children.size(); i++) {
                        if (i > 0) {
                            sb.append(", ");
                        }
                        sb.append(children.get(i));
                    }
                    sb.append(")");
            }
            
            return sb.toString();
        }
    }
    
    private int getArity(FunctionType function) {
        switch (function) {
            case ADD:
            case SUBTRACT:
            case MULTIPLY:
            case DIVIDE:
            case GT:
            case LT:
            case AND:
            case OR:
                return 2;
            case NOT:
                return 1;
            case IF:
                return 3;
            default:
                throw new IllegalArgumentException("Unknown function: " + function);
        }
    }
    
    /**
     * Train the GP model on the given dataset
     */
    public void train(Object trainingData) {
        DataLoader.Dataset data = (DataLoader.Dataset) trainingData; 
        numFeatures = data.numFeatures();
        featureNames = data.featureNames;
        
        // Initialize population
        initializePopulation();
        
        // Evaluate initial population
        evaluatePopulation(data);
        
        // Track best individual
        updateBestProgram();
        
        System.out.println("Initial population created with size: " + populationSize);
        System.out.println("Best initial fitness: " + bestFitness);
        
        // Run evolution for maxGenerations
        for (int generation = 0; generation < maxGenerations; generation++) {
            // Create new population
            TreeNode[] newPopulation = new TreeNode[populationSize];
            
            // Elitism - copy best individuals directly
            int eliteCount = (int) (populationSize * elitismPercentage);
            int[] bestIndices = getBestIndices(eliteCount);
            for (int i = 0; i < eliteCount; i++) {
                newPopulation[i] = new TreeNode(population[bestIndices[i]]);
            }
            
            // Fill rest of population with offspring
            for (int i = eliteCount; i < populationSize; i++) {
                double p = random.nextDouble();
                
                if (p < crossoverRate) {
                    // Crossover
                    int parent1Index = tournamentSelection();
                    int parent2Index = tournamentSelection();
                    
                    // Ensure different parents
                    while (parent2Index == parent1Index) {
                        parent2Index = tournamentSelection();
                    }
                    
                    newPopulation[i] = crossover(population[parent1Index], population[parent2Index]);
                } else if (p < crossoverRate + mutationRate) {
                    // Mutation
                    int parentIndex = tournamentSelection();
                    newPopulation[i] = mutate(population[parentIndex]);
                } else {
                    // Reproduction
                    int parentIndex = tournamentSelection();
                    newPopulation[i] = new TreeNode(population[parentIndex]);
                }
            }
            
            // Replace old population
            population = newPopulation;
            
            // Evaluate new population
            evaluatePopulation(data);
            
            // Update best program
            updateBestProgram();
            
            // Log progress every 5 generations
            if (generation % 5 == 0 || generation == maxGenerations - 1) {
                System.out.printf("Generation %3d: Best fitness = %.4f, Avg fitness = %.4f, Best size = %d, Best depth = %d%n", 
                                 generation, bestFitness, getAverageFitness(), 
                                 bestProgram.size(), bestProgram.depth());
            }
        }
        
        System.out.println("Training completed.");
        System.out.println("Best program: " + bestProgram);
    }
    
    // The rest of the methods remain largely the same, just update calculateFitness and predict
    
    /**
     * Calculate the fitness of an individual program tree
     * For a classifier, this is the accuracy on the training set
     */
    private double calculateFitness(TreeNode program, DataLoader.Dataset data) { // Adjusted type
        int correct = 0;
        int total = data.numInstances();
        
        for (int i = 0; i < total; i++) {
            double prediction = program.evaluate(data.features[i]) > 0 ? 1.0 : 0.0; // Binary classification
            double actual = data.labels[i];
            
            if (prediction == actual) {
                correct++;
            }
        }
        
        double accuracy = (double) correct / total;
        
        // Add a small penalty for tree size to promote simplicity
        double sizePenalty = 0.0001 * program.size();
        return accuracy - sizePenalty;
    }
    
    /**
     * Evaluate the fitness of all individuals in the population
     */
    private void evaluatePopulation(DataLoader.Dataset data) { // Adjusted type
        for (int i = 0; i < populationSize; i++) {
            fitness[i] = calculateFitness(population[i], data);
        }
    }

    /**
     * Use the best evolved program to make predictions on new data
     */
    public double[] predict(Object testData) {
        // Adjust the cast as necessary
        DataLoader.Dataset data = (DataLoader.Dataset) testData;
        int numInstances = data.numInstances();
        double[] predictions = new double[numInstances];
        
        for (int i = 0; i < numInstances; i++) {
            // For binary classification: > 0 = class 1, <= 0 = class 0
            predictions[i] = bestProgram.evaluate(data.features[i]) > 0 ? 1.0 : 0.0;
        }
        
        return predictions;
    }
    
    // Add the missing methods from the previous implementation
    private void initializePopulation() {
        population = new TreeNode[populationSize];
        fitness = new double[populationSize];
        
        // Use ramped half-and-half initialization
        for (int i = 0; i < populationSize; i++) {
            int depth = 2 + random.nextInt(maxInitialDepth - 1); // Between 2 and maxInitialDepth
            
            // Half grow, half full
            if (i % 2 == 0) {
                population[i] = generateGrowTree(depth);
            } else {
                population[i] = generateFullTree(depth);
            }
        }
    }
    
    private TreeNode generateGrowTree(int maxDepth) {
        if (maxDepth <= 1 || random.nextDouble() < 0.3) {
            // Create terminal node
            return createTerminalNode();
        } else {
            // Create function node
            FunctionType function = FUNCTION_SET[random.nextInt(FUNCTION_SET.length)];
            TreeNode node = new TreeNode(function);
            
            // Create children
            for (int i = 0; i < node.children.size(); i++) {
                node.children.set(i, generateGrowTree(maxDepth - 1));
            }
            
            return node;
        }
    }
    
    private TreeNode generateFullTree(int depth) {
        if (depth <= 1) {
            // Create terminal node
            return createTerminalNode();
        } else {
            // Create function node
            FunctionType function = FUNCTION_SET[random.nextInt(FUNCTION_SET.length)];
            TreeNode node = new TreeNode(function);
            
            // Create children
            for (int i = 0; i < node.children.size(); i++) {
                node.children.set(i, generateFullTree(depth - 1));
            }
            
            return node;
        }
    }
    
    private TreeNode createTerminalNode() {
        if (random.nextDouble() < 0.7) {
            // Feature terminal (70% probability)
            return new TreeNode(random.nextInt(numFeatures));
        } else {
            // Constant terminal (30% probability)
            return new TreeNode(random.nextDouble() * 10 - 5); // Value between -5 and 5
        }
    }
    
    private int tournamentSelection() {
        int best = random.nextInt(populationSize);
        double bestFit = fitness[best];
        
        for (int i = 1; i < tournamentSize; i++) {
            int idx = random.nextInt(populationSize);
            if (fitness[idx] > bestFit) {
                best = idx;
                bestFit = fitness[idx];
            }
        }
        
        return best;
    }
    
    private TreeNode crossover(TreeNode parent1, TreeNode parent2) {
        // Create copies of parents
        TreeNode offspring1 = new TreeNode(parent1);
        TreeNode parent2Copy = new TreeNode(parent2);
        
        // Get random node from each parent
        List<TreeNode> nodesParent1 = getAllNodes(offspring1);
        List<TreeNode> nodesParent2 = getAllNodes(parent2Copy);
        
        if (nodesParent1.isEmpty() || nodesParent2.isEmpty()) {
            return offspring1; // Fallback
        }
        
        TreeNode nodeParent1 = nodesParent1.get(random.nextInt(nodesParent1.size()));
        
        // Bias toward function nodes for parent2 (if available)
        List<TreeNode> functionNodes = new ArrayList<>();
        for (TreeNode node : nodesParent2) {
            if (node.isFunction()) {
                functionNodes.add(node);
            }
        }
        
        TreeNode nodeParent2;
        if (!functionNodes.isEmpty() && random.nextDouble() < 0.9) {
            nodeParent2 = functionNodes.get(random.nextInt(functionNodes.size()));
        } else {
            nodeParent2 = nodesParent2.get(random.nextInt(nodesParent2.size()));
        }
        
        // Swap subtrees
        swapSubtrees(offspring1, nodeParent1, nodeParent2);
        
        // Check if new tree exceeds max depth, if so return parent1
        if (offspring1.depth() > maxDepth) {
            return new TreeNode(parent1);
        }
        
        return offspring1;
    }
    
    private List<TreeNode> getAllNodes(TreeNode root) {
        List<TreeNode> nodes = new ArrayList<>();
        collectNodes(root, nodes);
        return nodes;
    }
    
    private void collectNodes(TreeNode node, List<TreeNode> nodes) {
        if (node == null) return;
        
        nodes.add(node);
        for (TreeNode child : node.children) {
            if (child != null) {
                collectNodes(child, nodes);
            }
        }
    }
    
    private void swapSubtrees(TreeNode tree, TreeNode nodeToReplace, TreeNode replacement) {
        // For primitive object values
        nodeToReplace.function = replacement.function;
        nodeToReplace.value = replacement.value;
        nodeToReplace.featureIndex = replacement.featureIndex;
        nodeToReplace.isConstant = replacement.isConstant;
        
        // Clear children
        nodeToReplace.children.clear();
        
        // Add clones of replacement's children
        for (TreeNode child : replacement.children) {
            if (child != null) {
                nodeToReplace.children.add(new TreeNode(child));
            } else {
                nodeToReplace.children.add(null);
            }
        }
    }
    
    private TreeNode mutate(TreeNode parent) {
        TreeNode offspring = new TreeNode(parent);
        List<TreeNode> nodes = getAllNodes(offspring);
        
        if (nodes.isEmpty()) {
            return offspring; // Fallback
        }
        
        // Select a random node to mutate
        TreeNode nodeToMutate = nodes.get(random.nextInt(nodes.size()));
        
        // Different mutation strategies based on node type
        if (random.nextDouble() < 0.5 && nodeToMutate.isFunction()) {
            // Function mutation - change function while keeping arity
            List<FunctionType> compatibleFunctions = new ArrayList<>();
            for (FunctionType func : FUNCTION_SET) {
                if (getArity(func) == getArity(nodeToMutate.function)) {
                    compatibleFunctions.add(func);
                }
            }
            
            if (!compatibleFunctions.isEmpty()) {
                nodeToMutate.function = compatibleFunctions.get(random.nextInt(compatibleFunctions.size()));
            }
        } else {
            // Subtree mutation - replace with random subtree
            int depth = 1 + random.nextInt(3); // Small random subtree (depth 1-3)
            TreeNode newSubtree = generateGrowTree(depth);
            
            swapSubtrees(offspring, nodeToMutate, newSubtree);
            
            // Check if new tree exceeds max depth, if so return unchanged parent
            if (offspring.depth() > maxDepth) {
                return new TreeNode(parent);
            }
        }
        
        return offspring;
    }
    
    private void updateBestProgram() {
        int bestIndex = 0;
        for (int i = 1; i < populationSize; i++) {
            if (fitness[i] > fitness[bestIndex]) {
                bestIndex = i;
            }
        }
        
        if (bestProgram == null || fitness[bestIndex] > bestFitness) {
            bestProgram = new TreeNode(population[bestIndex]);
            bestFitness = fitness[bestIndex];
        }
    }
    
    private int[] getBestIndices(int n) {
        int[] indices = new int[n];
        double[] tempFitness = Arrays.copyOf(fitness, fitness.length);
        
        for (int i = 0; i < n; i++) {
            int bestIdx = 0;
            for (int j = 1; j < populationSize; j++) {
                if (tempFitness[j] > tempFitness[bestIdx]) {
                    bestIdx = j;
                }
            }
            
            indices[i] = bestIdx;
            tempFitness[bestIdx] = Double.NEGATIVE_INFINITY; // Mark as used
        }
        
        return indices;
    }
    
    private double getAverageFitness() {
        double sum = 0;
        for (double f : fitness) {
            sum += f;
        }
        return sum / populationSize;
    }
    
    /**
     * Get the string representation of the best program
     */
    public String getBestProgram() {
        return bestProgram != null ? bestProgram.toString() : "No program evolved yet";
    }
    
    /**
     * Get the best fitness achieved
     */
    public double getBestFitness() {
        return bestFitness;
    }
}