# Compilation Instructions

## Prerequisites
- Java JDK 8 or higher installed
- Weka library (included in the lib folder)

## Compiling
1. Navigate to the project root directory
2. Compile all files with the Weka library in the classpath:
   javac -cp "lib/*" -d bin src/*.java src/models/*.java src/utils/*.java

## Creating JAR
1. Create a manifest file (manifest.txt) with:
   Main-Class: Main
   Class-Path: lib/weka.jar

2. Create the JAR file:
   jar cfm StockPredictor.jar manifest.txt -C bin .

## Running the JAR
   java -jar StockPredictor.jar