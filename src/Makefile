# Makefile for COS314-prac3 Java project

JAVAC=javac
JAVA=java
SRC_DIR=.
BIN_DIR=../bin
LIB_DIR=../lib
WEKA_JAR=$(LIB_DIR)/weka.jar
CLASSPATH=$(SRC_DIR):$(BIN_DIR):$(WEKA_JAR)
JAVA_FILES=$(shell find $(SRC_DIR) -name '*.java')
CLASS_FILES=$(shell find $(SRC_DIR) -name '*.class')

.PHONY: all run clean

all: compile

compile:
	$(JAVAC) -cp "$(WEKA_JAR)" -d $(BIN_DIR) $(JAVA_FILES)

run: all
	$(JAVA) -cp "$(CLASSPATH)" --add-opens java.base/java.lang=ALL-UNNAMED Main

clean:
	find $(SRC_DIR) -name '*.class' -delete
	find $(BIN_DIR) -name '*.class' -delete
