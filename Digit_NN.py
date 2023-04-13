''' NEURAL NETWORK FOR DIGIT DETECTION
This program is a shallow Neural Network that is trained to recognize digits written in a 5x3 box
'''
import random
import math
import csv

# Hyperparameters:

# speed (magnitude) at which algorithm adjusts weights
LEARNING_RATE = 0.3

# Feature –> individual and independent variables that measure a property or characteristic
# AKA input size (input is 3x5 box thus input size is 15)
FEATURE_SIZE = 15

# number of nodes in hidden layer
HIDDEN_SIZE = 12

# Class –> output variables in a classification model are referred to as classes (or labels)
# AKA output size (number of digits from 0-9 is 10 thus output size is 10)
CLASS_SIZE = 10

# range of random intiialized values centered around 0
INITIALIZATION_RANGE = 0.4

# number of iterations over all
NUM_ITERATIONS = 1000

# initializing weights from the trainingData and Validation .csv files using the csv library
trainingDataPath = 'NumberClassifierNN/TrainingData.csv'
with open(trainingDataPath, newline='') as f:
    reader = csv.reader(f)
    trainingData = [tuple(row) for row in reader]

validationDatapath = 'NumberClassifierNN/ValidationData.csv'
with open(validationDatapath, newline='') as f:
    reader = csv.reader(f)
    validationData = [tuple(row) for row in reader]

# fill weights with random numbers from -0.2 to 0.2
def initializeWeights(IKweights, KHweights):
    for i in range(FEATURE_SIZE):
        currentNodeWeights = []
        for j in range(HIDDEN_SIZE):
            currentNodeWeights.append(
                random.random()*INITIALIZATION_RANGE - INITIALIZATION_RANGE/2)
        IKweights.append(currentNodeWeights)

    for i in range(HIDDEN_SIZE):
        currentNodeWeights = []
        for j in range(CLASS_SIZE):
            currentNodeWeights.append(
                random.random()*INITIALIZATION_RANGE - INITIALIZATION_RANGE/2)
        KHweights.append(currentNodeWeights)

    return IKweights, KHweights

# set the input nodes for a given training input
def setInputNodes(trainingExample):
    inputNodes = []
    currentExample = trainingExample[0]
    for i in range(FEATURE_SIZE):
        inputNodes.append(currentExample[i])
    return inputNodes

# getting values of all nodes in j layer using the sum of previous i layer and weights from i to j
# xⱼ = ∑(xᵢ * wᵢ)
# actual operation is an implementation of dot product of matrices
def sumWeights(prevLayer, weights, currentLayerSize):
    currentLayer = []
    for i in range(currentLayerSize):
        sum = 0
        for j in range(len(prevLayer)):
            sum += float(prevLayer[j])*weights[j][i]
        currentLayer.append(sum)
    return currentLayer

# sigmoid activation function that "squishes" the node values to be between 0 and 1
# used to allow slow and steady learning
def sigmoidFunction(nodes):
    for i in range(len(nodes)):
        power = pow(math.e, -1*nodes[i])
        nodes[i] = 1/(1+power)
    return nodes

# main forward propogation function
def forwardPropogation(currentExample, inputWeights, outputWeights):
    inputNodes = setInputNodes(currentExample)
    hiddenNodes = sumWeights(inputNodes, inputWeights, HIDDEN_SIZE)
    hiddenNodes = sigmoidFunction(hiddenNodes)

    outputNodes = sumWeights(hiddenNodes, outputWeights, CLASS_SIZE)
    outputNodes = sigmoidFunction(outputNodes)
    return inputNodes, outputNodes, hiddenNodes

# find the error for each output node: σ(h)
def outputLayerError(trainingExample, outputNodes):
    error = []
    for i in range(len(outputNodes)):
        expectedOutputString = trainingExample[1]
        expectedOutput = int(expectedOutputString[i])
        actualOutput = outputNodes[i]
        error.append(actualOutput * (1 - actualOutput)
                     * (expectedOutput - actualOutput))
    return error

# find the error for each hidden node: σ(j)
def hiddenLayerError(hiddenNodes, outputNodes, outputWeights, outputError):
    error = []
    for i in range(len(hiddenNodes)):
        alpha = 0
        # get the value of alpha (calculated using all the output nodes that are connected from hidden node k: ⍺ = Σ(outputWeights * outputError))
        for j in range(len(outputNodes)):
            alpha += outputWeights[i][j]*outputError[j]

        actualOutput = hiddenNodes[i]
        error.append(actualOutput * (1 - actualOutput) * alpha)
    return error

# adjust each weight between i and j layers using learning rate, error from j layer, and node value from i layer
def adjustWeights(learningRate, error, weights, prevNode):
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            deltaWeight = learningRate * error[j] * float(prevNode[i])
            weights[i][j] = weights[i][j] + deltaWeight
    return weights

# main backwards propogation function
def backwardsPropogation(currentExample, inputNodes, output, outputWeights, hiddenNodes, hiddenWeights):
    # calulate error of final, output layer using the expected value
    outputError = outputLayerError(currentExample, output)

    # adjust weights from hidden layer to output layer using output layer error and values from hidden layer nodes
    outputWeights = adjustWeights(
        LEARNING_RATE, outputError, outputWeights, hiddenNodes)

    # calculate error of hidden layer using output error
    hiddenError = hiddenLayerError(
        hiddenNodes, output, outputWeights, outputError)

    # adjust weights from input layer to hiddne layer using hidden layer error and values from input layer nodes
    hiddenWeights = adjustWeights(
        LEARNING_RATE, hiddenError, hiddenWeights, inputNodes)

    return hiddenWeights, outputWeights


def main():
    # 1d arrays that hold the value of the nodes
    hiddenNodes = []
    outputNodes = []

    # Input is layer I, hidden layer is layer K, output layer is layer H
    # Each weights list is a 2D array with Row = stem node, Column = outgoing weight
    IKweights = []
    KHweights = []

    # initialize weights with random numbers
    IKweights, KHweights = initializeWeights(IKweights, KHweights)

    inputWeights = IKweights
    outputWeights = KHweights

    '''
    #TESTING FOR LOOP
    for i in range(10):
        #print("current example is " + str(currentExample))
        inputNodes, outputNodes, hiddenNodes = forwardPropogation(trainingData[0], inputWeights, outputWeights)
        inputWeights, outputWeights = backwardsPropogation(trainingData[0], inputNodes, outputNodes, outputWeights, hiddenNodes, inputWeights)

    #print("current example is " + str(currentExample))
    inputNodes, outputNodes, hiddenNodes = forwardPropogation(trainingData[0], inputWeights, outputWeights)
    print("output for example " + str(0))
    print(outputNodes)
    '''

    # main function that forward propogates and backward propogates for a set number of iterations
    for i in range(NUM_ITERATIONS):
        #shuffle the list to add randomization and prevent overfitting
        random.shuffle(trainingData)

        #train the model on each training example in the training set
        for j in range(len(trainingData)):
            exampleSelection = j
            currentExample = trainingData[exampleSelection]

            # forward propogate through the neural network using the input and weights to recieve an output
            inputNodes, outputNodes, hiddenNodes = forwardPropogation(
            currentExample, inputWeights, outputWeights)

            # backward propogate through the data, finding the error and adjusting weights accordingly
            inputWeights, outputWeights = backwardsPropogation(
            currentExample, inputNodes, outputNodes, outputWeights, hiddenNodes, inputWeights)

    showResults(validationData, inputWeights, outputWeights)

# function that returns the output of the NN and the expected output from the validation data
def returnResults(actualOutput, expectedOutput, example):
    actualDigit = 0
    expectedDigit = 0
    confidence = 0
    correctGuess = False

    # Find the algorithm's predicted digit by finding the largest value in the output node
    for i in range(len(actualOutput)):
        if (actualOutput[i] > actualOutput[actualDigit]):
            actualDigit = i
    confidence = actualOutput[actualDigit] * 100
    confidence = round(confidence, 2)

    # Find the expected output by finding which output is 1
    for i in range(len(expectedOutput[example][1])):
        if (expectedOutput[example][1][i:i+1] == "1"):
            expectedDigit = i

    if (actualDigit == expectedDigit):
        correctGuess = True

    return actualDigit, expectedDigit, confidence, correctGuess


def showResults(validationData, inputWeights, outputWeights):
    accuracy = 0

    for i in range(len(validationData)):
        correctnessString = 'incorreclty'
        inputNodes, outputNodes, hiddenNodes = forwardPropogation(
            validationData[i], inputWeights, outputWeights)
        actualOutput, expectedOutput, confidence, correctness = returnResults(
            outputNodes, validationData, i)

        if (correctness == True):
            accuracy += 1
            correctnessString = 'correctly'

        print(str(i) + ": model " + correctnessString + " predicted " + str(actualOutput) + " with a confidence of " + str(confidence)
              + "% actual digit was " + str(expectedOutput))

    print("accuracy for the model was " +
          str(accuracy/len(validationData)*100) + "%")


if __name__ == "__main__":
    main()
