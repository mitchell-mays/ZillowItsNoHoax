import FeedForwardNeuralNetwork
import numpy

processed_data_path = "D://Documents//kaggle//Zillow//processed//data.npy"
processed_class_path = "D://Documents//kaggle//Zillow//processed//class.npy"

iterations = 10000

data = numpy.load(processed_data_path)
labels = numpy.load(processed_class_path)

folds = 10
chunk = len(data)/folds
for fold in range(folds):
    testData = data[fold*chunk:(fold+1)*chunk]
    testLabels = labels[fold*chunk:(fold+1)*chunk]

    if fold > 0:
        if fold < (folds-1):
            trainData = numpy.concatenate((data[0:(fold*chunk)-1:], data[((fold+1)*chunk)+1:len(data)-1]))
            trainLabels = numpy.concatenate((labels[0:(fold*chunk)-1:], labels[((fold+1)*chunk)+1:len(labels)-1]))
        else:
            trainData = data[0:(fold*chunk)-1:]
            trainLabels = labels[0:(fold*chunk)-1:]
    else:
        trainData = data[((fold+1)*chunk)+1:len(data)-1]
        trainLabels = labels[((fold+1)*chunk)+1:len(data)-1]


    # build FFNN
    print("                            | Training -> FFNN")
    FFNN = FeedForwardNeuralNetwork.FFNN(trainData, trainLabels, testData, testLabels, len(trainData[0]), len(trainLabels[0]), iterations)

    FFNN_DATA = FFNN.trainAndClassify()
    print("                            | " + str(FFNN_DATA[1]))
    #NGRAM_accuracy += FFNN_DATA[1]