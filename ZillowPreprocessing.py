import numpy

def shiftDatesAndRemoveEmpties(trainData, classLabels):
    #hasClass = numpy.where(classLabels[trainData]!=0)
    nonZerosTrain = []
    nonZerosLabel = []
    zerosTrain = []
    zerosLabel = []
    for label in range(len(classLabels)):
        if classLabels[label] != 0:
            nonZerosTrain.append(trainData[label])
            nonZerosLabel.append(classLabels[label])
        else:
            zerosTrain.append(trainData[label])
            zerosLabel.append(classLabels[label])

    nonZerosTrain = numpy.array(nonZerosTrain)
    nonZerosLabel = numpy.array(nonZerosLabel)
    zerosTrain = numpy.array(zerosTrain)
    zerosLabel = numpy.array(zerosLabel)

    return nonZerosTrain, nonZerosLabel, zerosTrain, zerosLabel
