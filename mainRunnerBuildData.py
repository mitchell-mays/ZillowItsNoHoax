from enum import Enum
from scipy.stats import norm
from numpy._distributor_init import NUMPY_MKL  # requires numpy+mkl
from bisect import bisect
import numpy

import DataIDMatchup



#=======================================================================
#                   Problem specific imports
#=======================================================================
import ZillowPreprocessing


class ClassificationType(Enum):
    DEFINED = 1
    FLUID = 2


#General Information
orig_data_path = "D://Documents//kaggle//Zillow//properties_2016//properties_2016.csv"
class_path = "D://Documents//kaggle//Zillow//train_2016//train_2016.csv"

processed_data_path = "D://Documents//kaggle//Zillow//processed//data"
processed_class_path = "D://Documents//kaggle//Zillow//processed//class"

delimeter = ","
classType = ClassificationType.FLUID
numClasses = 2

#=======================================================================
#If these are both true, will use IDs to align class and data values
#Otherwise, will assume that class values are ordered to align with data
#=======================================================================
dataHasIdentifier = True
classHasIdentifier = True

#include index of features to be ignored -- make sure to ignore class if this is in file
featureIgnore = [0]

#if more than just the class info is stored, then ignore these also
classIgnore = []

#=======================================================================
#                             Extract Data
#=======================================================================
file = open(orig_data_path)
lines = file.readlines()
file.close()

#
#
#needed to add -1 to both shape and feature line since an extra /n was being included
#
#

featureIgnore.sort(reverse=True)
trainData = numpy.zeros(shape=(len(lines)-1, len(lines[1].split(delimeter))-1))
for line in range(len(lines)):
    if line != 0:
        featureLine = numpy.array([(float(x) if x else 0) for x in lines[line].split(delimeter)[:-1]])
        trainData[line-1] = featureLine

#trainData = numpy.array(trainLines)
if dataHasIdentifier:
    train_ids = trainData[:, 0]

#remove all ignored features from training data
numpy.delete(trainData, featureIgnore, axis=1)


#=======================================================================
#                             Create Labels
#=======================================================================
file = open(class_path)
lines = file.readlines()
file.close()

labelLines = []
trainLabels = numpy.zeros(shape=(len(lines)-1, len(lines[1].split(delimeter))-1))
for line in range(len(lines)):
    if line != 0:
        classLine = numpy.array([(float(x) if x else 0) for x in lines[line].split(delimeter)[:-1]])
        trainLabels[line-1] = classLine

if dataHasIdentifier and classHasIdentifier:
   trainLabels = DataIDMatchup.returnOrderedClassLabels(train_ids, trainLabels)

#
#
#
# Apply Problem specific preprocessing
trainData, trainLabels, noClassData, zeros = ZillowPreprocessing.shiftDatesAndRemoveEmpties(trainData, trainLabels)

#next shift over date to traindata array
#trainData = numpy.concatenate((trainData, trainLabels[:, 1]), axis=1)
#trainLabels = trainLabels[:, 0]
#
#
#

#normalize data
def data_normalize(x, maxVal, minVal):
    return (x-minVal)/(maxVal - minVal)


for column in range(len(trainData[0])):
    maxVal = max(trainData[:, column])
    minVal = max(trainData[:, column])

    if maxVal - minVal > 0:
        col = trainData[:, column]
        for item in col:
            data_normalize(item, maxVal, minVal)

        trainData[:, column] = col


if classType == ClassificationType.FLUID:
    minVal = min(trainLabels)
    maxVal = max(trainLabels)
    avg = numpy.average(trainLabels)
    std = numpy.std(trainLabels)

    sectBoundaries = []
    for i in range(numClasses-1):
        sectBoundaries.append(norm.ppf((i/numClasses)+(1/numClasses), loc=avg, scale=std))

    for classItem in range(len(trainLabels)):
        trainLabels[classItem] = bisect(sectBoundaries, trainLabels[classItem])

#expand out
trainLabelsTemp = numpy.zeros(shape=(len(trainLabels), numClasses))
for classItem in range(len(trainLabels)):
    trainLabelsTemp[classItem][int(trainLabels[classItem])-1] = 1

trainLabels = trainLabelsTemp
numpy.save(processed_data_path, trainData)
numpy.save(processed_class_path, trainLabels)