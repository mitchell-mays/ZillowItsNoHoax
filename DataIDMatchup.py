import numpy as np

def returnOrderedClassLabels(idData, classRows):
    newClassRows = np.zeros(len(idData))
    for elementId in range(len(classRows)):
        index = np.where(idData == classRows[elementId, 0])
        if len(index) > 0:
            newClassRows[index[0]] = classRows[elementId][1]

    return newClassRows
