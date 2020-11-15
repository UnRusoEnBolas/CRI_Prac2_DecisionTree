#ARTICULO PARA C4.5: https://sefiks.com/2018/05/13/a-step-by-step-c4-5-decision-tree-example/
#ARTICULO PARA ID3: https://nulpointerexception.com/2017/12/16/a-tutorial-to-understand-decision-tree-id3-learning-algorithm/

import numpy as np
import cProfile

class SplittingAlgorithm():
    """
    Esta es la clase base para los distintos algoritmos que eligen
    cuál debería ser el atributo por el que separar los datos (ID3, C4.5, etc...)
    """
    def __init__(self, dataFrame, targetAttribute, trueLabel = 1):
        """
        Se inicializa con un dataframe de pandas i especificando el nombre de la columna objetivo.
        El parámetro "trueLabel" sirve para indicar qué valor contienen las filas positivas de
        la variable objetivo, normalmente '1' (contrario de '0') o 'True' (contrario de 'False').
        Por defecto se usa '1'.

        Seguidamente calcula la inicializa la entropía del conjunto.
        """
        self.dataFrame = dataFrame
        self.targetAttribute = targetAttribute
        self.trueLabel = trueLabel
        self.initialEntropy = self.getEntropy(self.dataFrame)

    def getEntropy(self, subset):
        """
        Devuelve la entropía y las probabilidades respecto a la columna objetivo del dataframe
        que recibe por el parámetro subset.
        """
        totalCount = subset.shape[0]
        try:
            positiveCount = subset[subset[self.targetAttribute] == self.trueLabel].count()[1]
        except:
            positiveCount = 0
        negativeCount = totalCount - positiveCount
        positiveProbability = np.divide(positiveCount, totalCount)
        negativeProbability = np.divide(negativeCount, totalCount)

        positiveProbLog = 0 if positiveProbability == 0 else np.log2(positiveProbability)
        negativeProbLog = 0 if negativeProbability == 0 else np.log2(negativeProbability)
        positiveProbXLog = np.multiply(positiveProbability, positiveProbLog)
        negativeProbXLog = np.multiply(negativeProbability, negativeProbLog)
        entropy = -np.add(positiveProbXLog, negativeProbXLog)
        return entropy


class C4_5SplittingAlgorithm(SplittingAlgorithm):
    """
    Esta clase hereda de la clase base SplittingAlgorithm y es donde se implementa el
    algoritmo C4.5 (Proporción de la ganancia).
    """
    def __init__(self, dataFrame, targetAttribute, trueLabel):
        super().__init__(dataFrame, targetAttribute, trueLabel)
    
    def getSplittingAttribute(self):
        """
        Este método encuentra cuál es el atributo por el cuál se deben separar los datos para
        conseguir el mejor ratio de ganancia.
        """
        gainRatios = []
        for column in self.dataFrame.columns[:-1]: #['Wind', 'Outlook']:  
            entropies = []
            counts = []
            for value in self.dataFrame[column].unique():
                subset = self.dataFrame[self.dataFrame[column] == value]
                counts.append(subset.shape[0])
                entropy = self.getEntropy(subset)
                entropies.append(entropy)
            totalCount = np.sum(counts)
            gain = self.initialEntropy - np.sum(np.multiply(np.divide(counts, totalCount), entropies))
            splitInfo = -np.sum(np.multiply(np.divide(counts, totalCount), np.log2(np.divide(counts, totalCount))))
            gainRatio = gain/splitInfo if splitInfo != 0 else 0
            gainRatios.append(gainRatio)
        return self.dataFrame.columns[np.argmax(gainRatios)]


class ID3SplittingAlgorithm(SplittingAlgorithm):
    """
    Esta clase hereda de la clase base SplittingAlgorithm y es donde se implementa el
    algoritmo ID3 (Ganancia de información basado en entropía).
    """
    def __init__(self, dataFrame, targetAttribute, trueLabel):
        super().__init__(dataFrame, targetAttribute, trueLabel)
    
    def getSplittingAttribute(self):
        """
        Este método encuentra cuál es el atributo por el cuál se deben separar los datos para
        conseguir la mejor ganancia de información.
        """
        gains = []
        for column in self.dataFrame.columns[:-1]:
            entropies = []
            counts = []
            for value in self.dataFrame[column].unique():
                subset = self.dataFrame[self.dataFrame[column] == value]
                counts.append(subset.shape[0])
                entropy = self.getEntropy(subset)
                entropies.append(entropy)
            totalCount = np.sum(counts)
            informationGain = np.subtract(self.initialEntropy, np.sum(np.multiply(np.divide(counts, totalCount), entropies)))
            gains.append(informationGain)
        return self.dataFrame.columns[np.argmax(gains)]


class GiniSplittingAlgorithm(SplittingAlgorithm):
    """
    Esta clase hereda de la clase base SplittingAlgorithm y es donde se implementa el
    algoritmo Gini.
    """
    def __init__(self, dataFrame, targetAttribute, trueLabel):
        super().__init__(dataFrame, targetAttribute, trueLabel)
    
    def getSplittingAttribute(self):
        "Este método encuentra cuál es el atributo por el cuál se debe separar los datos"
        giniIndexes = []
        for column in self.dataFrame.columns[:-1]:
            giniSubindexes = []
            counts = []
            for value in self.dataFrame[column].unique():
                generalSubset = self.dataFrame[self.dataFrame[column] == value]
                counts.append(generalSubset.shape[0])
                positiveSubset = generalSubset[generalSubset[self.targetAttribute] == self.trueLabel]
                negativeSubset = generalSubset[generalSubset[self.targetAttribute] != self.trueLabel]
                giniSubindex = np.subtract(1, np.add(np.square(np.divide(positiveSubset.shape[0], generalSubset.shape[0])),np.square(np.divide(negativeSubset.shape[0], generalSubset.shape[0]))))
                giniSubindexes.append(giniSubindex)
            totalCount = np.sum(counts)
            giniIndex = np.sum(np.multiply(giniSubindexes, np.divide(counts, totalCount)))
            giniIndexes.append(giniIndex)
        return self.dataFrame.columns[np.argmin(giniIndexes)]