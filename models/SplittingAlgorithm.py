import pandas as pd
import numpy as np

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
        self.initialEntropy = self.calculateEntropy(self.dataFrame)

    def calculateEntropy(self, subset):
        totalCount = subset.shape[0]
        positiveCount = subset[subset[self.targetAttribute] == self.trueLabel].count()[1]
        negativeCount = totalCount - positiveCount
        positiveProbability = np.divide(positiveCount, totalCount)
        negativeProbability = np.divide(negativeCount, totalCount)

        positiveProbLog = np.log2(positiveProbability)
        negativeProbLog = np.log2(negativeProbability)
        positiveProbXLog = np.multiply(positiveProbability, positiveProbLog)
        negativeProbXLog = np.multiply(negativeProbability, negativeProbLog)
        entropy = -np.add(positiveProbXLog, negativeProbXLog)
        return entropy


class C4_5SplittingAlgorithm(SplittingAlgorithm):
    """
    Esta clase hereda de la classe base SplittingAlgorithm y es donde se implementa el
    algoritmo C4.5.
    """
    def __init__(self, dataFrame, targetAttribute, trueLabel):
        super().__init__(dataFrame, targetAttribute, trueLabel)
    
    def getSplittingAttribute(self):
        """
        Este método encuentra cuál es el atributo por el cuál se deben separar los datos para
        conseguir el mejor ratio de ganancia.
        """
        for column in ['Wind']: #self.dataFrame.columns: 
            entropies = []
            counts = []
            for value in self.dataFrame[column].unique():
                subset = self.dataFrame[self.dataFrame[column] == value]
                entropy = self.calculateEntropy(subset)
                counts.append(subset.shape[0])
                entropies.append(entropy)
        print(np.multiply(entropies, counts))
                





            