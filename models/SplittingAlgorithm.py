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
        self.initialEntropy = self.getEntropy(self.dataFrame)

    def getEntropy(self, subset):
        """
        Devuelve la entropía y las probabilidades respecto a la columna objetivo del dataframe
        que recibe por el parámetro subset.
        """
        totalCount = subset.shape[0]
        positiveCount = subset[subset[self.targetAttribute] == self.trueLabel].count()[1]
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
        for column in ['Wind', 'Outlook']: #self.dataFrame.columns: 
            entropies = []
            counts = []
            for value in self.dataFrame[column].unique():
                subset = self.dataFrame[self.dataFrame[column] == value]
                counts.append(subset.shape[0])
                entropy = self.getEntropy(subset)
                entropies.append(entropy)
            totalCount = np.sum(counts)
            gain = self.initialEntropy - np.sum(np.multiply(np.divide(counts, totalCount), entropies))
            #Now go for SplitInfo            





            