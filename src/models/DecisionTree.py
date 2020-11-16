import numpy as np
from models.DecisionTreeNode import DecisionTreeNode
from graphviz import Digraph
import json

class DecisionTree():
    """
    Esta es la clase que implementa los arboles de decisión i permite visualizarlos.
    Dependiendo de los parámetros que se pasen al constructor, se pueden crear árboles
    de decisión usando dinstintos criterios de separación (ID3, C4.5 y Gini).
    """
    def __init__(self, dataFrame = None, targetAttribute = None, trueLabel = None, falseLabel = None, splittingAlgorithm = None, maxDepth = None):
        """
        dataFrame: pandas.DataFrame sobre el que queremos generar el árbol de decisión.
        targetAttribute: String que coincide con el nombre de la columna del dataFrame que queremos usar como objetivo del árbol de decisión.
        trueLabel: String/Número/Booleano que indica de que manera representa la columna objetivo el valor positivo, por ejemplo: "Si", 1, True...
        splittingAlgorithm: String que indica que criterio de separación queremos usar: "ID3", "C4.5", "Gini".
        maxDepth: Número entero que determina la profundidad máxima del árbol.
        """
        self.dataFrame = dataFrame
        self.splittingAlgorithm = splittingAlgorithm
        self.targetAttribute = targetAttribute
        self.trueLabel = trueLabel
        self.falseLabel = falseLabel
        self.maxDepth = maxDepth
        self.rootNode = DecisionTreeNode(self, dataFrame, None, None, isRoot=True)

    def fromJSON(self, fileName):
        path = f'./outputs/modelsOutputs/{fileName}.json'
        file = open(path ,'r')
        treeDict = json.load(file)
        file.close()
        self.rootNode = DecisionTreeNode().fromDict(treeDict, None)
        return self
    
    def generate(self):
        """
        Método que inicia la recursión para generar el árbol de decisión.
        """
        self.__build(self.rootNode)
    
    def __build(self, currentNode):
        """
        Método (privado) recursivo que genera todos los nodos del árbol de decisión.
        """
        if currentNode.isLeaf: return
        childrenNodes = currentNode.getChildrenNodes()
        currentNode.childrenNodes = childrenNodes
        for childNode in childrenNodes:
            self.__build(childNode)
        return

    def predict(self, dataFrame):
        """
        Método que recive por parámetro un dataframe con un número indeterminado de registros
        y devuelve una matriz con cada una de las predicciones del árbol para esos registros.
        """
        predictions = []
        for row in range(dataFrame.shape[0]):
            register = dataFrame.iloc[row, :]
            actualNode = self.rootNode
            while not actualNode.isLeaf:
                match = False
                registerValue = register[actualNode.splittingAttribute]
                for nextNode in actualNode.childrenNodes:
                    if nextNode.splittingValue == str(registerValue):
                        actualNode = nextNode
                        match = True
                        break
                if not match: actualNode = actualNode.childrenNodes[0]
            predictions.append(actualNode.prediction)
        return predictions

    def visualize(self, title):
        """
        Renderiza un PNG con la estructura final del árbol de decisión.
        title: Parámetro al que se le pasa un String para que se use como título de la imagen en
        la parte inferior del PNG y como nombre del archivo generado.
        """
        dot = Digraph(comment="Graphic representation of the resulting decision tree", format='png')
        dot.attr(label=title)
        self.__buildVisualization(dot, None, self.rootNode)
        dot.render(f'./outputs/graphOutputs/{title}.gv', view=True)

    def __buildVisualization(self, dot, previousNode, currentNode):
        """
        Método (privado) recursivo que genera la visualización del árbol.
        """
        if currentNode.isRoot:
            dot.node(currentNode.uuid, str(currentNode.splittingAttribute), shape="box")
            for childNode in currentNode.childrenNodes:
                self.__buildVisualization(dot, currentNode, childNode)
        elif currentNode.isLeaf:
            dot.node(currentNode.uuid, str(currentNode.prediction), shape="ellipse")
            dot.edge(previousNode.uuid, currentNode.uuid, label=str(currentNode.splittingValue))
            return
        else:
            childrenNodes = currentNode.childrenNodes
            dot.node(currentNode.uuid, str(currentNode.splittingAttribute), shape="box")
            dot.edge(previousNode.uuid, currentNode.uuid, label=str(currentNode.splittingValue))
            for childNode in childrenNodes:
                self.__buildVisualization(dot, currentNode, childNode)
            return
    
    def saveToFile(self, fileName):
        """
        Método que inicia una recursión por cada nodo para convertirlo en formato JSON para poder
        guardar el árbol en un fichero de texto y poder cargarlo posteriormente.
        Guarda el JSON generado en la ruta ./outputs/modelsOutputs/<parámetro fileName>.json.
        """
        file = open(f'./outputs/modelsOutputs/{fileName}.json', 'w+')
        json.dump(self.rootNode.toJSON(), file)
        file.close()

