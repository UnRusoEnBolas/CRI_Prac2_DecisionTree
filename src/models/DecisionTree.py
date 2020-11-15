import numpy as np
from models.DecisionTreeNode import DecisionTreeNode
from graphviz import Digraph
import uuid

from models.SplittingAlgorithm import SplittingAlgorithm

class DecisionTree():
    """
    Esta es la clase que implementa los arboles de decisión i permite visualizarlos.
    Dependiendo de los parámetros que se pasen al constructor, se pueden crear árboles
    de decisión usando dinstintos criterios de separación (ID3, C4.5 y Gini).
    """
    def __init__(self, dataFrame, targetAttribute, trueLabel, falseLabel, splittingAlgorithm, maxDepth):
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
        predictions = []
        for row in range(dataFrame.shape[0]):
            register = dataFrame.iloc[row, :]
            actualNode = self.rootNode
            while not actualNode.isLeaf:
                registerValue = register[actualNode.splittingAttribute]
                for nextNode in actualNode.childrenNodes:
                    if nextNode.splittingValue == registerValue:
                        actualNode = nextNode
                        break
            predictions.append(actualNode.prediction)
        return predictions

    def visualize(self, title=""):
        """
        Renderiza un PNG con la estructura final del árbol de decisión.
        title: Parámetro opcional al que se le puede pasar un String para que se use como título de la imagen en
        la parte inferior del PNG.
        """
        dot = Digraph(comment="Graphic representation of the resulting decision tree", format='png')
        dot.attr(label=title)
        self.__buildVisualization(dot, None, self.rootNode)
        dot.render(f'../outputs/graphOutputs/{str(uuid.uuid4())}.gv', view=True)

    def __buildVisualization(self, dot, previousNode, currentNode):
        if currentNode.isRoot:
            dot.node(currentNode.uuid, str(currentNode.splittingAttribute), shape="box")
            for childNode in currentNode.childrenNodes:
                self.__buildVisualization(dot, currentNode, childNode)
        elif currentNode.isLeaf:
            dot.node(currentNode.uuid, str(currentNode.prediction), shape="ellipse")
            dot.edge(previousNode.uuid, currentNode.uuid, label=str(currentNode.splittingValue))
            return
        else:
            childrenNodes = currentNode.getChildrenNodes()
            dot.node(currentNode.uuid, str(currentNode.splittingAttribute), shape="box")
            dot.edge(previousNode.uuid, currentNode.uuid, label=str(currentNode.splittingValue))
            for childNode in childrenNodes:
                self.__buildVisualization(dot, currentNode, childNode)
            return
    
    def saveToFile(self, path):
        print("Nothing yet")

