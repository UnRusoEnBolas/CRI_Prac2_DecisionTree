from models.DecisionTreeNode import DecisionTreeNode
from graphviz import Digraph

class DecisionTree():
    def __init__(self, dataFrame, targetAttribute, trueLabel, splittingAlgorithm):
        self.dataFrame = dataFrame
        self.splittingAlgorithm = splittingAlgorithm
        self.targetAttribute = targetAttribute
        self.trueLabel = trueLabel
        self.rootNode = DecisionTreeNode(self, dataFrame, None, None, isRoot=True)
    
    def generate(self):
        self.build(self.rootNode)
    
    def build(self, currentNode):
        if currentNode.isLeaf: return
        childrenNodes = currentNode.getChildrenNodes()
        currentNode.childrenNodes = childrenNodes
        for childNode in childrenNodes:
            self.build(childNode)
        return

    def visualize(self):
        dot = Digraph(comment="Graphic representation of the resulting decision tree", format='png')
        self.__buildVisualization(dot, None, self.rootNode)
        dot.render('./outputs/graphOutputs/0.gv', view=True)

    def __buildVisualization(self, dot, previousNode, currentNode):
        if currentNode.isRoot:
            dot.node(currentNode.uuid, currentNode.splittingAttribute, shape="box")
            for childNode in currentNode.childrenNodes:
                self.__buildVisualization(dot, currentNode, childNode)
        elif currentNode.isLeaf:
            dot.node(currentNode.uuid, currentNode.prediction, shape="ellipse")
            dot.edge(previousNode.uuid, currentNode.uuid, label=currentNode.splittingValue)
            return
        else:
            childrenNodes = currentNode.getChildrenNodes()
            dot.node(currentNode.uuid, currentNode.splittingAttribute, shape="box")
            dot.edge(previousNode.uuid, currentNode.uuid, label=currentNode.splittingValue)
            for childNode in childrenNodes:
                self.__buildVisualization(dot, currentNode, childNode)
            return
