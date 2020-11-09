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
        dot.node(self.rootNode.splittingAttribute, self.rootNode.splittingAttribute)
        for childNode in self.rootNode.childrenNodes:
            childSplitAttr = childNode.prediction if childNode.isLeaf else childNode.splittingAttribute
            dot.node(childSplitAttr, childSplitAttr)
            dot.edge(self.rootNode.splittingAttribute, childSplitAttr)
        dot.render('./outputs/graphOutputs/0.gv', view=True)


        