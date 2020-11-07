from models.DecisionTreeNode import DecisionTreeNode

class DecisionTree():
    def __init__(self, dataFrame, targetAttribute, trueLabel, splittingAlgorithm):
        self.dataFrame = dataFrame
        self.splittingAlgorithm = splittingAlgorithm
        self.targetAttribute = targetAttribute
        self.trueLabel = trueLabel
        self.rootNode = DecisionTreeNode(self, dataFrame, None)