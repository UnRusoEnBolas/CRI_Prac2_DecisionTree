from models.SplittingAlgorithm import C4_5SplittingAlgorithm, ID3SplittingAlgorithm

class DecisionTreeNode():
    def __init__(self, decisionTree, dataFrame, parentNode):
        self.dataFrame = dataFrame
        self.decisionTree = decisionTree
        self.parentNode = parentNode
        if decisionTree.splittingAlgorithm == "ID3":
            self.splittingAlgorithm = ID3SplittingAlgorithm(self.dataFrame, self.decisionTree.targetAttribute, self.decisionTree.trueLabel)
        elif decisionTree.splittingAlgorithm == "C4.5":
            self.splittingAlgorithm = C4_5SplittingAlgorithm(self.dataFrame, self.decisionTree.targetAttribute, self.decisionTree.trueLabel)
        self.childrenNodes = []
    
    def getchildrenNodes(self):
        splittingAttribute = self.splittingAlgorithm.getSplittingAttribute()
        possibleChoices = self.dataFrame[splittingAttribute].unique()
        childNodes = []
        for value in possibleChoices:
            subset = self.dataFrame[self.dataFrame[splittingAttribute]==value]
            childNodes.append(DecisionTreeNode(self.decisionTree, subset, self))
        return childNodes


