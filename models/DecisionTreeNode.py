from models.SplittingAlgorithm import C4_5SplittingAlgorithm, ID3SplittingAlgorithm
import uuid

class DecisionTreeNode():
    def __init__(self, decisionTree, dataFrame, parentNode, splittingValue, isRoot=False, isLeaf=False):
        self.uuid = str(uuid.uuid4())
        self.decisionTree = decisionTree
        self.dataFrame = dataFrame
        self.parentNode = parentNode
        self.splittingValue = splittingValue
        self.isRoot = isRoot
        
        self.isLeaf = isLeaf
        self.splittingAttribute = None
        if isLeaf:
            self.prediction = self.dataFrame[self.decisionTree.targetAttribute].unique()[0]
        else:
            self.prediction = None
        
        if decisionTree.splittingAlgorithm == "ID3":
            self.splittingAlgorithm = ID3SplittingAlgorithm(self.dataFrame, self.decisionTree.targetAttribute, self.decisionTree.trueLabel)
        elif decisionTree.splittingAlgorithm == "C4.5":
            self.splittingAlgorithm = C4_5SplittingAlgorithm(self.dataFrame, self.decisionTree.targetAttribute, self.decisionTree.trueLabel)

        self.childrenNodes = []
        self.depth = 0 if self.isRoot else self.parentNode.depth+1
        
    
    def getChildrenNodes(self):
        splittingAttribute = self.splittingAlgorithm.getSplittingAttribute()
        self.splittingAttribute = splittingAttribute
        possibleChoices = self.dataFrame[splittingAttribute].unique()
        childNodes = []
        for value in possibleChoices:
            subset = self.dataFrame[self.dataFrame[splittingAttribute]==value]
            subset.drop(splittingAttribute, axis=1, inplace=True)
            isLeaf = True if subset[self.decisionTree.targetAttribute].nunique() == 1 else False
            childNodes.append(DecisionTreeNode(self.decisionTree, subset, self, value, isLeaf=isLeaf))
        return childNodes


