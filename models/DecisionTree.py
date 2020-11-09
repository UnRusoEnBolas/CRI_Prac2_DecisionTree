from models.DecisionTreeNode import DecisionTreeNode

class DecisionTree():
    def __init__(self, dataFrame, targetAttribute, trueLabel, splittingAlgorithm):
        self.dataFrame = dataFrame
        self.splittingAlgorithm = splittingAlgorithm
        self.targetAttribute = targetAttribute
        self.trueLabel = trueLabel
        self.rootNode = DecisionTreeNode(self, dataFrame, None, None, isRoot=True)
    
    def generate(self):
        self.buildTree(self.rootNode)
    
    def buildTree(self, currentNode):
        if currentNode.isLeaf: return
        childrenNodes = currentNode.getChildrenNodes()
        currentNode.childrenNodes = childrenNodes
        for childNode in childrenNodes:
            self.buildTree(childNode)
        return

        