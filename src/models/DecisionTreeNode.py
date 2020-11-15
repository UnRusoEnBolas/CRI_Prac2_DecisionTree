from models.SplittingAlgorithm import C4_5SplittingAlgorithm, GiniSplittingAlgorithm, ID3SplittingAlgorithm
import uuid

class DecisionTreeNode():
    """
    Classe que implementa cada uno de los nodos de los árboles de decisión.
    """
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
            self.prediction = self.dataFrame[self.decisionTree.targetAttribute].value_counts().index[0]
        else:
            self.prediction = None
        
        if decisionTree.splittingAlgorithm == "ID3":
            self.splittingAlgorithm = ID3SplittingAlgorithm(self.dataFrame, self.decisionTree.targetAttribute, self.decisionTree.trueLabel)
        elif decisionTree.splittingAlgorithm == "C4.5":
            self.splittingAlgorithm = C4_5SplittingAlgorithm(self.dataFrame, self.decisionTree.targetAttribute, self.decisionTree.trueLabel)
        elif decisionTree.splittingAlgorithm == 'Gini' or 'gini' or 'GINI':
            self.splittingAlgorithm = GiniSplittingAlgorithm(self.dataFrame, self.decisionTree.targetAttribute, self.decisionTree.trueLabel)

        self.childrenNodes = []
        self.depth = 1 if self.isRoot else self.parentNode.depth+1
    
    def getChildrenNodes(self):
        """
        Método que devuelve los DecisionTreeNode descendientes del actual. (Si el actual es un nodo hoja
        devuelve una lista vacía)
        """
        splittingAttribute = self.splittingAlgorithm.getSplittingAttribute()
        self.splittingAttribute = splittingAttribute
        possibleChoices = self.dataFrame[splittingAttribute].unique()
        childNodes = []
        for value in possibleChoices:
            subset = self.dataFrame[self.dataFrame[splittingAttribute]==value]
            subset.drop(splittingAttribute, axis=1, inplace=True)
            isLeaf = True if subset[self.decisionTree.targetAttribute].nunique() == 1 or self.depth == self.decisionTree.maxDepth else False
            childNodes.append(DecisionTreeNode(self.decisionTree, subset, self, value, isLeaf=isLeaf))
        return childNodes

    def toJSON(self):
        node = {}
        node['uuid'] = str(self.uuid)


