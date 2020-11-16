from models.SplittingAlgorithm import C4_5SplittingAlgorithm, GiniSplittingAlgorithm, ID3SplittingAlgorithm
import uuid

class DecisionTreeNode():
    """
    Clase que implementa cada uno de los nodos de los árboles de decisión.
    """
    def __init__(self, decisionTree = None, dataFrame = None, parentNode = None, splittingValue = None, isRoot=False, isLeaf=False):
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
        
        if decisionTree == None or decisionTree.splittingAlgorithm == None:
            self.splittingAlgorithm = None
        elif decisionTree.splittingAlgorithm == "ID3":
            self.splittingAlgorithm = ID3SplittingAlgorithm(self.dataFrame, self.decisionTree.targetAttribute, self.decisionTree.trueLabel)
        elif decisionTree.splittingAlgorithm == "C4.5":
            self.splittingAlgorithm = C4_5SplittingAlgorithm(self.dataFrame, self.decisionTree.targetAttribute, self.decisionTree.trueLabel)
        elif decisionTree.splittingAlgorithm == 'Gini' or 'gini' or 'GINI':
            self.splittingAlgorithm = GiniSplittingAlgorithm(self.dataFrame, self.decisionTree.targetAttribute, self.decisionTree.trueLabel)

        self.childrenNodes = []
        try:
            self.depth = 1 if self.isRoot else self.parentNode.depth+1
        except: 
            self.depth = None
    
    def fromDict(self, nodeDict, parentNode):
        """
        Método que construye un DecisionTreeNode desde un diccionario de python. (Se usa cuando
        se quieren generar los nodos de un árbol guardado en ficheor JSON)
        """
        self.uuid = nodeDict['uuid'] if nodeDict['uuid'] else None
        self.parentNode = parentNode
        self.depth = nodeDict['depth'] if nodeDict['depth'] else None
        self.splittingAttribute = nodeDict['splittingAttirbute'] if 'splittingAttirbute' in nodeDict else None
        self.splittingValue = nodeDict['splittingValue'] if 'splittingValue' in nodeDict else None
        self.isRoot = True if parentNode == None else False
        self.isLeaf = nodeDict['isLeaf']
        self.childrenNodes = []
        if not self.isLeaf:
            for childDict in nodeDict['children']:
                self.childrenNodes.append(DecisionTreeNode().fromDict(childDict, self))
        self.prediction = nodeDict['prediction'] if nodeDict["isLeaf"] else None
        return self



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
        """
        Método que convierte en JSON el nodo del árbol para poder guardarlo en fichero JSON y
        posteriormente cargarlo.
        """
        node = {}
        node['uuid'] = str(self.uuid)
        node['depth'] = self.depth
        if self.parentNode != None: node['parent_uuid'] = str(self.parentNode.uuid)
        if self.splittingValue != None: node['splittingValue'] = str(self.splittingValue)
        if self.splittingAttribute != None: node['splittingAttirbute'] = self.splittingAttribute
        node['children'] = []
        if self.isLeaf:
            node['isLeaf'] = True
            node['prediction'] = self.prediction
            return node
        else:
            node['isLeaf'] = False
            node['children'] = []
            for child in self.childrenNodes:
                node['children'].append(child.toJSON())
        return node


       
        


