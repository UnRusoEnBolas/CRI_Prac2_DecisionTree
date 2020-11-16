from models.DecisionTree import DecisionTree
import pandas as pd
from sklearn.metrics import classification_report
pd.options.mode.chained_assignment = None


testData = pd.read_csv('./data/test.csv')

tree = DecisionTree().fromJSON('C4.5_maxDepth15')
predictions = tree.predict(testData)
print(classification_report(testData['class'], predictions))
print('\n')
tree = DecisionTree().fromJSON('ID3_maxDepth15')
predictions = tree.predict(testData)
print(classification_report(testData['class'], predictions))
print('\n')
tree = DecisionTree().fromJSON('Gini_maxDepth15')
predictions = tree.predict(testData)
print(classification_report(testData['class'], predictions))