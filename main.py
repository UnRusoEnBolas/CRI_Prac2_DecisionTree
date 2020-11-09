from models.DecisionTree import DecisionTree
import pandas as pd
pd.options.mode.chained_assignment = None

'''
testDataset = pd.read_csv('data/testDataset/continuousTestData.csv')
yColumn = 'Decision'

sa = C4_5SplittingAlgorithm(testDataset, yColumn, trueLabel='Yes')
print(sa.getSplittingAttribute())

print("------------------------")

testDataset = pd.read_csv('data/testDataset/categoricalTestData.csv')
yColumn = 'Decision'

sa = ID3SplittingAlgorithm(testDataset, yColumn, trueLabel='Yes')
print(sa.getSplittingAttribute())
'''

testDataset = pd.read_csv('data/testDataset/categoricalTestData.csv')
yColumn = 'Decision'

decisionTreeID3 = DecisionTree(testDataset, yColumn, 'Yes', 'ID3')
decisionTreeID3.generate()

decisionTreeC45 = DecisionTree(testDataset, yColumn, 'Yes', 'C4.5')
decisionTreeC45.generate()
