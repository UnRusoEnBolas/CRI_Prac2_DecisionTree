from models.DecisionTree import DecisionTree
import pandas as pd
pd.options.mode.chained_assignment = None

testDataset = pd.read_csv('data/testDataset/categoricalTestData.csv')
yColumn = 'Decision'
trueValue = 'Yes'

decisionTreeID3 = DecisionTree(testDataset, yColumn, trueValue, 'ID3')
decisionTreeID3.generate()
decisionTreeID3.visualize(title="Using ID3 split criterion")

decisionTreeC45 = DecisionTree(testDataset, yColumn, trueValue, 'C4.5')
decisionTreeC45.generate()
decisionTreeC45.visualize(title="Using C4.5 split criterion")

decisionTreeGini = DecisionTree(testDataset, yColumn, trueValue, 'Gini')
decisionTreeGini.generate()
decisionTreeGini.visualize(title="Using Gini split criterion")