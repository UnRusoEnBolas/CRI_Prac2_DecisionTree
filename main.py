from models.DecisionTree import DecisionTree
import pandas as pd
pd.options.mode.chained_assignment = None

testDataset = pd.read_csv('data/testDataset/categoricalTestData.csv')
yColumn = 'Decision'
trueValue = 'Yes'

decisionTreeID3 = DecisionTree(testDataset, yColumn, trueValue, 'ID3')
decisionTreeID3.generate()

decisionTreeC45 = DecisionTree(testDataset, yColumn, trueValue, 'C4.5')
decisionTreeC45.generate()