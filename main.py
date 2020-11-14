from models.DecisionTree import DecisionTree
from modules.DatasetModifications import discretizeDataframe, deleteRowsWithValues
import pandas as pd
pd.options.mode.chained_assignment = None

testDataset = pd.read_csv('data/testDataset/continuous+unknownTestData.csv')
continuousColumns = ['Temp', 'Humidity']
nbins = 4
testDataset = discretizeDataframe(testDataset, continuousColumns, nbins)
testDataset = deleteRowsWithValues(testDataset, 'unknown')
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