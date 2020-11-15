from models.DecisionTree import DecisionTree
from modules.DatasetModifications import discretizeDataframe, deleteRowsWithValues
import pandas as pd
pd.options.mode.chained_assignment = None

data = pd.read_csv('finalAdvertismentsDataset.csv')
continuousColumns = ['height', 'width', 'aratio']
nbins = 4
data = deleteRowsWithValues(data, 'unknown')
data = discretizeDataframe(data, continuousColumns, nbins)
yColumn = 'class'
trueValue = 1


print('Starting tree...')
decisionTreeID3 = DecisionTree(data, yColumn, trueValue, 'ID3', maxDepth=1)
decisionTreeID3.generate()
decisionTreeID3.visualize(title="Using C4.5 split criterion")

'''
data = pd.read_csv('data/testDataset/continuousTestData.csv')
continuousColumns = ['Temp', 'Humidity']
nbins = 4
data = discretizeDataframe(data, continuousColumns, nbins)
yColumn = 'Decision'
trueValue = 'Yes'

decisionTreeID3 = DecisionTree(data, yColumn, trueValue, 'ID3', maxDepth=1)
decisionTreeID3.generate()
decisionTreeID3.visualize(title="ID3 Play tennis dataset")
'''