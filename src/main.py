from models.DecisionTree import DecisionTree
from modules.misc import discretizeDataframe, deleteRowsWithValues, train_test_split
import pandas as pd
from sklearn.metrics import classification_report
pd.options.mode.chained_assignment = None

data = pd.read_csv('finalAdvertismentsDataset.csv')
yColumnName = 'class'
trueValue = 1
falseValue = 0
continuousColumns = ['height', 'width', 'aratio']
nbins = 4
data = deleteRowsWithValues(data, 'unknown')
data = discretizeDataframe(data, continuousColumns, nbins)
trainData, testData = train_test_split(data, 0.35)

for model in ['ID3', 'Gini', 'C4.5']:
    for maxDepth in ['3','4','5','6','7','8','9','10']:
        print(f'Progress: Model -> {model} with maximum depth: {maxDepth}')
        tree = DecisionTree(trainData, yColumnName, trueValue, falseValue, model, maxDepth=maxDepth)
        tree.generate()
        tree.saveToFile(f'{model}_maxDepth{maxDepth}')
    print('\n')

'''
tree2 = DecisionTree(trainData, yColumn, trueValue, 'C4.5', maxDepth=5)
tree2.generate()
tree2.visualize(title="Using C4.5 split criterion")

tree3 = DecisionTree(trainData, yColumn, trueValue, 'Gini', maxDepth=5)
tree3.generate()
tree3.visualize(title="Using Gini split criterion")
'''


data = pd.read_csv('data/testDataset/continuousTestData.csv')
continuousColumns = ['Temp', 'Humidity']
nbins = 4
data = discretizeDataframe(data, continuousColumns, nbins)
yColumn = 'Decision'
trueValue = 'Yes'
falseValue = 'No'

trainData, testData = train_test_split(data, 0.3)
print(f'Registros de train: {trainData.shape}, registros de test: {testData.shape}')

decisionTreeID3 = DecisionTree(data, yColumn, trueValue, falseValue, 'ID3', maxDepth=2)
decisionTreeID3.generate()
#decisionTreeID3.visualize(title="ID3 Play tennis dataset")
predictions = decisionTreeID3.predict(testData)

print(classification_report(testData[yColumn], predictions))
decisionTreeID3.saveToFile('ID3PlayTennisDepth2')
