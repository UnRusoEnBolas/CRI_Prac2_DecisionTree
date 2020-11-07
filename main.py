from models.SplittingAlgorithm import C4_5SplittingAlgorithm, ID3SplittingAlgorithm
import pandas as pd

testDataset = pd.read_csv('data/testDataset/continuousTestData.csv')
yColumn = 'Decision'

sa = C4_5SplittingAlgorithm(testDataset, yColumn, trueLabel='Yes')
print(sa.getSplittingAttribute())

print("------------------------")

testDataset = pd.read_csv('data/testDataset/categoricalTestData.csv')
yColumn = 'Decision'

sa = ID3SplittingAlgorithm(testDataset, yColumn, trueLabel='Yes')
print(sa.getSplittingAttribute())