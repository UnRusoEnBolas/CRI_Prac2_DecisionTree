from models.SplittingAlgorithm import C4_5SplittingAlgorithm
import pandas as pd

testDataset = pd.read_csv('data/testDataset/testData1.csv')
yColumn = 'Decision'

sa = C4_5SplittingAlgorithm(testDataset, yColumn, trueLabel='Yes')
sa.getSplittingAttribute()