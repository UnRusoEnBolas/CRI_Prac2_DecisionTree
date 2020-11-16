"""
Script que procesa los ficheros de datos para pdoer hacerlos usables.
También genera archivos de test y de train además de discretizar columnas y
tartar los 'missing-values'.
"""
import pandas as pd
from modules.misc import deleteRowsWithValues, discretizeDataframe, train_test_split

dataset = pd.read_csv('./data/processed_advertisments_dataset.csv')
for contCol in ['height', 'width', 'aratio']:
    dataset.loc[dataset[contCol] == 0, contCol] = 'unknown'
dataset.loc[dataset['class'] == 'ad', 'class'] = 1
dataset.loc[dataset['class'] == 'noad', 'class'] = 0
dataset.to_csv('./data/finalAdvertismentsDataset.csv', index=False)

data = pd.read_csv('./data/finalAdvertismentsDataset.csv')
yColumnName = 'class'
trueValue = 1
falseValue = 0
continuousColumns = ['height', 'width', 'aratio']
nbins = 4
data = deleteRowsWithValues(data, 'unknown')
data = discretizeDataframe(data, continuousColumns, nbins)
trainData, testData = train_test_split(data, 0.35)
trainData.to_csv('./data/train.csv', index=False)
testData.to_csv('./data/test.csv', index=False)