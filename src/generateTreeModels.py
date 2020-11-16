"""
Script que genera árboles de distintas profundidades màximas para 
todos los criterios de partición programados.
"""
from models import DecisionTree
import pandas as pd

trainData = pd.read_csv('./data/train.csv')
yColumnName = 'class'
trueValue = 1.0
falseValue = 0.0

for model in ['ID3', 'Gini', 'C4.5']:
    for maxDepth in [3, 4, 5, 6, 7, 8, 9, 10]:
        print(f'Progress: Model -> {model} with maximum depth: {maxDepth}')
        tree = DecisionTree(trainData, yColumnName, trueValue, falseValue, model, maxDepth=maxDepth)
        tree.generate()
        tree.saveToFile(f'{model}_maxDepth{maxDepth}')
    print('\n')