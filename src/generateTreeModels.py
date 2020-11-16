"""
Script que genera árboles de distintas profundidades màximas para 
todos los criterios de partición programados.
"""
from models.DecisionTree import DecisionTree
import pandas as pd

trainData = pd.read_csv('./data/train.csv')
yColumnName = 'class'
trueValue = 1.0
falseValue = 0.0

"""
Creamos modelos con distintas profundidades máximas para poder decidir
qué hiperparámetrod e profundiad máxima nos conviene.
"""
for model in ['ID3', 'Gini', 'C4.5']:
    for maxDepth in [3, 5, 7, 9, 11, 13, 15]:
        print(f'Progress: Model -> {model} with maximum depth: {maxDepth}')
        tree = DecisionTree(trainData, yColumnName, trueValue, falseValue, model, maxDepth=maxDepth)
        tree.generate()
        tree.saveToFile(f'{model}_maxDepth{maxDepth}')
    print('\n')