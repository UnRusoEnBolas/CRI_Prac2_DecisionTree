"""
Script que genera árboles de distintas profundidades màximas para 
todos los criterios de partición programados.
"""
from models.DecisionTree import DecisionTree
import pandas as pd
import warnings
import threading
warnings.filterwarnings('ignore')

'''
trainData = pd.read_csv('./data/train.csv')
yColumnName = 'class'
trueValue = 1.0
falseValue = 0.0

"""
Creamos modelos con distintas profundidades máximas para poder decidir
qué hiperparámetro de profundiad máxima nos conviene.
"""
for model in ['ID3', 'Gini', 'C4.5']:
    for maxDepth in [3, 5, 7, 9, 11, 13, 15]:
        print(f'Progress: Model -> {model} with maximum depth: {maxDepth}')
        tree = DecisionTree(trainData, yColumnName, trueValue, falseValue, model, maxDepth=maxDepth)
        tree.generate()
        tree.saveToFile(f'{model}_maxDepth{maxDepth}')
    print('\n')
'''

"""
Modelos K-FOLD
"""
MAX_DEPTH = 1

kfoldPartitions = []
for i in range (1, 6):
    kfoldPartitions.append(pd.read_csv(f'./data/kfold/partition{i}of5.csv'))

def generateModelAndSave(tree, fileName):
    tree.generate()
    tree.saveToFile(fileName)

for kIdx in range(1,6):
    print(f"PROGRESS ===> Current K index {kIdx}/5")
    trainPartitions = []
    for partitionIdx in range(5):
        if partitionIdx+1 != kIdx: trainPartitions.append(kfoldPartitions[partitionIdx])
    trainData = pd.concat(trainPartitions)

    tree1 = DecisionTree(trainData, 'class', 1.0, 0.0, 'ID3', maxDepth=MAX_DEPTH)
    tree2 = DecisionTree(trainData, 'class', 1.0, 0.0, 'Gini', maxDepth=MAX_DEPTH)
    tree3 = DecisionTree(trainData, 'class', 1.0, 0.0, 'C4.5', maxDepth=MAX_DEPTH)
    t1 = threading.Thread(target=generateModelAndSave, args=(tree1, f'ID3_maxDepth{MAX_DEPTH}__Partition{kIdx}of5_isOutOfTraining'))
    t2 = threading.Thread(target=generateModelAndSave, args=(tree2, f'Gini_maxDepth{MAX_DEPTH}__Partition{kIdx}of5_isOutOfTraining'))
    t3 = threading.Thread(target=generateModelAndSave, args=(tree3, f'C4.5_maxDepth{MAX_DEPTH}__Partition{kIdx}of5_isOutOfTraining'))
    t1.start()
    t2.start()
    t3.start()
    t1.join()
    print('--- PREOGRESS ===> ID3 FINISHED')
    t2.join()
    print('--- PREOGRESS ===> Gini FINISHED')
    t3.join()
    print('--- PREOGRESS ===> C4.5 FINISHED')
    print('\n')