import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('data/realDataset/Processed/processed_advertisments_dataset.csv')
for contCol in ['height', 'width', 'aratio']:
    dataset.loc[dataset[contCol] == 0, contCol] = 'unknown'
dataset.loc[dataset['class'] == 'ad', 'class'] = 1
dataset.loc[dataset['class'] == 'noad', 'class'] = 0
dataset.to_csv('./finalAdvertismentsDataset.csv', index=False)