from models.DecisionTree import DecisionTree
import pandas as pd
from sklearn.metrics import classification_report
pd.options.mode.chained_assignment = None

tree = DecisionTree().fromJSON('C4.5_maxDepth3')
tree.visualize("First try for saved tree")