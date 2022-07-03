import pandas as pd
df = pd.read_csv("C:/Users/CC-062/Desktop/jayshree/IRIS.csv")
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X = df.drop(['species'], axis=1)
Y = df['species']
best = SelectKBest(score_func=chi2, k='all')
fit = best.fit(X, Y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumn = pd.DataFrame(X.columns)
featurescores = pd.concat([dfcolumn, dfscores], axis=1)
featurescores.columns = ['Specs', 'Score']
print(featurescores)


'''
OUTPUT
          Specs       Score
0  sepal_length   10.817821
1   sepal_width    3.594499
2  petal_length  116.169847
3   petal_width   67.244828

Process finished with exit code 0
'''
