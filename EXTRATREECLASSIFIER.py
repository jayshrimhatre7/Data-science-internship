import pandas as pd
df = pd.read_csv("C:/Users/CC-062/Desktop/jayshree/IRIS.csv")

X = df.drop(['species'], axis=1)
Y = df['species']

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_)
feat_importance = pd.Series(model.feature_importances_)
feat_importance.nlargest(4).plot(kind='barh')
plt.show()

'''
OUTPUT:
[0.07923035 0.05584424 0.45943512 0.40549029]'''





