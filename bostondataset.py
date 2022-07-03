import pandas as pd
df = pd.read_csv('C:/Users/CC-062/Desktop/jayshree/HousingData.csv')
#print(df)
#from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#bos = load_boston
#print(bos.keys())
for col in df.columns:
    df[col].fillna((df[col].mean()), inplace=True)

print(df)

reg = LinearRegression()
x = df.drop(['MEDV'], axis=1)
print(x)
y = df['MEDV']
print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.3)
train = reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)
print(mean_squared_error(y_test,y_pred))