import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import  accuracy_score

df = pd.read_csv("C:/Users/hp/Desktop/datasets/Titanic_tested.csv")
#print(df)

reg = LinearRegression ()

x = df.drop('PassengerId',axis=1)
x = x.drop('Survived', axis=1)
x = x.drop('Name',axis=1)
x = x.drop('Ticket',axis=1)
x = x.drop('Cabin',axis=1)
x = x.drop('Embarked',axis=1)
x = x.drop('Parch',axis=1)
x = x.drop('Sex',axis=1)
'''le = LabelEncoder()
le.fit(x)
x = le.transform(x)
print(x)
'''
y = df['Survived']

x['Age'].fillna((x['Age'].mean()), inplace=True)
x['Fare'].fillna((x['Fare'].mean()), inplace=True)
# print(x.info())
# print(y.info())



x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)
reg_train = reg.fit(x_train, y_train)
reg_pred = reg.predict(x_test)

print('Mean Squared Error:', mean_squared_error(y_test, reg_pred))
