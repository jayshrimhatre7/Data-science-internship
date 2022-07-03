import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer

rf = RandomForestClassifier(random_state=1)
nb=MultinomialNB()
dt=DecisionTreeClassifier(random_state=0)
df = pd.read_json('C:/Users/CC-062/Desktop/jayshree/cusine_train.json/train.json')
#print(df['cuisine'].unique())
d_c = ['greek', 'southern_us', 'filipino', 'indian', 'jamaican', 'spanish', 'italian', 'mexican', 'chinese', 'british', 'thai', 'vietnamese', 'cajun_creole',  'brazilian', 'french', 'japanese', 'irish', 'korean', 'moroccan', 'russian']
x=df['ingredients']
#z=df['all_ingredients'] = df['ingredients'].map(";".join)#here commas are considered to separate two different attributes or features in a list. But here the individual ingredients act as unique values so they need not be represented as a list element. Here, all commmas get replaced with semi colons to avoid this issue. Join function will join the ingredient values and it will be treated like a whole sentence. This helps in fuctions of Count Vectorizer.
#print(z)
y = df['cuisine'].apply(d_c.index)

df['all_ingredients'] = df['ingredients'].map(";".join)
cv = CountVectorizer()
X = cv.fit_transform(df['all_ingredients'].values)
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

dt.fit(x_train,y_train)
rf.fit(x_train,y_train)
nb.fit(x_train,y_train)

y_pred1=dt.predict(x_test)
y_pred2=rf.predict(x_test)
y_pred3=nb.predict(x_test)


print("Decision tree", accuracy_score(y_test, y_pred1))
print("Random Forest", accuracy_score(y_test, y_pred2))
print("Naive Bayes", accuracy_score(y_test, y_pred3))
