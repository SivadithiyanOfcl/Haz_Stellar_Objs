import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

df= pd.read_csv("neo_v2.csv")

le = LabelEncoder()
df["hazardous"] = le.fit_transform(df["hazardous"])
df["hazardous"] = df["hazardous"].astype(int)

df = df.drop(['id','name','orbiting_body','sentry_object'], axis = 1)

X = df.iloc[:,:5]
y = df.iloc[:,-1]
X.to_csv('scaler_data.csv', index=False)

sc = StandardScaler()
X = sc.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42,shuffle = True)

KNN_classifier = KNeighborsClassifier(n_neighbors = 5,metric = 'minkowski',p = 2)
KNN_classifier.fit(x_train,y_train)
pickle.dump(KNN_classifier,open('knn.pkl','wb'))

SV_classifier = SVC(kernel = 'rbf',random_state = 42)
SV_classifier.fit(x_train,y_train)
pickle.dump(SV_classifier,open('svc.pkl','wb'))

nb = GaussianNB()
nb.fit(x_train,y_train)
pickle.dump(nb,open('nb.pkl','wb'))

Tree = DecisionTreeClassifier(criterion = 'entropy',random_state = 42)
Tree.fit(x_train,y_train)
pickle.dump(Tree,open('tree.pkl','wb'))

Forest = RandomForestClassifier(n_estimators = 500, criterion = 'entropy',random_state = 42)
Forest.fit(x_train,y_train)
pickle.dump(Forest,open('forest.pkl','wb'))

gbc = GradientBoostingClassifier()
gbc.fit(x_train,y_train)
pickle.dump(gbc,open('gbc.pkl','wb'))