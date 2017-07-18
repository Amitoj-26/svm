import pandas as pd
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
train_data = pd.read_csv("/home/amitoj/Downloads/data.csv")
del train_data['id']

X = train_data[train_data.columns[1:30]].values
Y = train_data.loc[:, ['diagnosis']].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

X_Train = preprocessing.Binarizer().fit_transform(X_train)
print(X_Train)
X_Test = preprocessing.Binarizer(threshold=1.0).fit_transform(X_test)
print(X_Test)
Y_train = (np.ravel(np.array(y_train)))
Y_test = (np.ravel(np.array(y_test)))

le = preprocessing.LabelEncoder()
Y_Train = le.fit_transform(Y_train)
Y_Test = le.fit_transform(Y_test)
# print(Y_Train)
# print(len(Y_Test))

Nb_clf = BernoulliNB()
print(Nb_clf.fit(X_train, Y_Train))
prediction = Nb_clf.predict(X_Test)

accuracy = np.count_nonzero(prediction == Y_Test)
print(accuracy)
percentage_accuracy = (accuracy/len(Y_Test) * 100)
print(percentage_accuracy)


