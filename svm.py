import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

train_data = pd.read_csv("/home/amitoj/Downloads/data.csv")

### read train data
# print(train_data.head()
del train_data['id']

#### deleting the id of train_data since it is irrelevant for training purpose

#### Splitting the whole dataset into 0.6 of training dat and 0.4 of test data for validation purposes
X = train_data[train_data.columns[1:30]].values

Y = train_data.loc[:, ['diagnosis']].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)

#### PREPROCESSING
X_Train = preprocessing.scale(X_train)
X_Test= preprocessing.scale(X_test)
# print(X_scaled_train)
# print(X_scaled_test)
# scaling is done to have zero mean and unit variance

#### converting the 30 features(attributes) as set of real number vectors
Y_train = (np.ravel(np.array(y_train)))
Y_test = (np.ravel(np.array(y_test)))

le = preprocessing.LabelEncoder()
Y_Train = le.fit_transform(Y_train)
Y_Test = le.fit_transform(Y_test)
print(len(Y_Test))
#### Model the data using svm with rbf kernel since the instances in the data set is less than features.

rbf_svc = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto').fit(X_Train, Y_Train)

#### Predicting the score of the test data features
print(rbf_svc.score(X_train, Y_Train))
# print(rbf_svc.predict(X_Test))
prediction = rbf_svc.predict(X_Test)
print(prediction)

#### checking accuracy of the linear model with rbf kernel
accuracy = np.count_nonzero(prediction == Y_Test)
print(accuracy)
percentage_accuracy = (accuracy/len(Y_Test) * 100)
print(percentage_accuracy)


