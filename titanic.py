import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

#### GETTING TEST AND TRAIN DATA AS CSV FILE IN DATAFRAME
train_data = pd.read_csv("/home/amitoj/Downloads/train.csv")
test_data = pd.read_csv("/home/amitoj/Downloads/test.csv")

### PROBLEM STTAEMENT IS WHAT SORT OF PEOPLE IS GOING TO SURVIVE WOMEN AND CHILDREN AND FIRST CLASS PEOPLE ARE HIGHLY LIKELY TO SURVIVE

# print(train_data.head())
## preview some traindata

### information about train & test data_data
# print(train_data.info())

# print("-------------")
# print(test_data.info())
#### DATA EXPLORATION AND DATA VISUALIZATION
# print(train_data.isnull().sum())
# print(test_data.isnull().sum())
# #### Dropping irrelevant column such as NAME, PASSENGER ID ,TICKET ,CABIN train data  and test data (exclude passenger id) respectively.
### since they are not directly related with target varaible survival ####  Age & Embarkement has null values Let us deal with embarket first
train_data = train_data.drop(['Name', 'PassengerId', 'Ticket','Cabin'],axis=1)
test_data = test_data.drop(['Name', 'Ticket','Cabin','PassengerId'],axis =1)
# print(train_data.groupby(['Survived']).mean())   # groupby function will perfom grouping the required varaible
### Passenger in first class having high fare are likely to survive

# print(train_data.groupby(['Embarked']).size())  ### size will include nan values while counting,
## replacing embarked field Nan values with S since S embarked are more survived than others
train_data['Embarked'] =train_data['Embarked'].fillna("S")  # since test data emarked contains no nan values
# print(train_data[train_data['Embarked'].isnull()])
sns.factorplot('Embarked', 'Survived', data =train_data, size=4, aspect=3)  ### used to plot facet grid for categorical data to show multiple axes
fig, (axis1, axis2) = plt.subplots(1, 2, figsize =(15, 5))
# plt.show()
# Here the survial rate is closet in values to C embarkemnt
sns.countplot(x='Embarked', hue='Survived', data = train_data, ax=axis1)
# plt.show()
# Passengers with embarkemnt S & C are more likely to survive
## computing mean of survived passengers corrwsponding to each embarkement
survived_perc  = train_data[["Embarked", "Survived"]].groupby(['Embarked'], as_index= False).mean()  # as_index retains the index after operation
sns.barplot(x='Embarked', y='Survived', data =survived_perc,ax=axis2)
# plt.show()

### Removing S emabarkement as it has less mean rate of survival for both train & test data
### For removing S , we have to convert it into dummy varaible (indicative) since embarkment is cateogerical varaible.

embark_dummies_train_data = pd.get_dummies(train_data['Embarked']) # it converts categorical vararible into dummy (indicative) varaible
# print(embark_dummies_train_data)
embark_dummies_train_data.drop(['S'], axis=1, inplace=True)  # inplace = True # modify the dataframe inplace of the object
# print(embark_dummies_train_data)

embark_dummies_test_data = pd.get_dummies(test_data['Embarked'])
embark_dummies_test_data.drop(['S'], axis=1, inplace=True)
train_data = train_data.join(embark_dummies_train_data)
test_data    = test_data.join(embark_dummies_test_data)

train_data.drop(['Embarked'], axis=1,inplace=True)
test_data.drop(['Embarked'], axis=1,inplace=True)
# print(train_data)


#### it contains one missing value in fare
##### imputatioon of fare value in test data by median
test_data_fare =(test_data["Fare"].fillna(test_data["Fare"].median(),inplace=True))
# since only test data fare contain missing value
# print(test_data['Fare'])
test_data['Fare'] = test_data['Fare'].astype(int)
train_data['Fare'] = train_data['Fare'].astype(int)       ### for converting fare  into numerical values
# print(test_data['Fare'])

#Get mean & std of fare for survived
fare_mean= train_data[["Fare", "Survived"]].groupby(['Survived'],as_index=False).mean()
fare_std= train_data[["Fare", "Survived"]].groupby(['Survived'],as_index=False).std()
# print(fare_mean,fare_std)
fig,(axis1) =plt.subplots(1,1,figsize=(10,15))
# print(train_data["Fare"].describe()) # to know max & min value of fare to decide no of classes & width of bin while plotting histogram
train_data['Fare'].plot(kind='hist', figsize=(15,3), xlim=(0,50),bins = 100)  # histogram is used for plotting frequency of varaible having numerical range
## fares in the range 5-30 are more in number
fare_mean.plot(yerr=fare_std, kind='bar',legend = False) # yerr = stddev means that amount of error bar in y dirn equivalent to fare_std is there
# plt.show()

#### Age expolration

average_age_train_data   = train_data["Age"].mean()
std_age_train_data       = train_data["Age"].std()
count_nan_age_train_data = train_data["Age"].isnull().sum()
#
average_age_test_data   = test_data["Age"].mean()
std_age_test_data       = test_data["Age"].std()
count_nan_age_test_data = test_data["Age"].isnull().sum()

# #### we will now impute age values by assigning random number in the range of (mean-std) and (mean+std)
rand_1 = np.random.randint(average_age_train_data - std_age_train_data, average_age_train_data + std_age_train_data, size = count_nan_age_train_data)
rand_2 = np.random.randint(average_age_test_data - std_age_test_data, average_age_test_data + std_age_test_data, size = count_nan_age_test_data)
# print(rand_1,rand_2)


# #
# # ### fill nan values with random values generated in the column
pd.options.mode.chained_assignment = None   # This command is used to avoid chained assignment warning
train_data['Age'][np.isnan(train_data['Age'])] = rand_1
test_data['Age'][np.isnan(test_data['Age'])] = rand_2
# print(train_data['Age'])
# print(test_data['Age'])
# ### checking null values
# print(train_data['Age'].isnull().sum())

### Havig facet plot for passengers survived or not survived with age
##### facet grid is subplot grid for plotting varaible  relationships.
#
facet = sns.FacetGrid(train_data,hue='Survived' ,aspect =4)

# # ketoplot is Fit and plot a univariate or bivariate kernel density estimate for varaible distribution
facet.map(sns.kdeplot,'Age',shade=True)
facet.add_legend() # add legend to the plot
facet.set(xlim= (0, train_data['Age'].max()))
# plt.show()
## we get peaks of survived/ not survived wrt age varaible

#### computation of mean age by survived passengers
fig,ax1 = plt.subplots(1,1,figsize =(18,4))
avg_age = train_data[['Age','Survived']].groupby(['Age'],as_index = False).mean()
# print(avg_age)# as_index= false means index will remain intact
sns.barplot(x='Age', y='Survived', data=avg_age)
# plt.show()
###  Plot shows that in age group 0-15 are more likely to survive

# ### Family column
# ## we are having Family column instead of Parch & Sibsp column
train_data['Family'] = train_data['Parch'] + train_data['SibSp']
#### We are assigning chances of arrival as 1 if they are having family members & vice versa
train_data['Family'].loc[train_data['Family']>0] = 1 # chances of survival
train_data['Family'].loc[train_data['Family']==0] = 0  ###chances of non survival

### For test data
test_data['Family'] =  test_data["Parch"] + test_data["SibSp"]
test_data['Family'].loc[test_data['Family'] > 0] = 1
test_data['Family'].loc[test_data['Family'] == 0] = 0

### droppping parch & sibsp
train_data = train_data.drop(['SibSp','Parch'],axis= 1)
test_data  = test_data.drop(['SibSp','Parch'],axis=1)

## plot
fig,(axis1,axis2) = plt.subplots(1,2,sharex = True,figsize=(10,15))
sns.factorplot('Family', 'Survived', data =train_data, size=4, aspect=3) # persons with family members are having high chances of survival
sns.countplot(x='Family', data=train_data, order=[1,0], ax=axis1)
axis1.set_xticklabels(["WithFamily","Alone"],rotation = 0)
# plt.show()
### plotting mean of groupby family memebers survvied
family_avg = train_data[["Family", "Survived"]].groupby(['Family'],as_index=False).mean()
# print(family_avg)
sns.barplot(x ='Family', y='Survived',data = family_avg, order=[1,0],ax = axis2)
# plt.show()
### So net conclusion is person having family member has high chances of survival


### passenger class
sns.factorplot('Pclass', 'Survived',order = [1,2,3], data = train_data,size = 5)
# plt.show()
## Class 3 is having lowest rate of survival,while class 1 & class 2 is having high chances of survival.
#
# ## dropping 3rd class varabile  as it has lowest avergae of survived passengers
# ## creating dummy varaible for passenger column
pclass_dummies_train_data = pd.get_dummies(train_data['Pclass'])
pclass_dummies_train_data.columns =['Class1','Class2','Class3'] # designating columns of different passenger class
# print(pclass_dummies_train_data)
pclass_dummies_train_data.drop(['Class3'],axis=1,inplace = True)

pclass_dummies_test_data = pd.get_dummies(test_data['Pclass'])
pclass_dummies_test_data.columns =['Class1','Class2','Class3']
pclass_dummies_test_data.drop(['Class3'],axis=1,inplace = True)

train_data.drop(['Pclass'],axis=1,inplace=True)
test_data.drop(['Pclass'],axis=1,inplace=True)

train_data = train_data.join(pclass_dummies_train_data)
test_data = test_data.join(pclass_dummies_test_data)
# print(train_data)
# print(test_data)

##### Sex column
# ## As we see,children (age< 16) has high rate of survival , to have high chance of survival
# ### So, classifying passengers into males,females & child
def get_person(passenger):
    age,sex = passenger
    if age < 16:
        return 'child'
    else:
        return sex
train_data['Person'] = train_data[['Age','Sex']].apply(get_person,axis=1)  ### applying get_person function on person column
test_data['Person']    = test_data[['Age','Sex']].apply(get_person,axis=1)
# print(train_data)

### Dropping sex column since we created person column
train_data.drop(['Sex'],inplace=True,axis =1)
test_data.drop(['Sex'],inplace=True,axis=1)
# print(train_data)
# print(test_data)


# Plotting it
fig,(axis1,axis2)= plt.subplots(1,2,figsize=(10,15))
sns.countplot(x='Person',data=train_data,ax=axis1)
# plt.show()
person_avg = train_data[['Person','Survived']].groupby(['Person'],as_index=False).mean()
# print(person_avg) ###Male has lowest rate of survival
sns.barplot(x='Person',y='Survived',data=train_data,ax=axis2,order=['male','female','child'])
# plt.show()
#### creating dummy coloumn from person column & drop Male as it has the low survival rate


person_dummies_train_data = pd.get_dummies(train_data['Person'])
person_dummies_train_data.columns = ['Child','Male','Female']
person_dummies_train_data.drop(['Male'],axis = 1 , inplace = True)
# print(person_dummies_train_data)
person_dummies_test_data = pd.get_dummies(test_data['Person'])
person_dummies_test_data.columns = ['Child','Male','Female']
person_dummies_test_data.drop(['Male'],axis = 1 , inplace = True)
### Dropping person column from traindata & test data
train_data.drop(['Person'],inplace=True,axis=1)
test_data.drop(['Person'],inplace=True,axis=1)


train_data =train_data.join(person_dummies_train_data)
test_data  =test_data.join(person_dummies_test_data)
# print(train_data)
# print(test_data)

####MODEL

##### define training and test sets
X_Train = train_data.drop("Survived", axis = 1) # removing target varaible from train data
# print(X_Train)
Y_Train = train_data["Survived"]                # separating target varaible and placing it in another varaible
# print(Y_Train)
X_test = test_data

# print(X_test)
# print(train_data.shape,test_data.shape)

# # #### Applying logistic regression model
logreg = LogisticRegression()
logreg.fit(X_Train,Y_Train)

Y_Pred = logreg.predict(X_test)
print(Y_Pred)
print(logreg.score(X_test,Y_Pred))
### 100 % accuracy

# Applying SVC algorithm
svc = SVC()
svc.fit(X_Train,Y_Train)
Y_Pred =svc.predict(X_test)
print(svc.score(X_test,Y_Pred))
# 100 accuracy

# Applying random forest algorithm
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_Train, Y_Train)
Y_pred = random_forest.predict(X_test)
print(random_forest.score(X_test, Y_Pred))
# 73% accuracy

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_Train, Y_Train)
Y_pred = knn.predict(X_test)
print(knn.score(X_test, Y_Pred))
## 83% accuracy

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_Train, Y_Train)
Y_pred = gaussian.predict(X_test)
print(gaussian.score(X_test, Y_Pred))
##82 % accuracy