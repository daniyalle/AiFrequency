# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset

dataset = pd.read_csv('data.csv')
x = dataset.iloc[:, 1:31].values
y = dataset.iloc[:, 31].values

# Find the dimension of the dataset

dataset.head()
print('Cancer data set dimension: {}'.format(dataset.shape))

# Find missing or null data points

dataset.isnull().sum()
dataset.isna().sum()

# Encoding categorical data values

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
#print(y)

# Splitting the dataset into the training set and test set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Using Logistic Regression Algorithm to training set

#from sklearn.linear_model import LogisticRegression
#classifier = LogisticRegression(random_state=0)
#classifier.fit(x_train, y_train)

# Using KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm

#from sklearn.neighbors import KNeighborsClassifier
#classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
#classifier.fit(x_train, y_train)

# Using SVC method of svm class to use Support Vector Machine Algorithm

from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(x_train, y_train)

# Using SVC method of svm class to use Kernel SVM Algorithm

#from sklearn.svm import SVC
#classifier = SVC(kernel='rbf', random_state=0)
#classifier.fit(x_train, y_train)

# Using GaussianNB method of naive bayes class to use Naive Bayes Algorithm

#from sklearn.naive_bayes import GaussianNB
#classifier = GaussianNB()
#classifier.fit(x_train, y_train)

# Using DecisionTreeClassifier of tree class to use Decision Tree Algorithm

#from sklearn.tree import DecisionTreeClassifier
#classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
#classifier.fit(x_train, y_train)

# Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm

#from sklearn.ensemble import RandomForestClassifier
#classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
#classifier.fit(x_train, y_train)

# Predict the test result

y_pred = classifier.predict(x_test)

# Check the accuracy

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Accuracy results:
# Logistic Regression : 96%
# Nearest Neighbor : 95%
# Support Vector Machine : 98%
# Kernel SVM : 98%
# Naive Bayes : 90%
# Decision Tree Algorithm : 92%
# Random Forest Classification : 97%
