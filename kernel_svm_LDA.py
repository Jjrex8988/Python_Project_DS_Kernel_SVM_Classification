# Kernel SVM

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values


# Encoding categorical data
# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# One Hot Encoding the "Geography" column
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder="passthrough")
X = np.array(ct.fit_transform(X))


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
print(X_train)
print('-' * 38)
print(y_train)
print('-' * 38)
print(X_test)
print('-' * 38)
print(y_test)
print('-' * 38)


# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)
print('-' * 38)
print(X_test)
print('-' * 38)


# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=1)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)


# Training the Kernel SVM model on the Training set
from sklearn.svm import SVC

classifier = SVC(kernel='rbf', random_state=0) # Radial Basis Function or Gaussian
classifier.fit(X_train, y_train)


# Predicting a new result
# print(classifier.predict(sc.transform([[30, 87000]])))
# print(classifier.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
# print('-' * 38)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(y_pred)
print('-' * 38)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
print('-' * 38)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
print(cm)
print('-' * 38)
print(accuracy_score(y_test, y_pred))
print('-' * 38)



