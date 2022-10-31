import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('train.csv')

def fill_in_na_values(cols):
    age = cols[0]
    pclass = cols[1]
    if pd.isnull(age):
        if pclass == 1:
            return round(train_data[train_data['Pclass'] == 1]['Age'].mean())
        elif pclass == 2:
            return round(train_data[train_data['Pclass'] == 2]['Age'].mean())
        elif pclass == 3:
            return round(train_data[train_data['Pclass'] == 3]['Age'].mean())
    else:
        return age

train_data['Age'] = train_data[['Age', 'Pclass']].apply(fill_in_na_values, axis=1)
train_data.drop(['Cabin'], axis=1, inplace=True)
train_data.dropna(inplace=True)
train_data.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

sex = pd.get_dummies(train_data['Sex'], drop_first=True)
embarked = pd.get_dummies(train_data['Embarked'], drop_first=True)

train_data.drop(['Sex', 'Embarked'], axis=1, inplace=True)
train_data = pd.concat([train_data, sex, embarked], axis=1)

X = train_data.drop('Survived', axis=1)
Y = train_data['Survived']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.svm import SVC
# svm = SVC()
# svm.fit(X_train, Y_train)
# predictions = svm.predict(X_test)
# print(classification_report(Y_test, predictions))
# print(confusion_matrix(Y_test, predictions))

# from sklearn.model_selection import GridSearchCV
# param_grid = {'C': [0.5, 1, 10, 50, 100, 1000], 'gamma': [1, 0.1, 0.001, 0.0001, 0.00001, 0.000001]}
# grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
# grid.fit(X_train, Y_train)
# grid_predictions = grid.predict(X_test)
# print(classification_report(Y_test, grid_predictions))
# print(confusion_matrix((Y_test, grid_predictions)))

# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression()
# lr.fit(X_train, Y_train)
# lr_predictions = lr.predict(X_test)
# print(classification_report(Y_test, lr_predictions))
# print(confusion_matrix((Y_test, lr_predictions)))

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
knn_predictions = knn.predict(X_test)
print(classification_report(Y_test, knn_predictions))
print(confusion_matrix((Y_test, knn_predictions)))


