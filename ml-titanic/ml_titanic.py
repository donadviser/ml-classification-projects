# Titanic Classification

## Objectives
# Classify if a person would survive or not.

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


class ClassifierTester:
    def __init__(self, data, target, test_size=0.3):
        self.data = data
        self.target = target
        self.test_size = test_size
        
    def test_classifiers(self):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.target, test_size=self.test_size, random_state=42)

        # Define the hyperparameters for each classifier
        rf_params = {'n_estimators': [10, 50, 100], 'max_depth': [None, 5, 10]}
        lr_params = {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}
        svc_params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly'], 'degree': [2, 3, 4]}
        knn_params = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance'], 'p': [1, 2]}
        nb_params = {}
        dt_params = {'criterion': ['gini', 'entropy'], 'max_depth': [None, 5, 10]}
        xgboost_params = {'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 1.0]}
        nn_params = {'batch_size': [16, 32, 64], 'epochs': [10, 20, 30]}

        # Instantiate the classifiers
        rf = RandomForestClassifier()
        lr = LogisticRegression()
        svc = SVC()
        knn = KNeighborsClassifier()
        nb = GaussianNB()
        dt = DecisionTreeClassifier()
        xgboost = XGBClassifier()
        nn = self.create_neural_network()

        # Define the grid search for each classifier
        rf_grid = GridSearchCV(rf, rf_params, cv=5, n_jobs=-1)
        lr_grid = GridSearchCV(lr, lr_params, cv=5, n_jobs=-1)
        svc_grid = GridSearchCV(svc, svc_params, cv=5, n_jobs=-1)
        knn_grid = GridSearchCV(knn, knn_params, cv=5, n_jobs=-1)
        nb_grid = GridSearchCV(nb, nb_params, cv=5, n_jobs=-1)
        dt_grid = GridSearchCV(dt, dt_params, cv=5, n_jobs=-1)
        xgboost_grid = GridSearchCV(xgboost, xgboost_params, cv=5, n_jobs=-1)
        nn_grid = GridSearchCV(nn, nn_params, cv=5, n_jobs=-1)

        # Train and test the classifiers
        classifiers = {'Random Forest': rf_grid, 'Logistic Regression': lr_grid, 'SVC': svc_grid, 'K-Nearest Neighbors': knn_grid, 'Naive Bayes': nb_grid, 'XGBoost': xgboost_grid}
        for name, clf in classifiers.items():
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='macro')
                recall = recall_score(y_test, y_pred, average='macro')
                f1 = f1_score(y_test, y_pred, average='macro')
                print(f'{name} Classifier Results:')
                print(f'Accuracy: {accuracy:.4f}')
                print(f'Precision: {precision:.4f}')
                print(f'Recall: {recall:.4f}')
                print(f'F1-Score: {f1:.4f}')
                print('-'*50)
                
    def create_neural_network(self):
        nn = Sequential([
            Dense(64, activation='relu', input_shape=(self.data.shape[1],)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(3, activation='softmax')
        ])
        nn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return nn
    
# Load the Titanic dataset into a Pandas dataframe
df = pd.read_csv('titanic.csv')
# Load the Titanic dataset
df = sns.load_dataset('titanic')
df.head()
# Clean the dataset using Pandas chaining method
df_clean = (
    df
    # Drop unnecessary columns
    .drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    # Fill missing values in the 'Age' column with the median age
    .fillna({'Age': df['Age'].median()})
    # Fill missing values in the 'Embarked' column with the most frequent value
    .fillna({'Embarked': df['Embarked'].mode()[0]})
    # Create a new column 'FamilySize' by adding the 'SibSp' and 'Parch' columns
    .assign(FamilySize=lambda x: x['SibSp'] + x['Parch'])
    # Create a new column 'IsAlone' to indicate whether the passenger is alone or not
    .assign(IsAlone=lambda x: x['FamilySize'].apply(lambda x: 1 if x == 0 else 0))
    # Convert categorical variables into dummy variables
    .pipe(pd.get_dummies, columns=['Pclass', 'Sex', 'Embarked'])
    # Drop redundant columns and columns with missing values
    .drop(['SibSp', 'Parch', 'FamilySize'], axis=1)
    .dropna()
)

# Print the cleaned dataset
print(df_clean.head())
"""This code performs the following cleaning operations:

Drops unnecessary columns that are not useful for classification (PassengerId, Name, Ticket, and Cabin).
Fills missing values in the 'Age' column with the median age.
Fills missing values in the 'Embarked' column with the most frequent value.
Creates a new column 'FamilySize' by adding the 'SibSp' and 'Parch' columns.
Creates a new column 'IsAlone' to indicate whether the passenger is alone or not.
Converts categorical variables (Pclass, Sex, and Embarked) into dummy variables.
Drops redundant columns (SibSp, Parch, and FamilySize) and columns with missing values.
The resulting dataset should be suitable for classification using machine learning algorithms."""


# chain the methods to clean the dataset
cleaned_df = (
    df
    # drop unnecessary columns
    .drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    # fill missing values with median
    .fillna({'Age': df['Age'].median(), 'Embarked': df['Embarked'].mode()[0]})
    # convert categorical columns to numerical
    .assign(Sex=lambda x: x['Sex'].map({'male': 0, 'female': 1}))
    .assign(Embarked=lambda x: x['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}))
    # create a new column for family size
    .assign(FamilySize=lambda x: x['SibSp'] + x['Parch'] + 1)
    # create a new column for whether the passenger was alone or not
    .assign(IsAlone=lambda x: x['FamilySize'].map(lambda size: 1 if size == 1 else 0))
    # drop the original columns
    .drop(['SibSp', 'Parch'], axis=1)
    # set the 'Survived' column as the target variable
    .rename(columns={'Survived': 'target'})
)

import pandas as pd
import seaborn as sns

# Load the Titanic dataset
titanic = sns.load_dataset('titanic')

# Chain the methods to clean and preprocess the data
clean_titanic = (
    titanic
    # Drop the unnecessary columns
    .drop(['alive', 'class', 'who', 'embark_town', 'deck'], axis=1)
    # Drop the rows with missing values
    .dropna()
    # Convert the categorical features into numerical
    .assign(
        sex=lambda x: x.sex.map({'male': 0, 'female': 1}),
        embarked=lambda x: x.embarked.map({'S': 0, 'C': 1, 'Q': 2})
    )
    # Remove the outliers using IQR method
    .loc[lambda x: (x['age'] > x['age'].quantile(0.25) - 1.5*(x['age'].quantile(0.75) - x['age'].quantile(0.25))) & 
                   (x['age'] < x['age'].quantile(0.75) + 1.5*(x['age'].quantile(0.75) - x['age'].quantile(0.25))) &
                   (x['fare'] > x['fare'].quantile(0.25) - 1.5*(x['fare'].quantile(0.75) - x['fare'].quantile(0.25))) &
                   (x['fare'] < x['fare'].quantile(0.75) + 1.5*(x['fare'].quantile(0.75) - x['fare'].quantile(0.25)))]
    # Set the index to the passenger ID
    .set_index('passenger_id')
)

# Visualize the cleaned data
sns.pairplot(clean_titanic, hue='survived')