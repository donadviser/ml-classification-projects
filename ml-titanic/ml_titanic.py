# Titanic - Machine Learning from Disaster

## Objectives
# Classify if a person would survive or not.

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Transformers
from sklearn.preprocessing import (
    LabelEncoder, 
    OneHotEncoder,
    StandardScaler, 
    MinMaxScaler,
)

# Modelling Evaluation
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
    GridSearchCV,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


#Pipelines
from sklearn.pipeline import (
    Pipeline,
    FeatureUnion,
)

from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)



from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# file and data management
import urllib.request
import zipfile


def extract_zip(src, dst, member_name):
    """Function to extract a member file from a zip file and read it into a pandas 
    DataFrame.

    Parameters:
        src (str): URL of the zip file to be downloaded and extracted.
        dst (str): Local file path where the zip file will be written.
        member_name (str): Name of the member file inside the zip file 
            to be read into a DataFrame.

    Returns:
        pandas.DataFrame: DataFrame containing the contents of the 
            member file.

    usage:
        raw = extract_zip(url, fname, member_name)

    example:
        url = 'https://github.com/mattharrison/datasets/raw/master/data/kaggle-survey-2018.zip'
        fname = 'kaggle-survey-2018.zip'
        member_name = 'multipleChoiceResponses.csv'
    """    
    url = src
    fname = dst
    fin = urllib.request.urlopen(url)
    data = fin.read()
    with open(dst, mode='wb') as fout:
        fout.write(data)
    with zipfile.ZipFile(dst) as z:
        kag = pd.read_csv(z.open(member_name))
        kag_questions = kag.iloc[0]
        raw = kag.iloc[1:]
        return raw
    

# Load the Titanic dataset
data_path = "https://github.com/donadviser/datasets/raw/master/data/titanic3.xls"
titanic_raw = pd.read_excel(data_path)

# Data Analysis and Preprocessing

(titanic_raw
    # .head(10)  # View the first 10 rows
    # .shape  # Get the dimensions of the dataframe
    # .info()  # Get data types, memory usage, and non-null values
    # .describe(include='all').T  # Summary statistics for all columns, transposed
    .columns  # List column names
    # .value_counts(dropna=False)  # Count unique values in each column, including NaN
    # .nunique()  # Count unique values in each column (excluding NaN)
    # .isnull().sum()  # Check for missing values in each column
    # .duplicated().sum()  # Check for duplicate rows
    # .hist()  # Plot histograms of numerical columns
    # .corr()  # Calculate the correlation matrix between numerical columns
    # .sort_values(by='column_name')  # Sort by a specific column
    # .groupby('column_name').agg(function)  # Group data and apply aggregate function
    # .sample(5, random_state=42)  # Get a random sample of 5 rows
    # .groupby('speaker_id').agg(mean_pitch=pd.NamedAgg(column='pitch', aggfunc='mean'))  # Custom aggregation
    # .pivot_table(values='duration', index='speaker_id', columns='emotion', aggfunc='mean')  # Pivot table
    # .resample('1S').mean().rolling(window=10).mean()  # Time series analysis (if applicable)
    # .apply(pd.to_numeric, errors='coerce')  # Attempt numeric conversion for potential mixed-type columns
    # .select_dtypes(include=['category']).head()  # Explore categorical columns
    # .plot(kind='box', subplots=True, layout=(3, 3), figsize=(15, 10))  # Boxplots for visual analysis
    # .corrwith(titanic_raw['target_column'])  # Explore correlations with a target column
)

# Clean the dataset using Pandas chaining method
"""This code performs the following cleaning operations:

Drops unnecessary columns that are not useful for classification (PassengerId, Name, Ticket, and Cabin).
Fills missing values in the 'Age' column with the median age.
Fills missing values in the 'Embarked' column with the most frequent value.
Creates a new column 'FamilySize' by adding the 'SibSp' and 'Parch' columns.
Creates a new column 'IsAlone' to indicate whether the passenger is alone or not.
Converts categorical variables (Pclass, Sex, and Embarked) into dummy variables.
Drops redundant columns (SibSp, Parch, and FamilySize) and columns with missing values.
Rename the feature "survived" to "target"
Remove the outliers for features age and fare using the IQR method
The resulting dataset should be suitable for classification using machine learning algorithms."""

# Data Cleaning and Preprocessing
"""__Exploratory Data Analysis (EDA)__ is one of the crucial step in data science that allows us to achieve certain insights 
and statistical measures that are essential for prediction. Since we have so much to discuss, we will keep this section short! 
but always make it a habit to spend some time analysing and cleaning the dataset before training the data.
"""

def handle_outliers(df, cols):
    """Handles outliers in multiple columns using the IQR method."""
    q1 = df[cols].quantile(0.25)
    q3 = df[cols].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return df.loc[~((df[cols] < lower_bound) | (df[cols] > upper_bound)).any(axis=1)]

titanic = (titanic_raw
            # Drop unnecessary columns
            .drop(['name', 'ticket', 'cabin', 'boat', 'body', 'home.dest'], axis=1)
            # Fill missing values in the 'age', 'embarked' and 'fare columns with the median age, mode and mean respectively
            .assign(age=lambda x: x['age'].fillna(x['age'].median()),
                    embarked=lambda x: x['embarked'].fillna(x['embarked'].mode()[0]),
                    fare=lambda x: x['fare'].fillna(x['fare'].mean())
                    )
            # Create a new column 'family_size' by adding the 'sibsp' and 'parch' columns
            .assign(family_size=lambda x: x['sibsp'] + x['parch'])
            # Create a new column 'is_alone' to indicate whether the passenger is alone or not
            .assign(is_alone=lambda x: x['family_size'].apply(lambda x: 1 if x == 0 else 0))
            # Convert categorical variables into dummy variables or into numerical, use .assign below
            #.pipe(pd.get_dummies, columns=['pclass', 'sex', 'embarked'])
            .assign(sex=lambda x: x['sex'].map({'male': 0, 'female': 1}),
                    embarked=lambda x: x['embarked'].map({'S': 0, 'C': 1, 'Q': 2})
                    )
            # Drop redundant columns and columns with missing values
            .drop(['sibsp', 'parch', 'family_size'], axis=1)
            .dropna()
            # set the 'Survived' column as the target variable
            .rename(columns={'survived': 'target'})
            # Remove outliers in 'age' and 'fare' simultaneously
            .pipe(handle_outliers, ['age', 'fare'])
            .reset_index(drop=True)
            #.info()
            #.describe()
            #. head(10)
        )

sns.histplot(titanic['age'],kde=True)
print("Skew: ",titanic['age'].skew())
print("Kurtosis: ",titanic['age'].kurt())

sns.boxplot(titanic['age'])

pd.crosstab(titanic['sex'], titanic['target']).apply(lambda r: round((r/r.sum())*100,1), axis=1)


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
    
 
 


def explore_data(dataframe, method=None, **kwargs):
    """
    Explore a DataFrame using various methods.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame to explore.
        method (str): The exploration method to apply.
        **kwargs: Additional keyword arguments for the method.

    Returns:
        pd.DataFrame or Series or None: The result of the exploration method.
    """
    if method == 'head':
        return dataframe.head(**kwargs)
    elif method == 'shape':
        return dataframe.shape
    elif method == 'info':
        return dataframe.info()
    elif method == 'describe':
        return dataframe.describe(include='all').T
    elif method == 'columns':
        return dataframe.columns
    elif method == 'value_counts':
        return dataframe.value_counts(dropna=False)
    elif method == 'nunique':
        return dataframe.nunique()
    elif method == 'isnull_sum':
        return dataframe.isnull().sum()
    elif method == 'duplicated_sum':
        return dataframe.duplicated().sum()
    elif method == 'hist':
        return dataframe.hist(**kwargs)
    elif method == 'corr':
        return dataframe.corr()
    elif method == 'sort_values':
        return dataframe.sort_values(by=kwargs.get('by'))
    elif method == 'groupby_agg':
        return dataframe.groupby(kwargs.get('by')).agg(kwargs.get('agg_function'))
    elif method == 'sample':
        return dataframe.sample(**kwargs)
    elif method == 'pivot_table':
        return dataframe.pivot_table(**kwargs)
    elif method == 'resample_mean_rolling_mean':
        return dataframe.resample('1S').mean().rolling(window=10).mean()
    elif method == 'apply_numeric_conversion':
        return dataframe.apply(pd.to_numeric, errors='coerce')
    elif method == 'select_dtypes':
        return dataframe.select_dtypes(include=[kwargs.get('include')]).head()
    elif method == 'plot_box':
        return dataframe.plot(kind='box', subplots=True, layout=(3, 3), figsize=(15, 10))
    elif method == 'corr_with_target_column':
        return dataframe.corrwith(dataframe[kwargs.get('target_column')])
    else:
        print("Invalid method specified.")
        return None

 
# Call the function with the desired method and parameters
#result_head = explore_data(titanic, method='head', n=10)
#print(result_head)

#result_shape = explore_data(titanic, method='duplicated_sum')
#print(result_shape)
    
def split_data_to_dataframes(data, train_size=0.7, random_seed=None):
    """
    Splits data into train and test DataFrames with random sampling.

    Args:
        data: An array-like object or a pandas DataFrame containing the data to split.
        train_size: The proportion of data to be used for training (between 0 and 1).
        random_seed: An optional integer seed for reproducibility.

    Returns:
        train_df: A pandas DataFrame containing the training data.
        test_df: A pandas DataFrame containing the testing data.
    """

    # Convert data to NumPy array
    data = np.asarray(data)

    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # Shuffle data before splitting
    np.random.shuffle(data)

    # Calculate split indices
    train_size = int(train_size * len(data))
    test_size = len(data) - train_size

    # Split data into train and test sets
    train_data = data[:train_size]
    test_data = data[train_size:]

    # Split data into train and test sets (as DataFrames)
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    return train_df, test_df

train_data, test_data = split_data_to_dataframes(data)

# Defining variables for the columns in the dataframe to perform a train test split.
    
numerical_columns = ['age', 'fare']
categorical_columns = ["pclass", "sex",
                       "sibsp", "parch", "embarked"]


#Creating ss transformer to scale the continuous numerical data with StandardScaler()
numeric_transformer = Pipeline(
    steps=[('imputer', SimpleImputer(strategy='mean')),
           ('scaler', StandardScaler())])
 
#Creating ohe transformer to encode the categorical data with OneHotEncoder()
#categorical_transformer = Pipeline(steps=[('ohe', OneHotEncoder(drop='first'))])
categorical_transformer = Pipeline(
    steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
           ('onehot', OneHotEncoder(handle_unknown='ignore'))])
 
#Creating preprocess column transformer to combine the ss and ohe pipelines


preprocessor = ColumnTransformer(
    transformers=[('num', numeric_transformer, numerical_columns),
                  ('cat', categorical_transformer, categorical_columns)])

# Creating evaluation function to plot a confusion matrix and return the accuracy, precision, recall, and f1 scores
def evaluation(y, y_hat, title = 'Confusion Matrix'):
    cm = confusion_matrix(y, y_hat)
    precision = precision_score(y, y_hat)
    recall = recall_score(y, y_hat)
    accuracy = accuracy_score(y,y_hat)
    f1 = f1_score(y,y_hat)
    print('Recall: ', recall)
    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('F1: ', f1)
    sns.heatmap(cm,  cmap= 'PuBu', annot=True, fmt='g', annot_kws=    {'size':20})
    plt.xlabel('Predicted', fontsize=18)
    plt.ylabel('Actual', fontsize=18)
    plt.title(title, fontsize=18)
    
    plt.show();

# Performing train_test_split on the data
y = train["survived"]
X = train.drop(['survived'], axis=1)

y = LabelEncoder().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= .8, random_state=42)


"""
Creating cross_validate function
Defines the full pipeline with the preprocess and classifier pipelines
Loop through each fold in the cross validator (default is 5)
Fit the classifier on the train set, train_ind (prevents data leakage from test set)
Predict on the training set
Predict on the validation set
Print out an evaluation report containing a confusion matrix and the mean accuracy scores for both train and validation sets
"""

def cross_validate(classifier, cv):
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    train_acc = []
    test_acc = []
    for train_ind, val_ind in cv.split(X_train, y_train):
        X_t, y_t = X_train.iloc[train_ind], y_train[train_ind]
        pipeline.fit(X_t, y_t)
        y_hat_t = pipeline.predict(X_t)
        train_acc.append(accuracy_score(y_t, y_hat_t))
        X_val, y_val = X_train.iloc[val_ind], y_train[val_ind]
        y_hat_val = pipeline.predict(X_val)
        test_acc.append(accuracy_score(y_val, y_hat_val))
    print(evaluation(y_val, y_hat_val))
    print('Training Accuracy: {}'.format(np.mean(train_acc)))
    print('\n')
    print('Validation Accuracy: {}'.format(np.mean(test_acc)))
    print('\n')

    cross_validate(RandomForestClassifier(), KFold())