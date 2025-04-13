import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import zscore

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder,StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# All preprocessing steps in functions:
def remove_minus_9999(df, features):
    return df[~(df[features] == -9999).any(axis=1)].copy()

def impute_u_by_class(df, target_col='class'):
    means = df.groupby(target_col)['u'].mean()
    for cls, mean_val in means.items():
        mask = (df[target_col] == cls) & (df['u'].isnull())
        df.loc[mask, 'u'] = mean_val
    return df

def remove_outliers_iqr(df, features, threshold=2.5):
    for col in features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df.copy()

def feature_engineer(df):
    df = df.copy()
    df['u_g'] = df['u'] - df['g']
    df['g_r'] = df['g'] - df['r']
    df['r_i'] = df['r'] - df['i']
    df['i_z'] = df['i'] - df['z']
    return df

def drop_columns(df, cols):
    return df.drop(columns=cols, errors='ignore')

DROP_COLS = ['rerun_ID', 'obj_ID', 'spec_obj_ID']
FILTER_COLS = ['u', 'g', 'r', 'i', 'z']

# Refine training data
final_df = pd.read_csv('train.csv')

le = LabelEncoder()
final_df['class'] = le.fit_transform(final_df['class'])

final_df = remove_minus_9999(final_df, FILTER_COLS)
final_df = impute_u_by_class(final_df, target_col='class')
final_df = remove_outliers_iqr(final_df, features=FILTER_COLS + ['MJD', 'redshift'])

# Split data
X = final_df.drop(columns='class')
y = final_df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)

# Complete preprocessing pipeline
scale_transform = ColumnTransformer([
    ('scale', StandardScaler(), make_column_selector(dtype_include='number'))
])

preprocess_pipe = Pipeline([
    ('feature_eng', FunctionTransformer(feature_engineer)),
    ('drop_cols', FunctionTransformer(lambda df: drop_columns(df, DROP_COLS))),
    ('scale', scale_transform)
])

# Pipeline 8: Voting with several classifiers

dt = DecisionTreeClassifier(max_depth=8, min_samples_split=4, min_samples_leaf=2, random_state=42)
rf = RandomForestClassifier(n_estimators=150, max_depth=16, min_samples_split=2, random_state=42, n_jobs=-1)
svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
gb = GradientBoostingClassifier(
    random_state=42,
    n_estimators=200,
    max_depth=5
)

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=10, min_samples_leaf=5),
    random_state=42
)

vote8 = VotingClassifier(
    estimators=[('gb', gb), ('ada', ada), ('dt', dt), ('rf', rf), ('svm', svm)],
    voting='soft',
    n_jobs=-1
)

pipe8 = Pipeline([
    ('pre', preprocess_pipe),
    ('vote', vote8)
])

pipe8.fit(X_train, y_train)

y_pred8 = pipe8.predict(X_test)
f1_macro8 = f1_score(y_test, y_pred8, average='macro')
print(f"Macro F1-score: {f1_macro8:.4f}")

print("\n Classification Report:")
print(classification_report(y_test, y_pred8, digits=4))


y_pred = pipe8.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='BuPu', values_format='d')
plt.title("Confusion Matrix")
plt.show()