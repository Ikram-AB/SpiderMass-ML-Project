import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, classification_report
import plotly.figure_factory as ff
import plotly.io as pio
import joblib
import os

# Import ML models
from sklearn.linear_model import SGDClassifier, RidgeClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC


def train_and_evaluate_pipeline(all_data, n_components, output_dir):

    # Split dataset into features and target
    X = all_data.drop(['Class', 'File', 'RT', 'Sum'], axis=1)
    y = all_data['Class']

    # Encode target labels if they are strings
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Identify numeric columns (spectrometry features are numeric)
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns

    # Define preprocessing for numeric features:
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                ('scaler', StandardScaler())
            ]), numeric_features)
        ],
        remainder='passthrough'
    )

    # Dimensionality reduction step (useful for high-dimensional data)
    svd = TruncatedSVD(n_components=n_components)

    # Define all models to compare
    models = {
        'SGD': SGDClassifier(),
        'RandomForestClassifier': RandomForestClassifier(n_estimators=100, max_depth=30),
        'AdaBoostClassifier': AdaBoostClassifier(n_estimators=100, algorithm="SAMME"),
        'Ridge Classifier': RidgeClassifierCV(),
        'SVC': SVC(C=1.0, kernel='rbf', degree=3, gamma='scale'),
        'Gaussian Naive Bayes': GaussianNB(),
        'Decision Tree Classifier': DecisionTreeClassifier(),
        'Hist Gradient Boosting Classifier': HistGradientBoostingClassifier(max_iter=100),
        'BernoulliNB': BernoulliNB()
    }

    pipelines = {}
    results = {}

    # Train and evaluate each model
    for name, model in models.items():

        # Build pipeline: preprocess -> SVD -> classifier
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('svd', svd),
            ('model', model)
        ])

        # Fit the pipeline on the full dataset
        pipeline.fit(X, y)

        # Use stratified CV to keep class distribution stable in folds
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=10)

        # Get cross-validated predictions
        y_pred = cross_val_predict(pipeline, X, y, cv=cv)

        # Compute confusion matrix
        conf_matrix = confusion_matrix(y, y_pred)

        # Compute a text report (precision, recall, F1-score)
        report = classification_report(y, y_pred)

        # Compute cross-validation scores (accuracy by default)
        scores = cross_val_score(pipeline, X, y, cv=cv)
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        # Create an interactive confusion matrix with Plotly
        classes = np.unique(y)
        fig = ff.create_annotated_heatmap(
            z=conf_matrix.tolist(),
            x=classes.tolist(),
            y=classes.tolist(),
            colorscale='Viridis',
            showscale=True
        )
        fig.update_layout(title=f'Confusion matrix for model {name}')

        # Save confusion matrix figure as a JSON file
        plot_path = os.path.join(output_dir, f'{name}_confusion_matrix.json')
        pio.write_json(fig, plot_path)

        # Store results for this model
        results[name] = {
            'classification_report': report,
            'mean_score': mean_score,
            'std_score': std_score,
            'confusion_matrix_json': plot_path
        }

        # Store the trained pipeline
        pipelines[name] = pipeline

    # Return pipelines, selected numeric features, and evaluation results
    return pipelines, numeric_features.tolist(), results


def save_selected_models(pipelines, name='', output_dir=''):
    if name in pipelines:
        save_path = os.path.join(output_dir, f'{name}_pipeline.pkl')
        joblib.dump(pipelines[name], save_path)
        print(f"Pipeline {name} saved as '{name}_pipeline.pkl'.")
    else:
        print(f"Model {name} not found in available pipelines.")


def save_mzstrain(numeric_features, mass_range_name='mzs_glioblastome', output_dir=''):
    save_path = os.path.join(output_dir, f'{mass_range_name}.pkl')
    joblib.dump(numeric_features, save_path)
    print(f"Numeric features saved as '{mass_range_name}.pkl'.")