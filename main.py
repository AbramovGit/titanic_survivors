from sklearn import tree, model_selection, naive_bayes
import matplotlib.pyplot as plt
import numpy as np
from sklearn import ensemble
from sklearn import preprocessing
import xgboost as xgb
from sklearn.model_selection import train_test_split, ShuffleSplit, LearningCurveDisplay
import joblib
import pandas as pd

filename = 'weights/titanic_survivor_grad_boost_classifier.pkl'


def convert_csv_to_arrays(path_to_csv: str, train: bool):
    original_df = pd.read_csv(path_to_csv, sep=',', header=0)\
        .drop('Ticket', axis=1)
    original_df.interpolate(inplace=True)

    if train:
        out_df = original_df['Survived'].copy()
        in_df = original_df.drop('Survived', axis=1)
    else:
        in_df = original_df.copy()

    in_df = in_df
    in_df['Sex'] = in_df['Sex'].map(lambda x: 1 if x == 'male' else 0)
    label_encoder = preprocessing.LabelEncoder()
    names_encoder = preprocessing.OrdinalEncoder()
    cabin_encoder = preprocessing.LabelEncoder()

    in_df['Embarked'] = label_encoder.fit_transform(in_df['Embarked'])
    in_df['Name'] = names_encoder.fit_transform(in_df['Name'].values.reshape(-1, 1))
    in_df['Cabin'] = cabin_encoder.fit_transform(in_df['Cabin'].values.reshape(-1, 1))
    print(in_df.head(5))
    X = in_df.drop('PassengerId', axis=1).to_numpy()
    if train:
        y = out_df.to_numpy()
        return y, X
    return X, original_df['PassengerId']


if __name__ == '__main__':
    y, X = convert_csv_to_arrays('data/train.csv', True)
    x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                        shuffle=True,
                                                        test_size=0.3)
    param_grid = {
                  'criterion': ['squared_error', 'friedman_mse'],
                  'learning_rate': [0.1, 0.2, 0.3],
                  'max_depth': [2, 4, 6, 8, 10, None],
                  'max_features': [None, 1.0],
                  'n_estimators': [50, 100, 200],
                  'min_impurity_decrease': [0.0, 0.1],
                  'validation_fraction': [0.1, 0.2],
                  'random_state': [0, None]
                  }
    gb_clf = model_selection.GridSearchCV(ensemble.GradientBoostingClassifier(), scoring='roc_auc', param_grid=param_grid)
    rf_clf = model_selection.GridSearchCV(ensemble.RandomForestClassifier(), scoring='roc_auc', param_grid=param_grid)
    gb_clf.fit(x_train, y_train)
    rf_clf.fit(x_train, y_train)
    print(f'Accuracy of gb is:           {100 * gb_clf.score(x_test, y_test):.2f}%\n')
    print(f'Accuracy of rf is:           {100 * rf_clf.score(x_test, y_test):.2f}%\n')

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    common_params = {
        "X": X,
        "y": y,
        "train_sizes": np.linspace(0.1, 1.0, 5),
        "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=0),
        "score_type": "both",
        "n_jobs": 4,
        "line_kw": {"marker": "o"},
        "std_display_style": "fill_between",
        "score_name": "Accuracy",
    }
    for ax_idx, estimator in enumerate([gb_clf, rf_clf]):
        LearningCurveDisplay.from_estimator(estimator, **common_params, ax=ax[ax_idx])
        handles, label = ax[ax_idx].get_legend_handles_labels()
        ax[ax_idx].legend(handles[:2], ["Training Score", "Test Score"])
        ax[ax_idx].set_title(f"Learning Curve for {estimator.__class__.__name__}")

    plt.show()


