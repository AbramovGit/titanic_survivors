from sklearn import tree, model_selection
import graphviz
from sklearn import ensemble
from sklearn import preprocessing
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

filename = 'weights/titanic_survivor_grad_boost_classifier.pkl'


def convert_csv_to_arrays(path_to_csv: str, train: bool):
    original_df = pd.read_csv(path_to_csv, sep=',', header=0)\
        .drop('Ticket', axis=1)\
        .drop('Cabin', axis=1)
    original_df.interpolate(inplace=True)

    if train:
        out_df = original_df['Survived'].copy()
        in_df = original_df.drop('Survived', axis=1)
    else:
        in_df = original_df.copy()

    in_df = in_df
    in_df['Sex'] = in_df['Sex'].map(lambda x: 1 if x == 'male' else 0)
    lb = preprocessing.LabelEncoder()
    min_max_scaler = preprocessing.MaxAbsScaler()

    in_df['Embarked'] = lb.fit_transform(in_df['Embarked'])
    in_df['Name'] = lb.fit_transform(in_df['Name'])
    # in_df['Ticket'] = lb.fit_transform(in_df['Ticket'])
    # in_df['Cabin'] = lb.fit_transform(in_df['Cabin'])
    in_array = in_df.drop('PassengerId', axis=1).to_numpy()
    if train:
        out_array = out_df.to_numpy()
        return out_array, in_array
    return in_array, original_df['PassengerId']


if __name__ == '__main__':
    out_array, in_array = convert_csv_to_arrays('data/train.csv', True)
    x_train, x_test, y_train, y_test = train_test_split(in_array, out_array,
                                                        shuffle=True,
                                                        test_size=0.3)
    param_grid = {'criterion': ['squared_error', 'friedman_mse'],
                  'learning_rate': [0.1, 0.2, 0.3],
                  'max_depth': [2, 4, 6, 8, 10, None],
                  'max_features': [None, 1.0],
                  'n_estimators': [50, 100, 200],
                  'min_impurity_decrease': [0.0, 0.1],
                  'validation_fraction': [0.1, 0.2],
                  'random_state': [0, None]
                  }
    clf = model_selection.GridSearchCV(ensemble.GradientBoostingClassifier(), scoring='roc_auc', param_grid=param_grid)
    clf.fit(x_train, y_train)
    print(f'Accuracy is:           {100 * clf.score(x_test, y_test):.2f}%\n')

