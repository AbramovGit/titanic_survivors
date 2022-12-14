from sklearn import svm, ensemble
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd


def convert_csv_to_arrays(path_to_csv: str, train: bool):
    original_df = pd.read_csv(path_to_csv, sep=',', header=0)
    if train:
        out_df = original_df['Survived'].copy()
        in_df = original_df.drop('Survived', axis=1)
    else:
        in_df = original_df.copy()
    in_df['Sex'] = in_df['Sex'].map(lambda x: 1 if x == 'male' else 0)
    lb = preprocessing.LabelEncoder()
    min_max_scaler = preprocessing.MaxAbsScaler()

    in_df['Ticket'] = lb.fit_transform(in_df['Ticket'])
    in_df['Name'] = lb.fit_transform(in_df['Name'])
    in_df['Cabin'] = lb.fit_transform(in_df['Cabin'])
    in_df['Embarked'] = lb.fit_transform(in_df['Embarked'])
    in_array = in_df.drop('PassengerId', axis=1).to_numpy()
    in_array = min_max_scaler.fit_transform(in_array)
    if train:
        out_array = out_df.to_numpy()
        return out_array, in_array
    return in_array


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    out_array, in_array = convert_csv_to_arrays('data/train.csv', True)
    x_train, x_test, y_train, y_test = train_test_split(in_array, out_array,
                                                        shuffle=True,
                                                        test_size=0.3)
    clf = ensemble.HistGradientBoostingClassifier()
    clf.fit(x_train, y_train)
    accuracy = 0
    print(x_test[0].reshape(1, -1))

    for i in range(0, len(y_test)):
        if y_test[i] == clf.predict(x_test[i].reshape(1, -1)):
            accuracy += 1
    print(f'accuracy:{100 * accuracy/len(y_test):.2f}%')
