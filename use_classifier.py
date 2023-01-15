import joblib
import pandas as pd

from main import filename, convert_csv_to_arrays

if __name__ == '__main__':
    clf = joblib.load(filename)
    in_array, ids = convert_csv_to_arrays('data/test.csv', False)
    print(ids)
    predictions = pd.Series(clf.predict(in_array), name='Survived')
    print(predictions)
    predictions.index = ids
    result_df = predictions
    print(result_df.head())
    result_df.to_csv('results/titanic_survivors_result.csv', sep=',', encoding='utf-8')

