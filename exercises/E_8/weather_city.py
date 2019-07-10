import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from  sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
import sys


def transform(X):
    scaler = StandardScaler()
    scaler.fit(X)
    # X.values = pd.to_numeric(X.values, downcast='float')
    return scaler.transform(X)


def main():
    labelled = pd.read_csv(sys.argv[1])
    unlabelled = pd.read_csv(sys.argv[2])
    # labelled = pd.read_csv("monthly-data-labelled.csv")
    # unlabelled = pd.read_csv("monthly-data-unlabelled.csv")

    X_labelled = labelled.drop(columns="city")
    y_labelled = labelled["city"]
    X_train, X_valid, y_train, y_valid = train_test_split(X_labelled, y_labelled)

    model = make_pipeline(
        FunctionTransformer(transform, validate=False),
        SVC(kernel='linear', C=500, gamma='auto')
    )
    model.fit(X_train, y_train)
    unlabelled = transform(unlabelled.drop(columns="city"))
    prediction = model.predict(unlabelled)
    pd.Series(prediction).to_csv(sys.argv[3], index=False, header=False)
    print(model.score(X_valid, y_valid))
    #df = pd.DataFrame({'truth': y_valid, 'prediction': model.predict(X_valid)})
    #print(df[df['truth'] != df['prediction']])


if __name__ == '__main__':
    main()
