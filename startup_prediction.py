from __future__ import print_function

import argparse
import os
import pandas as pd

from sklearn import tree
from sklearn.externals import joblib


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Sagemaker specific arguments. Defaults are set in the environment variables.

    #Saves Checkpoints and graphs
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

    #Save model artifacts
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    #Train data
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    file = os.path.join(args.train, "emails.csv")
    df = pd.read_csv(file, engine="python")
    #df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # labels are in the first column
    X = df.iloc[:,1:3001]

    Y = df.iloc[:,-1].values

    # Encoding categorical data
    #from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    #labelencoder = LabelEncoder()
    #X[:, 3] = labelencoder.fit_transform(X[:, 3])
    #onehotencoder = OneHotEncoder(categorical_features = [3])
    #X = onehotencoder.fit_transform(X).toarray()

    # Avoiding the Dummy Variable Trap
    #X = X[:, 1:]
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=100,criterion='entropy', min_samples_split=2) # gini, log_loss, entropy
    # n_estimators = No. of trees in the forest
    # criterion = basis of making the decision tree split, either on gini impurity('gini'), or on infromation gain('entropy')
    rfc.fit(X_train,y_train)

    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(rfc, os.path.join(args.model_dir, "model.joblib"))


def model_fn(model_dir):
    """Deserialized and return fitted model
    
    Note that this should have the same name as the serialized model in the main method
    """
    rfc = joblib.load(os.path.join(model_dir, "model.joblib"))
    return rfc
