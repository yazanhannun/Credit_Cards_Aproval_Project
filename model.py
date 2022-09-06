import pandas as pd
import numpy as np

def model(data):
    # Load dataset
    cc_apps = pd.read_csv('cc_approvals.data', header=None)

    #----------------------------------------------------------------------
    # Import train_test_split
    from sklearn.model_selection import train_test_split

    # Drop the features 11 and 13
    cc_apps = cc_apps.drop([11, 13], axis = 1)
    #----------------------------------------------------------------------
    # Replace the '?'s with NaN in the train and test sets
    cc_apps = cc_apps.replace('?', np.NaN)
    # #----------------------------------------------------------------------
    # Impute the missing values with mean imputation
    cc_apps.fillna(cc_apps[[2, 7, 10, 14]].mean(), inplace=True)
    #----------------------------------------------------------------------
    '''
    you will get a series here that is sorted, to access the most frequent items, use index zero since the series
    will be ordered in decending order'''
    # print(cc_apps_train[0].value_counts().index[0]) # you will get a series here

    # Iterate over each column of cc_apps_train
    for col in cc_apps.columns:
        # Check if the column is of object type
        if cc_apps[col].dtypes == 'object':
            # Impute with the most frequent value
            cc_apps = cc_apps.fillna(cc_apps[col].value_counts().index[0])

    # Import LabelEncoder
    from sklearn.preprocessing import LabelEncoder

    # Instantiate LabelEncoder
    le = LabelEncoder()

    # Iterate over all the values of each column and extract their dtypes
    for col in cc_apps.columns:
        # Compare if the dtype is object
        if cc_apps[col].dtypes=='object':
            # Use LabelEncoder to do the numeric transformation
            cc_apps[col]=le.fit_transform(cc_apps[col])

    # Since our values for targeted one after label encoder '+' will be 0 and '-' will be 1, this logically not accepted
    # for this reason we will reverse this issue using for loop

    for i in range(0, 690):
        if cc_apps.iloc[i, 13] == 1:
            cc_apps.iloc[i, 13] = 0
        else:
            cc_apps.iloc[i, 13] = 1


    cc_apps_train, cc_apps_test = train_test_split(cc_apps, test_size=0.33, random_state=42)

    # #----------------------------------------------------------------------

    # Import MinMaxScaler
    from sklearn.preprocessing import MinMaxScaler

    # Segregate features and labels into separate variables
    X_train = cc_apps_train.iloc[:, :-1].values
    y_train = cc_apps_train.iloc[:, [-1]].values
    # X_test = cc_apps_test.iloc[:, :-1].values
    # y_test =  cc_apps_test.iloc[:, [-1]].values

    #Instantiate MinMaxScaler and use it to rescale X_train and X_test
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaledX_train = scaler.fit_transform(X_train)

    y_train = y_train.reshape((462, ))

    #----------------------------------------------------------------------

    # Import KNeighborsClassifier
    from sklearn.neighbors import KNeighborsClassifier

    # Instantiate a KNeighborsClassifier with default parameter values
    knn = KNeighborsClassifier(n_neighbors=8)

    # Fit knn to the train set
    knn.fit(rescaledX_train, y_train)

    # Use knn to predict instances from the test set and store it
    y_pred = knn.predict(data)

    return y_pred

