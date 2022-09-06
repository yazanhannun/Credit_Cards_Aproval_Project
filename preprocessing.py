import pandas as pd
import numpy as np

def prepro(x):
    import pandas as pd
    import numpy as np
    # x = pd.read_csv(data_pre, index_col=0)
    
    # Import train_test_split
    from sklearn.model_selection import train_test_split
    #----------------------------------------------------------------------
    # Replace the '?'s with NaN in the train and test sets
    x = x.replace('?', np.NaN)
    # #----------------------------------------------------------------------
    # Impute the missing values with mean imputation
    x.fillna(x[['2', '7', '10', '14']].mean(), inplace=True)
    #----------------------------------------------------------------------
    '''
    you will get a series that is sorted, to access the most frequent items, use index zero since the series
    will be ordered in decending order
     
     Example:
     
     print(cc_apps_train[0].value_counts().index[0]) # you will get a series here'''

    # Iterate over each column of cc_apps_train
    for col in x.columns:
    # Check if the column is of object type
        if x[col].dtypes == 'object':
        # Impute with the most frequent value
            x = x.fillna(x[col].value_counts().index[0])

    # Import LabelEncoder
    from sklearn.preprocessing import LabelEncoder

    # Instantiate LabelEncoder
    le = LabelEncoder()

    # Iterate over all the values of each column and extract their dtypes
    for col in x.columns:
    # Compare if the dtype is object
        if x[col].dtypes=='object':
    # Use LabelEncoder to do the numeric transformation
            x[col]=le.fit_transform(x[col])

    #----------------------------------------------------------------------
    # Since our values for targeted one after label encoder '+' will be 0 and '-' will be 1, this logically not accepted
    # for this reason we will reverse this issue using for loop
    s = x.shape
   
    for i in range(0, s[0]):
        if x.iloc[i, 13] == 1:
            x.iloc[i, 13] = 0
        else:
            x.iloc[i, 13] = 1

    #----------------------------------------------------------------------

    # Import MinMaxScaler
    from sklearn.preprocessing import MinMaxScaler

    # Segregate features and labels into separate variables
    X_to_pred = x.iloc[:, :-1].values

    #Instantiate MinMaxScaler and use it to rescale X_train and X_test
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaledX_to_pred = scaler.fit_transform(X_to_pred)

    return rescaledX_to_pred