import numpy as np
import pandas as pd

def generalized_VIF(df):
    '''
    Calculates the Generalized VIF (GVIF, Fox and Monette 1992) for a data set. GVIF ** (1 / (2 * Df)) ** 2 < 5 is equivalent to VIF.
    The function assumes that categorical data are typed as 'category' or 'object' and automatically performs one-hot encoding. The function
    will work properly if the data frame has columns previously one-hot encoded from binary data, but it will not work properly if the data 
    frame has multi-nomial columns that have been previously one-hot encoded.

    Args:
        df (pandas data frame): Data frame with the response column(s) removed

    Returns:
        pandas data frame: a data frame, indexed by factor of the GVIF, GVIF^(1/2Df), VIF^(1/2Df)^2 
        dictionary: Dictionary of column names (keys) and GVIF ** (1 / (2 * Df)) ** 2 (values)
    '''

    # Save categorical column names, append with prefix
    onehot_list = list(df.select_dtypes(include=['category', 'object', 'string']).columns)

    # Since we do not include all of the indicator variables in the model so as to avoid the dummy variable trap, one of the indicator variables is dropped
    df_1hot = pd.get_dummies(df, drop_first=True, dummy_na=False, prefix_sep='_')

    # Create empty df to store GVIF results
    gvif_df = pd.DataFrame(columns = ['factor', 'GVIF', 'GVIF^(1/2Df)', 'GVIF^(1/2Df)^2'])

    # Iterate over columns
    for (columnName, columnData) in df.iteritems():

        # Select predictor as response: if dummy encoded, select all columns for variable
        # Could all be done in the first condition, but that could result in incorrect column selection with similar column names
        if columnName in onehot_list:
            X1 = df_1hot.loc[:, df_1hot.columns.str.startswith(columnName)]
            X2 = df_1hot.loc[:, ~df_1hot.columns.str.startswith(columnName)]
        else:
            X1 = df_1hot[[columnName]].values
            X2 = df_1hot.loc[:, df_1hot.columns != columnName].values

        # Calculate gvif
        gvif = np.linalg.det(np.array(np.corrcoef(X1, rowvar=False), ndmin=2)) * np.linalg.det(np.corrcoef(X2, rowvar=False)) / np.linalg.det(np.corrcoef(np.append(X1, X2, axis=1), rowvar=False))

        gvif_12df = np.power(gvif, 1 / (2 * X1.shape[1]))
        gvif_12df_sq = gvif_12df ** 2

        # Update results df
        new_row = {'factor': columnName, 'GVIF': gvif, 'GVIF^(1/2Df)': gvif_12df, 'GVIF^(1/2Df)^2': gvif_12df_sq}
        gvif_df = gvif_df.append(new_row, ignore_index=True)

    gvif_df = gvif_df.set_index('factor')
    gvif_filter = gvif_df.loc[gvif_df['GVIF^(1/2Df)^2'] >= 5]['GVIF^(1/2Df)^2'].to_dict()

    return gvif_df, gvif_filter
