import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess(df: pd.DataFrame, is_train=True, label_encoders=None):
    """Preprocess the dataset (both train and test) consistently."""

    # Define columns where NaN means 'None' or 0
    none_cols = [
        'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish',
        'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
        'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'Electrical'
    ]
    zero_cols = ['GarageYrBlt', 'MasVnrArea', 'BsmtFullBath', 'BsmtHalfBath',
                 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']

    for col in none_cols:
        if col in df.columns:
            df[col] = df[col].fillna('None')
    for col in zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Add binary features
    df['HasPool'] = (df['PoolQC'] != 'None').astype(int)
    df['HasGarage'] = (df['GarageType'] != 'None').astype(int)
    df['HasBasement'] = (df['BsmtQual'] != 'None').astype(int)
    df['HasFireplace'] = (df['FireplaceQu'] != 'None').astype(int)
    df['HasFence'] = (df['Fence'] != 'None').astype(int)

    # Impute LotFrontage by median per Neighborhood (only for training set)
    if 'LotFrontage' in df.columns and 'Neighborhood' in df.columns:
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )

    # Encode categoricals
    if is_train:
        label_encoders = {}
        cat_cols = df.select_dtypes(include='object').columns
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        return df, label_encoders
    else:
        for col, le in label_encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col].fillna('None'))
        return df
