import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

def preprocess(df: pd.DataFrame, is_train=True, ordinal_encoder=None, cat_cols=None):
    """Preprocess the dataset using semantic imputations, new features, and ordinal encoding."""

    # --- 1. Missing value treatment ---
    none_cols = [
        'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish',
        'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
        'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'Electrical'
    ]
    zero_cols = [
        'GarageYrBlt', 'MasVnrArea', 'BsmtFullBath', 'BsmtHalfBath',
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF'
    ]

    for col in none_cols:
        if col in df.columns:
            df[col] = df[col].fillna('None')
    for col in zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Impute LotFrontage by Neighborhood median
    if 'LotFrontage' in df.columns and 'Neighborhood' in df.columns:
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

    # Fill other remaining numeric NaNs with -1 to mark missingness
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(-1)

    # --- 2. Feature Engineering ---
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBathrooms'] = (
        df['FullBath'] + 0.5 * df['HalfBath'] +
        df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
    )
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    df['IsRemodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)

    # Binary flags
    df['HasPool'] = (df['PoolQC'] != 'None').astype(int)
    df['HasGarage'] = (df['GarageType'] != 'None').astype(int)
    df['HasBasement'] = (df['BsmtQual'] != 'None').astype(int)
    df['HasFireplace'] = (df['FireplaceQu'] != 'None').astype(int)
    df['HasFence'] = (df['Fence'] != 'None').astype(int)
    df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)

    # Log-transform skewed features
    for col in ['GrLivArea', 'LotArea', 'TotalSF']:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    # --- 3. Categorical Encoding ---
    if is_train:
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        df[cat_cols] = df[cat_cols].astype(str)

        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df[cat_cols] = encoder.fit_transform(df[cat_cols]).astype(np.float32)
        return df, encoder, cat_cols
    else:
        if cat_cols is None:
            cat_cols = ordinal_encoder.feature_names_in_.tolist()

        df[cat_cols] = df[cat_cols].astype(str)
        df[cat_cols] = ordinal_encoder.transform(df[cat_cols]).astype(np.float32)
        return df
