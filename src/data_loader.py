import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data():
    header = ['age', 'work', 'fnlwgt', 'edu', 'yredu', 'marriage', 'occupation', 
              'relationship', 'race', 'sex', 'capgain', 'caploss', 'workhr', 'country']
    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        names=header, index_col=False
    )

    # Normalize continuous features
    continuous_features = ["age", "yredu", "capgain", "caploss", "workhr"]
    for feature in continuous_features:
        min_val = df[feature].min()
        max_val = df[feature].max()
        df[feature] = (df[feature] - min_val) / (max_val - min_val)
    
    # Selecting relevant columns and removing records with missing values
    contcols = ["age", "yredu", "capgain", "caploss", "workhr"]
    catcols = ["work", "marriage", "occupation", "edu", "relationship", "sex"]
    features = contcols + catcols
    df = df[features]
    
    # Filter rows with missing categorical values
    missing = pd.concat([df[c] == " ?" for c in catcols], axis=1).any(axis=1)
    df_not_missing = df[~missing]
    
    # One-hot encode the categorical data
    data = pd.get_dummies(df_not_missing)
    
    # Create mappings for categorical features
    cat_index = {}
    cat_values = {}
    for i, header in enumerate(data.keys()):
        if "_" in header:
            feature, value = header.split("_")
            if feature not in cat_index:
                cat_index[feature] = i
                cat_values[feature] = [value]
            else:
                cat_values[feature].append(value)
    
    # Convert to numpy array for model input
    datanp = data.values.astype(np.float32)
    np.random.seed(50)
    
    # Splitting data into train, validation, and test sets
    total_size = datanp.shape[0]
    train_size = int(0.7 * total_size)
    valid_size = int(0.15 * total_size)
    shuffled_indices = np.random.permutation(total_size)
    datanp = datanp[shuffled_indices]
    
    return datanp[:train_size], datanp[train_size:train_size + valid_size], datanp[train_size + valid_size:], catcols, cat_index, cat_values
