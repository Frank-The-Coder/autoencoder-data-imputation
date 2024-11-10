import numpy as np
import torch
import random

def zero_out_feature(records, feature, cat_index, cat_values):
    start_index = cat_index[feature]
    stop_index = start_index + len(cat_values[feature])
    records[:, start_index:stop_index] = 0
    return records

def zero_out_random_feature(records, catcols, cat_index, cat_values):
    return zero_out_feature(records, random.choice(catcols), cat_index, cat_values)

def get_onehot(record, feature, cat_index, cat_values):
    """Extracts the one-hot encoded section of a record for a given feature."""
    start_index = cat_index[feature]
    stop_index = start_index + len(cat_values[feature])
    return record[start_index:stop_index]

def get_categorical_value(onehot, feature, cat_values):
    """Converts a one-hot encoded feature back to its categorical value."""
    max_index = np.argmax(onehot)
    return cat_values[feature][max_index]

def get_feature(record, feature, cat_index, cat_values):
    """Extracts the categorical feature value from a record."""
    onehot = get_onehot(record, feature, cat_index, cat_values)
    return get_categorical_value(onehot, feature, cat_values)

def get_accuracy(model, data_loader, catcols, cat_index, cat_values):
    """Calculates accuracy of the autoencoder model on categorical features."""
    total, correct = 0, 0
    for feature in catcols:
        for batch in data_loader:
            inputs = batch.numpy()
            outputs = model(zero_out_feature(batch.clone(), feature, cat_index, cat_values)).detach().numpy()
            for i in range(outputs.shape[0]):
                correct += int(get_feature(outputs[i], feature, cat_index, cat_values) == get_feature(inputs[i], feature, cat_index, cat_values))
                total += 1
    return correct / total
