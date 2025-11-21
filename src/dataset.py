"""
Dataset Module
Handles data loading, preprocessing, and splitting operations.
"""

# This module will contain:
# - load_data() function to load CSV data
# - split_data() function to split into train/val/test (60/20/20)

def load_data():
    '''
    
    load_data() fonksiyonu, /data/hw1_data.csv dosyasını okur ve numpy array olarak döndürür.
    '''
    
    data = np.loadtxt('/data/hw1_data.csv', delimiter=',',dtype=np.float32)
    return data


def split_data(data, train_ratio, val_ratio, test_ratio):
    '''
    
    split_data() fonksiyonu, data arrayini train, val ve test setlerine böler. (%60 train, %20 validation, %20 test)
    '''
    
    train_data = data[:int(len(data) * train_ratio)]
    val_data = data[int(len(data) * train_ratio):int(len(data) * (train_ratio + val_ratio))]
    test_data = data[int(len(data) * (train_ratio + val_ratio)):]
    return train_data, val_data, test_data
