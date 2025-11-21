"""
Dataset Module
Handles data loading, preprocessing, and splitting operations.
"""

# This module will contain:
# - load_data() function to load CSV data
# - split_data() function to split into train/val/test (60/20/20)
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_process_data(filepath, train_ratio, random_state):
    '''
    
    load_and_process_data() fonksiyonu, /data/hw1_data.csv dosyasını okur ve numpy array olarak döndürür.
    '''
    raw_data = np.loadtxt(filepath,delimiter=',', dtype=np.float32)
    
    X = raw_data[:, 0:2]
    y = raw_data[:, 2]
    
    
    # 1. KADEME: Veriyi %60 Train ve %40 (Val+Test) olarak ayır
    # test_size=0.4 -> Geriye kalan %60 Train olur.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=( 1- train_ratio), random_state=random_state
    )

    # 2. KADEME: Kalan %40'lık kısmı (X_temp), tam ortadan ikiye böl (%20 Val, %20 Test)
    # test_size=0.5 -> Temp verisinin yarısı Val, yarısı Test olur.
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state
    )
    
    
    
    train_mean = np.mean(X_train, axis=0)
    train_std = np.std(X_train, axis=0)
    
    #Standart sapmanın 0 olma ihtimaline karşı epsilon
    train_std = train_std + 1e-8
    
    

    X_train_scaled = (X_train - train_mean) / train_std
    
    X_val_scaled = (X_val - train_mean) / train_std
    
    X_test_scaled = (X_test - train_mean) / train_std

    return (X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test)

