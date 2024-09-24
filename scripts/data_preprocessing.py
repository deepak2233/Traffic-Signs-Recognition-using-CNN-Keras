import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import argparse

def run(args):
    data_dir = args.data_dir
    train_dir = os.path.join(data_dir, 'Train')
    test_dir = os.path.join(data_dir, 'Test')

    print("----------- Starting Data Preprocessing -----------")
    
    # Load dataset
    data = []
    labels = []
    classes = 43
    
    # Load training images
    for label in range(classes):
        label_dir = os.path.join(train_dir, str(label))
        for img_file in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_file)
            img = cv2.imread(img_path, -1)
            img = cv2.resize(img, (30, 30), interpolation=cv2.INTER_NEAREST)
            data.append(img)
            labels.append(label)

    data = np.array(data)
    labels = np.array(labels)
    print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
    
    # Splitting training and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # One hot encoding for labels
    y_train = to_categorical(y_train, classes)
    y_test = to_categorical(y_test, classes)

    # Save preprocessed data
    np.save(os.path.join(data_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(data_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(data_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(data_dir, 'y_test.npy'), y_test)

    print("----------- Data Preprocessing Complete -----------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preprocessing for Traffic Sign Recognition")
    parser.add_argument('--data_dir', type=str, default='data', help='Directory of the data')
    args = parser.parse_args()
    run(args)
