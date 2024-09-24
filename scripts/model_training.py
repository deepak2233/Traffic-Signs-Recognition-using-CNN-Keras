import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import argparse

def run(args):
    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir
    os.makedirs(model_dir, exist_ok=True)

    print("----------- Starting Model Training -----------")
    print("Loading preprocessed data...")
    
    # Load preprocessed data
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

    print("Building model...")
    model = Sequential()

    # First Layer
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))

    # Second Layer 
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))

    # Dense Layer
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(43, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Train the model
    print("Training the model...")
    history = model.fit(X_train, y_train, batch_size=args.batch_size, epochs=args.epochs, validation_data=(X_test, y_test))

    # Save the model and history
    model.save(os.path.join(model_dir, 'best_model.h5'))
    np.save(os.path.join(model_dir, 'history.npy'), history.history)

    print("----------- Model Training Complete -----------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training for Traffic Sign Recognition")
    parser.add_argument('--data_dir', type=str, default='data', help='Directory of the data')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory for outputs')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    args = parser.parse_args()
    run(args)
