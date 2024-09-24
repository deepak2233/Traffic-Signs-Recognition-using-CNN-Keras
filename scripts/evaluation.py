import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse

def run(args):
    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print("----------- Starting Model Evaluation -----------")
    print("Loading preprocessed data...")
    
    # Load preprocessed data
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

    print("Loading model...")
    model_path = os.path.join(model_dir, 'best_model.h5')
    model = tf.keras.models.load_model(model_path)

    print("Evaluating model on test data...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%")

    # Load training history
    history_path = os.path.join(model_dir, 'history.npy')
    if os.path.exists(history_path):
        print("Loading training history...")
        history = np.load(history_path, allow_pickle=True).item()

        # Plot accuracy and loss
        plt.figure(figsize=(12, 5))

        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Train Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.savefig(os.path.join(output_dir, 'evaluation_plots.png'))
        plt.show()

    print("----------- Model Evaluation Complete -----------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Evaluation for Traffic Sign Recognition")
    parser.add_argument('--data_dir', type=str, default='data', help='Directory of the data')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory for outputs')
    args = parser.parse_args()
    run(args)
