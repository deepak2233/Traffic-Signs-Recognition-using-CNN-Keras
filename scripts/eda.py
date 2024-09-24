import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def run(args):
    data_dir = args.data_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("----------- Starting Exploratory Data Analysis (EDA) -----------")
    
    # Load the training data
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    
    # Count the occurrences of each class
    class_counts = np.sum(y_train, axis=0)
    class_labels = np.arange(len(class_counts))

    # Class distribution plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=class_labels, y=class_counts)
    plt.title("Class Distribution of Traffic Signs")
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.xticks(class_labels)
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
    print(f"Class distribution plot saved to {output_dir}/class_distribution.png")
    plt.show()

    # Load sample images for visualization
    sample_images = []
    sample_labels = []
    classes = 43
    
    print("Loading sample images...")
    for label in range(classes):
        label_dir = os.path.join(data_dir, 'Train', str(label))
        sample_img = os.listdir(label_dir)[0]  # Get the first image
        img_path = os.path.join(label_dir, sample_img)
        img = plt.imread(img_path)
        sample_images.append(img)
        sample_labels.append(label)
    
    # Plot sample images
    plt.figure(figsize=(15, 10))
    for i in range(1, 11):
        plt.subplot(2, 5, i)
        plt.imshow(sample_images[i - 1])
        plt.title(f"Class: {sample_labels[i - 1]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_images.png'))
    print(f"Sample images plot saved to {output_dir}/sample_images.png")
    plt.show()
    
    print("----------- EDA Complete -----------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exploratory Data Analysis for Traffic Sign Recognition")
    parser.add_argument('--data_dir', type=str, default='data', help='Directory of the data')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory for outputs')
    args = parser.parse_args()
    run(args)
