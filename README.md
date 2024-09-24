# Traffic Sign Recognition

## Overview 
This project implements a Traffic Sign Recognition system using Convolutional Neural Networks (CNN) to classify images of traffic signs. The aim is to build an automated system that accurately recognizes and classifies various traffic signs from images, contributing to the development of advanced driver-assistance systems (ADAS) and autonomous vehicles.

### Problem Statement
Traffic signs are critical for ensuring road safety as they convey essential information to drivers. An automated recognition system can help improve safety and efficiency on the roads. For example, a system should recognize a "Stop" sign and alert the driver to stop the vehicle, thereby preventing accidents.

### Example Images
- **Stop Sign**:  
  ![Stop Sign](data/Test/00111.png)

- **Yield Sign**:  
  ![Yield Sign](data/Test/00120.png)

- **Speed Limit Sign**:  
  ![Speed Limit Sign](data/Test/00122.png)

## Datasets
The dataset used for this project is the **German Traffic Sign Recognition Benchmark (GTSRB)**. It contains over 50,000 images categorized into 43 classes of traffic signs.

### Dataset Structure
```
data/ 
    ├── Train/ # Contains train images organized by class
    ├── Test/ # Contains test images organized by class
    
```
### CSV Files
- **`Train.csv`**: Contains paths and labels for the training set.
- **`Test.csv`**: Contains paths and labels for the test set.

## Project Structure

```
    traffic-sign-recognition/
    │
    ├── data/
    │   ├── Meta/
    │   ├── Test/
    │   ├── Train/
    │   └── ...
    ├── scripts/
    │   ├── data_preprocessing.py
    │   ├── eda.py
    │   ├── model_training.py
    │   ├── evaluation.py
    │   ├── inference.py
    │   └── streamlit_app.py
    ├── main.py
    ├── requirements.txt
    └── README.md
```

## Setup and Installation
1. **Clone the Repository**:
   ```bash
       git clone <repository-url>
       cd traffic-sign-recognition
        python3 -m venv env
        source env/bin/activate  # On Windows use `env\Scripts\activate`
        pip install -r requirements.txt

# Usage

### Data Preprocessing

To preprocess the dataset:
```
    python main.py --data --data_dir data
```

### Exploratory Data Analysis (EDA)
To generate visualizations and understand the dataset:
```
python main.py --eda --data_dir data --output_dir outputs

```

### Model Training
To train the CNN model on the preprocessed data:

```
python main.py --training --data_dir data --model_dir models --epochs 20 --batch_size 64 --learning_rate 0.001
```

### Model Evaluation
To evaluate the trained model:
```
python main.py --evaluation --data_dir data --model_dir models --output_dir outputs
```

### Inference
To make predictions on new images:

```
python main.py --inference --model_dir models --image_path path/to/image.jpg

```

### Streamlit Application
To launch the Streamlit application for interactive traffic sign recognition:

```
streamlit run scripts/streamlit_app.py -- --model_path models/best_model.h5
```


### Notes
- **Image Paths**: Ensure to replace `data/Train/00000.png`, etc., with actual paths to your images.
- **Repository URL**: Replace `<repository-url>` with the actual URL of your GitHub repository.
- **License**: Ensure you have a LICENSE file if you include a license section.

This `README.md` provides a comprehensive overview of your project, making it easy for users to understand its purpose and how to use it effectively. Let me know if you need further adjustments or additions!

