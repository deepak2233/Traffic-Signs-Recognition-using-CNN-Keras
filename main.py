import argparse
from scripts import data_preprocessing, model_training, evaluation, inference, streamlit_app, eda

def main(args):
    if args.data:
        print("----------- Starting Data Preprocessing -----------")
        data_preprocessing.run(args)
        
    if args.eda:
        print("----------- Starting Exploratory Data Analysis (EDA) -----------")
        eda.run(args)
        
    if args.training:
        print("----------- Starting Model Training -----------")
        model_training.run(args)
        
    if args.evaluation:
        print("----------- Starting Model Evaluation -----------")
        evaluation.run(args)
        
    if args.inference:
        print("----------- Starting Inference -----------")
        inference.run(args)
        
    if args.streamlit:
        print("----------- Starting Streamlit App -----------")
        streamlit_app.run(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Traffic Sign Recognition Pipeline")
    parser.add_argument('--data', action='store_true', help='Run data preprocessing step')
    parser.add_argument('--eda', action='store_true', help='Run EDA step')
    parser.add_argument('--training', action='store_true', help='Run model training step')
    parser.add_argument('--evaluation', action='store_true', help='Run model evaluation step')
    parser.add_argument('--inference', action='store_true', help='Run model inference step')
    parser.add_argument('--streamlit', action='store_true', help='Run Streamlit app')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory of the data')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory for outputs')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    args = parser.parse_args()
    
    main(args)
