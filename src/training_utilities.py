import os
import pandas as pd
import matplotlib.pyplot as plt

def import_history(exp_dir, exp_name, checkpoint = False):
    csv_file_path = os.path.join(exp_dir,exp_name,'training_data.csv')
    
    if os.path.exists(csv_file_path) and checkpoint:
        df = pd.read_csv(csv_file_path)
        start_epoch = df['Epoch'].max() +1
        training_losses = df['Training Loss'].tolist()
        validation_losses = df['Validation Loss'].tolist()
        validation_accuracies = df['Validation Accuracy'].tolist()
        
        best_acc = df['Validation Accuracy'].max()
        best_epoch = df.loc[df['Validation Accuracy'].idxmax(), 'Epoch']
    else:
        start_epoch = 1
        training_losses = []
        validation_losses = []
        validation_accuracies = []
        best_acc = 0.0
        best_epoch = 0
    
    return start_epoch, training_losses, validation_losses, validation_accuracies, best_acc, best_epoch


def save_history(epoch, training_losses, validation_losses, validation_accuracies, exp_dir, exp_name, plots= False):
    directory_path = os.path.join(exp_dir,exp_name)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    csv_file_path = os.path.join(directory_path, 'training_data.csv')
    data = {
        'Epoch': range(1, epoch + 1),
        'Training Loss': training_losses,
        'Validation Loss': validation_losses,
        'Validation Accuracy': validation_accuracies,
    }
    
    df = pd.DataFrame(data)
    df.to_csv(csv_file_path, index=False)
    if plots : 
        # Visualisation des pertes
        plt.figure()
        plt.plot(range(1, len(training_losses) + 1), training_losses, label='Training Loss')
        plt.plot(range(1, len(validation_losses) + 1), validation_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        #Save the plots
        plt.savefig(os.path.join(directory_path,'Loss.png'))

        plt.figure()
        plt.plot(range(1, len(validation_accuracies) + 1), validation_accuracies)
        plt.title('Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.savefig(os.path.join(directory_path,'Accuracy.png'))