import torch
from torchvision import datasets, transforms 
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
import numpy as np


################################################
# Training tools
################################################

def train(dataloader, model, optimizer, device):
    """Define a train loop over the dataloader"""
    model.train()
    losses = []
    reconstruction_losses = []
    VQ_losses = []
    commitment_losses = []

    for images, labels in tqdm(dataloader, desc="batch", leave=False, total=len(dataloader)):
        # Reset gradient
        optimizer.zero_grad()

        # Move image to device and proceed to forward pass, and the loss
        images = images.to(device)
        pred_images, ze, zq, _= model(images)
        loss, reconstruction_loss, VQ_loss, commitment_loss = model.loss(images, pred_images, ze, zq)

        #Proceed the backpropagation
        loss.backward()
        optimizer.step()

        losses.append(loss.detach().cpu().numpy())
        reconstruction_losses.append(reconstruction_loss.detach().cpu().numpy())
        VQ_losses.append(VQ_loss.detach().cpu().numpy())
        commitment_losses.append(commitment_loss.detach().cpu().numpy())

        del images
        torch.cuda.empty_cache()

    return sum(losses)/len(losses), sum(reconstruction_losses)/len(reconstruction_losses), sum(VQ_losses)/len(VQ_losses), sum(commitment_losses)/len(commitment_losses)

def eval(dataloader, model, optimizer, device):
    """Define a val loop over the dataloader"""
    model.eval()
    losses = []
    reconstruction_losses = []
    VQ_losses = []
    commitment_losses = []

    for images, _ in dataloader:
        with torch.no_grad():

            images = images.to(device)
            pred_images, ze, zq, _ = model(images)
            loss, reconstruction_loss, VQ_loss, commitment_loss = model.loss(images, pred_images, ze, zq)

            losses.append(loss.detach().cpu().numpy())
            reconstruction_losses.append(reconstruction_loss.detach().cpu().numpy())
            VQ_losses.append(VQ_loss.detach().cpu().numpy())
            commitment_losses.append(commitment_loss.detach().cpu().numpy())

        del images
        torch.cuda.empty_cache()

    return sum(losses)/len(losses), sum(reconstruction_losses)/len(reconstruction_losses), sum(VQ_losses)/len(VQ_losses), sum(commitment_losses)/len(commitment_losses)

################################################
# Model checkpoints management
################################################

def save_checkpoint(model, optimizer, epoch, losses, path):
    """ Save the model, optimzer at epoch """
    print(f'Checkpoint saved at epoch {epoch}')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
    }, path)

def load_checkpoint(path, model, device):
    """ Reaload from checkpoint """ 
    if device == 'cpu':
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
        
    epoch = checkpoint['epoch']
    loss = checkpoint['losses']
    
    return model, epoch, loss

################################################
# Visualize losses
################################################

def load_losses(file_path='./checkpoints/losses_dicts'):
    """Load the losses from the pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def create_step_indices(val_data):
    """Create x-axis indices for training and validation steps."""
    train_steps = np.arange(0,len(val_data),2).astype(int)
    val_steps = np.arange(0, len(val_data), 5)  # Validation every 5 steps
    return train_steps, val_steps

def plot_losses(loaded_losses):
    train_steps, val_steps = create_step_indices(list(loaded_losses[0].items())[0][1][0])
    type_num_embdeding = {(25):"o",
                      (50):"s",
                      (100):"x"}

    type_embedding_dim = {(16):"red",
                        (32):"blue",
                        (64):"green"}
    type_loss = {(0): 'Global Loss',
                (1) : 'Reconstruction Loss',
                (2) : 'VQ Loss',
                (3) : 'Commitment Loss'}
    figure = plt.figure(figsize=(20,15))
    for i,losses in enumerate(loaded_losses):
        ax = figure.add_subplot(2,2,i+1)
        for key in losses.keys():
            ax.plot(train_steps, np.array(losses[key][0])[train_steps], marker=type_num_embdeding[key[0]], color=type_embedding_dim[key[1]], linestyle='solid')
            ax.plot(val_steps, losses[key][1], marker=type_num_embdeding[key[0]], color=type_embedding_dim[key[1]], linestyle='dashed')
        ax.set_yscale('log')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss value')
        ax.grid()
        ax.set_title(type_loss[i])



    legend = []
    for num_embedding in type_num_embdeding.keys():
        legend.append(plt.Line2D([0], [0], marker=type_num_embdeding[num_embedding],color='black', 
                                    label=f'num_embedding={num_embedding}', linestyle=''))
        
    for embedding_dim in type_embedding_dim.keys():
        legend.append(plt.Line2D([0], [0], color=type_embedding_dim[embedding_dim], 
                                    label=f'embedding_dim={embedding_dim}', linestyle='solid'))

    legend.append(plt.Line2D([0], [0], color='black', linestyle='solid', label='Train'))
    legend.append(plt.Line2D([0], [0], color='black', linestyle='dashed', label='Test'))
    figure.legend(handles=legend, title='Training parameters', loc='center right', bbox_to_anchor=(1.05, 0.5))
    plt.plot()


def find_best_model_params(loaded_losses):
    """Gather final training value (we use final value because that when we saved the checkpoint for the model)"""
    best_final_train_loss = {}
    best_final_test_loss = {}

    for key in loaded_losses[0].keys():
        best_final_train_loss[key] = loaded_losses[0][key][0][-1]
        best_final_test_loss[key] = loaded_losses[0][key][1][-1]
    
    # Sort the dictionnary according to the second dim
    best_final_train_loss = {k: v for k, v in sorted(best_final_train_loss.items(), key=lambda item: item[1])}
    best_final_test_loss = {k: v for k, v in sorted(best_final_train_loss.items(), key=lambda item: item[1])}

    best_train_params = list(best_final_train_loss.keys())[0]
    best_test_params = list(best_final_test_loss.keys())[0]
    print('Best parameters found in training, for the training set were:\n',
          f'        num_embedding={best_train_params[0]}\n',
          f'        embedding_dim={best_train_params[1]}\n'
          f'        with loss={best_final_train_loss[best_train_params]:.4e}')
    
    print('Best parameters found in training, for the test set were:\n',
          f'        num_embedding={best_test_params[0]}\n',
          f'        embedding_dim={best_test_params[1]}\n'
          f'        with loss={best_final_test_loss[best_test_params]:.4e}')

    # If they are the same
    if best_train_params == best_test_params:
        return best_train_params
    
    # Ask user to choose the one if wants if different
    else:
        choice =''
        while choice not in  ['train', 'test']:
            choice = input('Which parameters do you choose ? (train or test)')
        if choice == 'train':
            return best_train_params
        elif choice == 'test':
            return best_test_params