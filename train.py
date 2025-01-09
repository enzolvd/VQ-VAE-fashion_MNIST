###############################################################
# Libraries
###############################################################
import torch
from torchvision import datasets, transforms 
from tqdm import tqdm
from model import VQ_VAE
from utils import train, eval, save_checkpoint
import pickle

###############################################################
# Preparation
###############################################################

# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Add checkpoint path
check_path = f'./checkpoints/'

# Load the dataset
train_dataset = datasets.FashionMNIST(root='./data',download=True, train=True,transform=transforms.ToTensor())
test_dataset = datasets.FashionMNIST(root='./data',download=True, train=False,transform=transforms.ToTensor())

# Define the dataloaders
batch_size = 128
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size, num_workers=4)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size, num_workers=4)


###############################################################
# Training
###############################################################
# Training parameters
num_embeddings = [25,50,100]
embedding_dims = [16,32,64]
num_epochs = 50

# Saving objects
dict_losses = {}
dict_reconstructions_losses = {}
dict_VQ_losses = {}
dict_commitment_losses = {}
dict_models = {}

# Training loop
for num_embedding in num_embeddings:
    for embedding_dim in embedding_dims:
                
        beta = 0.25
        model = VQ_VAE(num_embedding,
                       embedding_dim).to(device)
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters())

        # Define run related object
        losses_train = []
        reconstruction_losses_train = []
        VQ_losses_train = []
        commitment_losses_train = []

        losses_val = []
        reconstruction_losses_val = []
        VQ_losses_val = []
        commitment_losses_val = []
        progress_bar = tqdm(range(num_epochs), desc='Epochs')
        for epoch in progress_bar:
            # Train
            loss_train, reconstruction_loss_train, VQ_loss_train, commitment_loss_train = train(train_dataloader, model, optimizer, device)
            losses_train.append(loss_train)
            reconstruction_losses_train.append(reconstruction_loss_train)
            VQ_losses_train.append(VQ_loss_train)
            commitment_losses_train.append(commitment_loss_train)
            


            if epoch%5==0: # Validation computation
                # Loss
                loss_val, reconstruction_loss_val, VQ_loss_val, commitment_loss_val = eval(test_dataloader, model, optimizer, device)
                losses_val.append(loss_val)
                reconstruction_losses_val.append(reconstruction_loss_val)
                VQ_losses_val.append(VQ_loss_val)
                commitment_losses_val.append(commitment_loss_val)
                progress_bar.set_description(f'Val: loss={loss_val}, recon={reconstruction_loss_val}, VQ={VQ_loss_val}, com={commitment_loss_val}')
                
            else:
                progress_bar.set_description(f'Train: loss={loss_train}, recon={reconstruction_loss_train}, VQ={VQ_loss_train}, com={commitment_loss_train}')
        
        # Save losses for later visualisation purposes
        dict_losses[(num_embedding,embedding_dim)] = (losses_train,losses_val)
        dict_reconstructions_losses[(num_embedding,embedding_dim)] = (reconstruction_losses_train,reconstruction_losses_val)
        dict_VQ_losses[(num_embedding,embedding_dim)] = (VQ_losses_train,VQ_losses_val)
        dict_commitment_losses[(num_embedding,embedding_dim)] = (commitment_losses_train,commitment_losses_val)
        # Save checkpoint
        save_path = check_path+f'checkpoints_{num_embedding}_{embedding_dim}.pt'
        save_checkpoint(model, optimizer, epoch, None, save_path)

# Save losses
losses = [dict_losses,
          dict_reconstructions_losses,
          dict_VQ_losses,
          dict_commitment_losses]

with open('./checkpoints/losses_dicts', 'wb') as f:
        pickle.dump(losses, f)           