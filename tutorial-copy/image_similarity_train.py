# Simplified Training Script
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random

from image_similarity_dataset import FolderDataset
from image_similarity_encoder_model import ConvEncoder
from image_similarity_decoder_model import ConvDecoder
from image_similarity_engine import train_step, val_step
from image_similarity_embedding import create_embedding

from utils import horizontal_boundary_tensor_image, vertical_boundary_tensor_image

transforms = T.Compose([T.ToTensor()]) # Normalize the pixels and convert to tensor.

full_dataset = FolderDataset("../face-images/", transforms) # Create folder dataset.
print(type(full_dataset))

image_to_show = 0

print(f"\n\n{len(full_dataset)} images in the dataset")
print(f"full_dataset is of type {type(full_dataset)}\n\n")

print(f"Each element of the dataset is a {type(full_dataset[image_to_show])}")
print(f"...with a length of {len(full_dataset[image_to_show])}\n\n")

print(f"Each element of the tuple is a {type(full_dataset[image_to_show][0])}\n\n")
print(f"The tensor values: [channels, vertical_pixels, horizontal_pixels]: {full_dataset[image_to_show][0].shape}\n\n")
tensor = full_dataset[image_to_show][0]
print(tensor)
print(f"channels: {tensor.shape[0]}")
print(f"vertical_pixels: {tensor.shape[1]}")
print(f"horizontal_pixels: {tensor.shape[2]}")

print(f"Each element is itself a {type(tensor[0][0][0])} and inside is a {type(tensor[0][0][0].item())}")

""" h_boundary_image_ref = [[   [1, 1, 1], 
                            [1, 1, 1],
                            [0, 0, 0]   ]]

v_boundary_image_ref = [[   [1, 1, 0],
                            [1, 1, 0],
                            [1, 1, 0]   ]]

vertical_boundary_tensor = vertical_boundary_tensor_image()
horizontal_boundary_tensor = horizontal_boundary_tensor_image()

class DummyDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.all_imgs = []
        for i in range(50):
            self.all_imgs.append(vertical_boundary_tensor_image())
            self.all_imgs.append(horizontal_boundary_tensor_image())

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        tensor_image = self.all_imgs[idx]
        return tensor_image, tensor_image

print(vertical_boundary_tensor)
print(vertical_boundary_tensor.shape)

print(horizontal_boundary_tensor)
print(horizontal_boundary_tensor.shape) """

# make sure random order
# exit()

# full_dataset = DummyDataset()

train_size = 0.75
val_size = 1 - train_size

# Split data to train and test
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size]) 

# Create the train dataloader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
 
# Create the validation dataloader
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

# Create the full dataloader
full_loader = torch.utils.data.DataLoader(full_dataset, batch_size=32)

loss_fn = nn.MSELoss() # We use Mean squared loss which computes difference between two images.

encoder = ConvEncoder() # Our encoder model
decoder = ConvDecoder() # Our decoder model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Shift models to GPU
encoder.to(device)
decoder.to(device)

# Both the enocder and decoder parameters
autoencoder_params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(autoencoder_params, lr=1e-3) # Adam Optimizer

# Time to Train !!!
EPOCHS = 1
max_loss = float('inf')
# Usual Training Loop
for epoch in tqdm(range(EPOCHS)):
        train_loss = train_step(encoder, decoder, train_loader, loss_fn, optimizer, device=device)
        
        print(f"Epochs = {epoch}, Training Loss : {train_loss}")
        
        val_loss = val_step(encoder, decoder, val_loader, loss_fn, device=device)
        
        print(f"Epochs = {epoch}, Validation Loss : {val_loss}")

        # Simple Best Model saving
        if val_loss < max_loss:
            print(f"Validation Loss decreased, saving new best model at epoch {epoch}")
            max_loss = val_loss
            torch.save(encoder.state_dict(), "encoder_model.pt")
            torch.save(decoder.state_dict(), "decoder_model.pt")

# Understand encoder model data structure            
encoder_dict = encoder.state_dict()
for key in encoder_dict:
    print(key)
    print(encoder_dict[key].shape)

# Save the feature representations.
EMBEDDING_SHAPE = (1, 256, 8, 8) # This we know from our encoder - I changed this to 8 from 16 - why did that work!?

# We need feature representations for complete dataset not just train and validation.
# Hence we use full loader here.
embedding = create_embedding(encoder, full_loader, EMBEDDING_SHAPE, device)

# Convert embedding to numpy and save them
numpy_embedding = embedding.cpu().detach().numpy()
num_images = numpy_embedding.shape[0]

# Save the embeddings for complete dataset, not just train
flattened_embedding = numpy_embedding.reshape((num_images, -1))
np.save("data_embedding.npy", flattened_embedding)