
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# Define the model (Variational Autoencoder for Model 2)
class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        self.capacity = 64
        self.latent_dims = 20
        # encoder
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.capacity, kernel_size=4, stride=2, padding=1)  # out: c x 14 x 14
        self.conv2 = nn.Conv2d(in_channels=self.capacity, out_channels=self.capacity * 2, kernel_size=4, stride=2, padding=1)  # out: c x 7 x 7
        self.fc_mu = nn.Linear(in_features=self.capacity * 2 * 7 * 7, out_features=self.latent_dims)
        self.fc_logvar = nn.Linear(in_features=self.capacity * 2 * 7 * 7, out_features=self.latent_dims)
        # decoder
        self.fc_decode = nn.Linear(in_features=self.latent_dims, out_features=self.capacity * 2 * 7 * 7)
        self.conv2_decode = nn.ConvTranspose2d(in_channels=self.capacity * 2, out_channels=self.capacity, kernel_size=4, stride=2, padding=1)
        self.conv1_decode = nn.ConvTranspose2d(in_channels=self.capacity, out_channels=1, kernel_size=4, stride=2, padding=1)

    def encoder(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # flatten batch of multi-channel feature maps to a batch of feature vectors
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar

    def decoder(self, x):
        x = self.fc_decode(x)
        x = x.view(x.size(0), self.capacity * 2, 7, 7)  # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = torch.relu(self.conv2_decode(x))
        x = torch.sigmoid(self.conv1_decode(x))  # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        return x

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.empty_like(std).normal_()
        return eps.mul(std).add_(mu)

# Custom MNIST Dataset class to load MNIST data from .pt files
class CustomMNISTDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.data = torch.load(data_path)  # Load the .pt data
        self.images = self.data[0]  # Assuming data[0] contains the images
        self.labels = self.data[1]  # Assuming data[1] contains the labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Manual transformation: normalize the images
def manual_transform(image):
    return (image - 0.5) / 0.5  # Normalize the image to have values between -1 and 1

# Function to load models' weights
def load_model_weights(model, weights_path):
    model.load_state_dict(torch.load(weights_path))
    model.eval()  # Set the model to evaluation mode
    return model

# Evaluate model
def evaluate_model(model, data_loader, criterion):
    correct = 0
    total = 0
    total_loss = 0

    for data, target in data_loader:
        output, _, _ = model(data)  # Get reconstruction output
        loss = criterion(output, data)  # Compute reconstruction loss
        total_loss += loss.item()

    average_loss = total_loss / len(data_loader)
    return average_loss

# Load datasets manually (custom)
holdout_data_path = '/path/to/your/holdout_dataset.pt'
test_data_path = '/path/to/your/test_dataset.pt'
holdout_dataset = CustomMNISTDataset(holdout_data_path, transform=manual_transform)
test_dataset = CustomMNISTDataset(test_data_path, transform=manual_transform)

# Create DataLoader objects for batching
batch_size = 64
holdout_loader = DataLoader(holdout_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load Model 2 (Variational Autoencoder)
model2_weights_path = '/path/to/your/amml_model2_weights.pth'
model2 = VariationalAutoencoder()
model2 = load_model_weights(model2, model2_weights_path)

# Define the reconstruction loss function (MSE for autoencoder)
criterion = nn.MSELoss()

# Evaluate Model 2
loss = evaluate_model(model2, test_loader, criterion)
print(f"Model 2 Reconstruction Loss: {loss}")
