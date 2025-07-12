import os
import sys
import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.image as mpimg
import matplotlib.cm as cm
import base64

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_hidden = 32
        self.encoder = nn.Sequential(
            nn.Linear(784, 256), 
            nn.ReLU(),  
            nn.Linear(256, self.num_hidden),  
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.num_hidden, 256),  
            nn.ReLU(),  
            nn.Linear(256, 784),  
            nn.Sigmoid(),  
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class CNNAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),  # Output: [16, H/2, W/2]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # Output: [32, H/4, W/4]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: [64, H/8, W/8]
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: [32, H/4, W/4]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # Output: [16, H/2, W/2]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),   # Output: [3, H, W]
            nn.Sigmoid(),
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class CNNVAE(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
        )
        self._initialize_shape()
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  
        )
    def _initialize_shape(self):
        dummy_input = torch.zeros(1, 3, 512, 768)  
        with torch.no_grad():
            encoded = self.encoder(dummy_input)
        self.flatten_shape = encoded.shape[1:] 
        self.flatten_size = encoded.numel()
    def encode(self, x):
        encoded = self.encoder(x)
        encoded_flat = encoded.view(x.size(0), -1)
        mu = self.fc_mu(encoded_flat)
        logvar = self.fc_logvar(encoded_flat)
        return mu, logvar
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def decode(self, z):
        decoded_input = self.decoder_input(z)
        decoded_input = decoded_input.view(z.size(0), *self.flatten_shape)
        decoded = self.decoder(decoded_input)
        return decoded
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z)
        return decoded, mu, logvar
    
class CNNVAE_Depth(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
        )
        self._initialize_shape()
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  
        )
    def _initialize_shape(self):
        dummy_input = torch.zeros(1, 1, 512, 768)  
        with torch.no_grad():
            encoded = self.encoder(dummy_input)
        self.flatten_shape = encoded.shape[1:] 
        self.flatten_size = encoded.numel()
    def encode(self, x):
        encoded = self.encoder(x)
        encoded_flat = encoded.view(x.size(0), -1)
        mu = self.fc_mu(encoded_flat)
        logvar = self.fc_logvar(encoded_flat)
        return mu, logvar
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def decode(self, z):
        decoded_input = self.decoder_input(z)
        decoded_input = decoded_input.view(z.size(0), *self.flatten_shape)
        decoded = self.decoder(decoded_input)
        return decoded
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z)
        return decoded, mu, logvar

class VAE(AutoEncoder):
    def __init__(self):
        super().__init__()
        self.mu = nn.Linear(self.num_hidden, self.num_hidden)
        self.log_var = nn.Linear(self.num_hidden, self.num_hidden)
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder(z)
        return encoded, decoded, mu, log_var
    def sample(self, num_samples):
        with torch.no_grad():
            z = torch.randn(num_samples, self.num_hidden).to(device)
            samples = self.decoder(z)
        return samples

def train_mnist_autoencoder(X_train, learning_rate=0.001, batch_size=64, epochs=20):
    X_train = torch.from_numpy(X_train).float()
    model = AutoEncoder()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    model.to(device)
    train_loader = torch.utils.data.DataLoader(
        X_train, batch_size=batch_size, shuffle=True
    )
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            encoded, decoded = model(data)
            loss = criterion(decoded, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)
        epoch_loss = total_loss / len(train_loader.dataset)
        print(
            "Epoch {}/{}: loss={:.4f}".format(epoch + 1, epochs, epoch_loss)
        )
    eval_data = next(iter(train_loader))[:8] 
    eval_data = eval_data.to(device)
    model.eval()
    with torch.no_grad():
        encoded, decoded = model(eval_data)
    return eval_data, decoded

def train_carla_autoencoder(X_train, learning_rate=0.001, batch_size=8, epochs=20):
    model = CNNAutoEncoder()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    model.to(device)
    train_loader = torch.utils.data.DataLoader(
        X_train, batch_size=batch_size, shuffle=True
    )
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            encoded, decoded = model(data)
            loss = criterion(decoded, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)
        epoch_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}: loss={epoch_loss:.4f}")
    eval_data = next(iter(train_loader))[:5].to(device)
    model.eval()
    with torch.no_grad():
        _, decoded = model(eval_data)
    return eval_data, decoded

def train_mnist_vae(X_train, learning_rate=1e-3, batch_size=32,  num_epochs=10):
    X_train = torch.from_numpy(X_train).float().to(device)
    model = VAE()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss(reduction="sum")
    model.to(device)
    train_loader = torch.utils.data.DataLoader(
        X_train, batch_size=batch_size, shuffle=True
    )
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            encoded, decoded, mu, log_var = model(data)
            KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = criterion(decoded, data) + 3 * KLD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)
        epoch_loss = total_loss / len(train_loader.dataset)
        print(
            "Epoch {}/{}: loss={:.4f}".format(epoch + 1, num_epochs, epoch_loss)
        )
    eval_data = next(iter(train_loader))[:8] 
    eval_data = eval_data.to(device)
    model.eval()
    with torch.no_grad():
        encoded, decoded, mu, log_var = model(eval_data)
    return eval_data, decoded

def train_carla_vae(X_train, learning_rate=1e-3, batch_size=4, num_epochs=10):
    model = CNNVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss(reduction="sum")
    train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True)
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            decoded, mu, logvar = model(data)
            recon_loss = criterion(decoded, data)
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 3 * KLD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}: loss={epoch_loss:.4f}")
    return model

def train_carla_vae_depth(X_train, learning_rate=1e-3, batch_size=4, num_epochs=10):
    model = CNNVAE_Depth().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss(reduction="sum")
    train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True)
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            decoded, mu, logvar = model(data)
            recon_loss = criterion(decoded, data)
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 3 * KLD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}: loss={epoch_loss:.4f}")
    return model
    
def loss_function_mnist_VAE(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def get_mnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(),  
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    X_train = train_dataset.data.numpy().reshape(-1, 28 * 28) / 255.0
    y_train = train_dataset.targets.numpy()
    X_test = test_dataset.data.numpy().reshape(-1, 28 * 28) / 255.0
    y_test = test_dataset.targets.numpy()
    return X_train, y_train, X_test, y_test

def get_carla_data(train_path):
    transform = transforms.Compose([
        transforms.Resize((512, 768)),
        transforms.ToTensor(), 
    ])
    train_images = []
    for dirpath_root, _, _ in os.walk(train_path):
        if os.path.basename(dirpath_root).startswith("Scenario"):
            for dirpath, _, filenames in os.walk(dirpath_root):
                if os.path.basename(dirpath).startswith("RGB-CAM"):
                    print(f"Loading images from: {dirpath}")
                    for filename in sorted(filenames):
                        if filename.endswith('.png') or filename.endswith('.jpg'):
                            image_path = os.path.join(dirpath, filename)
                            image = Image.open(image_path).convert('RGB')
                            image = transform(image)
                            train_images.append(image)
    X_train = torch.stack(train_images)
    y_train = torch.zeros(len(X_train))
    
    return X_train, y_train

def load_carla_test_data(test_path):
    transform = transforms.Compose([
        transforms.Resize((512, 768)),
        transforms.ToTensor(), 
    ])
    test_images = []
    for filename in sorted(os.listdir(test_path)):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image_path = os.path.join(test_path, filename)
            image = Image.open(image_path).convert('RGB')
            image = transform(image)
            test_images.append(image)
    X_test = torch.stack(test_images)
    y_test = torch.zeros(len(X_test))
    return X_test, y_test
    

def show_input_mnist_images(images, labels):
    pixels = images.reshape(-1, 28, 28)
    fig, axs = plt.subplots(
        ncols=len(images), nrows=1, figsize=(10, 3 * len(images))
    )
    for i in range(len(images)):
        axs[i].imshow(pixels[i], cmap="gray")
        axs[i].set_title("Label: {}".format(labels[i]))
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_xlabel("Index: {}".format(i))
    fig.subplots_adjust(hspace=0.5)
    plt.show()
    
def show_input_carla_images(images, labels):
    fig, axs = plt.subplots(
        ncols=len(images), nrows=1, figsize=(15, 5)
    )
    for i in range(len(images)):
        axs[i].imshow(images[i].permute(1, 2, 0)) 
        axs[i].set_title("Label: {}".format(labels[i]))
        axs[i].axis("off")
    plt.show()

def show_reconstructed_mnist_images(original_images, reconstructed_images):
    fig, axs = plt.subplots(2, len(original_images), figsize=(15, 4))
    for i in range(len(original_images)):
        axs[0, i].imshow(original_images[i].reshape(28, 28), cmap="gray")
        axs[0, i].set_title("Original")
        axs[0, i].axis("off")
        axs[1, i].imshow(reconstructed_images[i].reshape(28, 28), cmap="gray")
        axs[1, i].set_title("Reconstructed")
        axs[1, i].axis("off")
    plt.show()
    
def show_reconstructed_carla_images(original_images, reconstructed_images):
    index = {'i': 0}
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(bottom=0.2)
    def plot_image():
        axs[0].clear()
        axs[1].clear()
        axs[2].clear()
        original = original_images[index['i']]
        reconstructed = reconstructed_images[index['i']]
        axs[0].imshow(original.transpose(1, 2, 0))
        axs[0].set_title(f"Original [{index['i']+1}/{len(original_images)}]")
        axs[0].axis("off")
        axs[1].imshow(reconstructed.transpose(1, 2, 0))
        axs[1].set_title("Reconstructed")
        axs[1].axis("off")
        axs[2].set_title("Error Heatmap")
        axs[2].axis("off")
        plt.draw()
    def on_key(event):
        if event.key == 'right':
            index['i'] = (index['i'] + 1) % len(original_images)
        elif event.key == 'left':
            index['i'] = (index['i'] - 1) % len(original_images)
        plot_image()
    def on_close(event):
        plt.close('all')
        sys.exit()
    plot_image()
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('close_event', on_close)
    plt.show()
    
def load_carla_depth_images(depth_cam_path):
    depth_values_list = []
    depth_files = sorted(os.listdir(depth_cam_path))
    print("Loading depth images...")
    for idx, file_name in enumerate(depth_files):
        if not (file_name.endswith('.png') or file_name.endswith('.jpg')):
            continue
        file_path = os.path.join(depth_cam_path, file_name)
        org_img = mpimg.imread(file_path)  
        if org_img.dtype == np.uint8:
            org_img = org_img.astype(np.float32) / 255.0
        img = org_img * [65536.0, 256.0, 1.0]
        img = np.sum(img, axis=2)
        img /= 16777215.0
        normalized_depth = img
        logdepth = np.ones(normalized_depth.shape) + (np.log(normalized_depth + 1e-8) / 5.70378)
        logdepth = np.clip(logdepth, 0.0, 1.0)
        logdepth = 1.0 - logdepth  
        depth_values_list.append(logdepth)
        if (idx + 1) % 10 == 0 or (idx + 1) == len(depth_files):
            print(f"Loaded {idx + 1}/{len(depth_files)} images")
    return depth_values_list

def load_carla_depth_images_cv2(depth_cam_path, max_distance=20.0):
    depth_values_list = []
    depth_files = sorted(os.listdir(depth_cam_path))
    print("Loading depth images...")
    for idx, file_name in enumerate(depth_files):
        if not (file_name.endswith('.png') or file_name.endswith('.jpg')):
            continue
        file_path = os.path.join(depth_cam_path, file_name)
        org_img = cv2.imread(file_path).astype(np.float32)
        depth = (org_img[:, :, 2] + org_img[:, :, 1] * 256.0 + org_img[:, :, 0] * 256.0 * 256.0) / (256 ** 3 - 1)
        depth_in_meters = depth * 1000.0
        depth_in_meters = np.clip(depth_in_meters, 0, max_distance)
        depth_visual = (depth_in_meters / max_distance) * 255.0
        depth_visual = depth_visual.astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_visual, cv2.COLORMAP_MAGMA)
        depth_colored_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
        frame_id = int(file_name.split('_')[-1].split('.')[0])
        depth_values_list.append((frame_id, depth_colored_rgb))
        if (idx + 1) % 10 == 0 or (idx + 1) == len(depth_files):
            print(f"Loaded {idx + 1}/{len(depth_files)} images")
    depth_sorted = sorted(depth_values_list, key=lambda x: x[0])
    return depth_sorted

def view_sample_depth_images(depth_frames):
    num_samples = min(5, len(depth_frames))
    fig, axs = plt.subplots(1, num_samples, figsize=(20, 5))
    for i in range(num_samples):
        axs[i].imshow(depth_frames[i][1])  
        axs[i].set_title(f'Depth Image {i+1}')
        axs[i].axis('off')
    plt.show()
    
def show_pure_anomaly_visualization(original_images, reconstructed_images, error_maps, depth_maps, valid_anomalies):
    index = {'i': 0}
    fig, axs = plt.subplots(1, 5, figsize=(25, 5))
    plt.subplots_adjust(bottom=0.2)
    def plot_image():
        axs[0].clear()
        axs[1].clear()
        axs[2].clear()
        axs[3].clear()
        axs[4].clear()
        original = original_images[index['i']]
        reconstructed = reconstructed_images[index['i']]
        error_map = error_maps[index['i']]
        depth_map = depth_maps[index['i']]
        valid_mask = valid_anomalies[index['i']]
        axs[0].imshow(original.transpose(1, 2, 0))
        axs[0].set_title(f"Original [{index['i']+1}/{len(original_images)}]")
        axs[0].axis("off")
        axs[1].imshow(reconstructed.transpose(1, 2, 0))
        axs[1].set_title("Reconstructed")
        axs[1].axis("off")
        axs[2].imshow(error_map, cmap='hot')
        axs[2].set_title("Error Heatmap")
        axs[2].axis("off")
        axs[3].imshow(depth_map, cmap='gray')
        axs[3].set_title("Depth Map")
        axs[3].axis("off")
        axs[4].imshow(valid_mask, cmap='gray')
        axs[4].set_title("Valid 3D Anomalies")
        axs[4].axis("off")
        plt.draw()
    def on_key(event):
        if event.key == 'right':
            index['i'] = (index['i'] + 1) % len(original_images)
        elif event.key == 'left':
            index['i'] = (index['i'] - 1) % len(original_images)
        plot_image()
    def on_close(event):
        plt.close('all')
        sys.exit()
    plot_image()
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('close_event', on_close)
    plt.show()
    
def preprocess(X_train):
    N, C, H, W = X_train.shape
    middle = int(H * 0.525)
    X_train[:, :, :middle, :] = 0
    X_train[:, :, middle, :] = 0
    return X_train

def prepare_depth_for_testing(depth_cam_path, max_distance=10.0):
    depth_images = []
    depth_files = sorted(os.listdir(depth_cam_path))
    print("Preparing depth images for testing...")
    for idx, file_name in enumerate(depth_files):
        if not (file_name.endswith('.png') or file_name.endswith('.jpg')):
            continue
        file_path = os.path.join(depth_cam_path, file_name)
        org_img = cv2.imread(file_path).astype(np.float32)
        depth = (org_img[:, :, 2] + org_img[:, :, 1] * 256.0 + org_img[:, :, 0] * 256.0 * 256.0) / (256 ** 3 - 1)
        depth_in_meters = depth * 1000.0
        depth_in_meters = np.clip(depth_in_meters, 0, max_distance)
        depth_normalized = (depth_in_meters / max_distance).astype(np.float32)
        depth_tensor = torch.from_numpy(depth_normalized).unsqueeze(0) 
        depth_images.append(depth_tensor)
        if (idx + 1) % 10 == 0 or (idx + 1) == len(depth_files):
            print(f"Loaded {idx + 1}/{len(depth_files)} images")
    depth_dataset = torch.stack(depth_images)
    return depth_dataset

def prepare_depth_for_training(depth_cam_path, max_distance=10.0):
    depth_images = []
    print("Searching for depth images...")
    for dirpath_root, _, _ in os.walk(depth_cam_path):
        if os.path.basename(dirpath_root).startswith("Scenario"):
            for dirpath, _, filenames in os.walk(dirpath_root):
                if os.path.basename(dirpath).startswith("DEPTH_CAM"):
                    print(f"Loading images from: {dirpath}")
                    for filename in sorted(filenames):
                        if filename.endswith('.png') or filename.endswith('.jpg'):
                            file_path = os.path.join(dirpath, filename)
                            org_img = cv2.imread(file_path).astype(np.float32)
                            depth = (org_img[:, :, 2] + org_img[:, :, 1] * 256.0 + org_img[:, :, 0] * 256.0 * 256.0) / (256 ** 3 - 1)
                            depth_in_meters = depth * 1000.0
                            depth_in_meters = np.clip(depth_in_meters, 0, max_distance)
                            depth_normalized = (depth_in_meters / max_distance).astype(np.float32)
                            depth_tensor = torch.from_numpy(depth_normalized).unsqueeze(0)
                            depth_images.append(depth_tensor)

    print(f"Total depth images loaded: {len(depth_images)}")
    depth_dataset = torch.stack(depth_images)
    return depth_dataset

def test_trained_vae_model(file, test_images, batch_size=5):
    model = CNNVAE().to(device)
    model.load_state_dict(torch.load(file))
    test_loader = torch.utils.data.DataLoader(test_images, batch_size=batch_size, shuffle=False)
    model.eval()
    all_original = []
    all_reconstructed = []
    with torch.no_grad():
        for batch_data in test_loader:
            batch_data = batch_data.to(device)
            decoded, mu, logvar = model(batch_data)
            all_original.append(batch_data.cpu())
            all_reconstructed.append(decoded.cpu())
    all_original = torch.cat(all_original, dim=0)
    all_reconstructed = torch.cat(all_reconstructed, dim=0)
    return all_original, all_reconstructed, model

def validate_anomalies_with_depth(error_maps, depth_maps, threshold=0.3, error_threshold=0.1):
    valid_anomaly_masks = []
    for error_map, depth_map in zip(error_maps, depth_maps):
        anomaly_mask = error_map > error_threshold
        valid_mask = np.logical_and(anomaly_mask, depth_map > threshold)
        valid_anomaly_masks.append(valid_mask)
    return valid_anomaly_masks

def test_trained_vae_model_depth(file, test_images, batch_size=5):
    model = CNNVAE_Depth().to(device)
    model.load_state_dict(torch.load(file))
    test_loader = torch.utils.data.DataLoader(test_images, batch_size=batch_size, shuffle=False)
    model.eval()
    all_original = []
    all_reconstructed = []
    with torch.no_grad():
        for batch_data in test_loader:
            batch_data = batch_data.to(device)
            decoded, mu, logvar = model(batch_data)
            all_original.append(batch_data.cpu())
            all_reconstructed.append(decoded.cpu())
    all_original = torch.cat(all_original, dim=0)
    all_reconstructed = torch.cat(all_reconstructed, dim=0)
    return all_original, all_reconstructed, model

def show_reconstructed_carla_depth_images(original_images, reconstructed_images):
    index = {'i': 0}
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(bottom=0.2)
    def plot_image():
        axs[0].clear()
        axs[1].clear()
        axs[2].clear()
        original = original_images[index['i']][0] 
        reconstructed = reconstructed_images[index['i']][0] 
        error_map = np.square(original - reconstructed)
        axs[0].imshow(original, cmap='viridis')
        axs[0].set_title(f"Original [{index['i']+1}/{len(original_images)}]")
        axs[0].axis("off")
        axs[1].imshow(reconstructed, cmap='gray')
        axs[1].set_title("Reconstructed")
        axs[1].axis("off")
        axs[2].imshow(error_map, cmap='hot')
        axs[2].set_title("Error Heatmap")
        axs[2].axis("off")
        plt.draw()
    def on_key(event):
        if event.key == 'right':
            index['i'] = (index['i'] + 1) % len(original_images)
        elif event.key == 'left':
            index['i'] = (index['i'] - 1) % len(original_images)
        plot_image()
    def on_close(event):
        plt.close('all')
        sys.exit()
    plot_image()
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('close_event', on_close)
    plt.show()

def show_rgb_and_depth_anomalies(rgb_images, rgb_reconstructions, depth_images, depth_reconstructions, rgb_threshold=0.1, depth_threshold=0.1, unfiltered_rgb_images=None):
    index = {'i': 0}
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    plt.subplots_adjust(bottom=0.2)
    def plot_image():
        axs[0, 0].clear()
        axs[0, 1].clear()
        axs[0, 2].clear()
        axs[1, 0].clear()
        axs[1, 1].clear()
        axs[1, 2].clear()
        rgb_orig = rgb_images[index['i']].transpose(1, 2, 0)
        rgb_recon = rgb_reconstructions[index['i']].transpose(1, 2, 0)
        rgb_error = np.square(rgb_orig - rgb_recon).mean(axis=2)
        depth_orig = depth_images[index['i']][0]  
        depth_recon = depth_reconstructions[index['i']].mean(axis=0)  
        depth_error = np.square(depth_orig - depth_recon)
        rgb_anomaly = rgb_error > rgb_threshold
        depth_anomaly = depth_error > depth_threshold
        common_anomaly = np.logical_and(rgb_anomaly, depth_anomaly)
        pure_anomaly = post_process_pure_anomaly(common_anomaly)

        axs[0, 0].imshow(rgb_orig)
        axs[0, 0].set_title(f"RGB Image [{index['i']+1}/{len(rgb_images)}]")
        axs[0, 0].axis("off")
        axs[0, 1].imshow(rgb_error, cmap='hot')
        axs[0, 1].set_title("RGB Error Map")
        axs[0, 1].axis("off")
        axs[0, 2].imshow(depth_orig, cmap='magma')
        axs[0, 2].set_title("Depth Image")
        axs[0, 2].axis("off")
        axs[1, 0].imshow(depth_error, cmap='hot')
        axs[1, 0].set_title("Depth Error Map")
        axs[1, 0].axis("off")
        axs[1, 1].imshow(common_anomaly, cmap='gray')
        axs[1, 1].set_title("Common Anomalies")
        axs[1, 1].axis("off")
        if unfiltered_rgb_images is not None:
            rgb_unfiltered = unfiltered_rgb_images[index['i']].transpose(1, 2, 0)
        else:
            rgb_unfiltered = rgb_orig
        axs[1, 2].imshow(rgb_unfiltered)
        if np.any(pure_anomaly):
            coords = np.argwhere(pure_anomaly)
            (y_min, x_min), (y_max, x_max) = coords.min(axis=0), coords.max(axis=0)
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                edgecolor='red', facecolor='none', linewidth=2)
            axs[1, 2].add_patch(rect)
        axs[1, 2].set_title("Pure Anomalies")
        axs[1, 2].axis("off")
        plt.draw()
    def on_key(event):
        if event.key == 'right':
            index['i'] = (index['i'] + 1) % len(rgb_images)
        elif event.key == 'left':
            index['i'] = (index['i'] - 1) % len(rgb_images)
        plot_image()
    def on_close(event):
        plt.close('all')
        sys.exit()
    plot_image()
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('close_event', on_close)
    plt.show()
    
def post_process_pure_anomaly(anomaly_mask, min_area=200):
    anomaly_mask = anomaly_mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(anomaly_mask, connectivity=8)
    refined_mask = np.zeros_like(anomaly_mask)
    height, width = anomaly_mask.shape
    for label in range(1, num_labels): 
        x, y, w, h, area = stats[label]
        if x <= 0 or y <= 0 or (x + w) >= width or (y + h) >= height:
            continue  
        if area < min_area:
            continue  
        refined_mask[labels == label] = 1
    return refined_mask


#############################################################################################################################
################################################### MNIST Data AE and VAE ###################################################
#############################################################################################################################

    
# X_train, y_train, X_test, y_test = get_mnist_data()

# print(len(X_train), "training samples")

# sample_images = X_train[:5]
# sample_labels = y_train[:5]
# show_input_mnist_images(sample_images, sample_labels)


# learning_rate = 0.001
# batch_size = 64
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# epochs = 20

# # Autoencoders
# eval_data_ae, decoded_ae = train_mnist_autoencoder(X_train, learning_rate, batch_size, epochs)
# original_images = eval_data_ae.cpu().numpy()
# reconstructed_images = decoded_ae.cpu().numpy()
# show_reconstructed_mnist_images(original_images, reconstructed_images)


# # VAE
# eval_data_vae, decoded_vae = train_mnist_vae(X_train, learning_rate, batch_size, epochs)
# original_images = eval_data_vae.cpu().numpy()
# reconstructed_images = decoded_vae.cpu().numpy()
# show_reconstructed_images(original_images, reconstructed_images)

#############################################################################################################################
################################################### CARLA RGB AE and VAE ####################################################
#############################################################################################################################

train_path = "/home/zozo/workspaces/carla_ws/anovox/Data/Outputs/th1_55_no-anomaly"
test_path = "/home/zozo/workspaces/carla_ws/anovox/Data/Outputs/Final_Output_2025_07_03-12_16/Scenario_4428e77f-1d42-4e37-8f2d-3f1f6c5ddc14/RGB-CAM(0, 0, 1.8)(0, 0, 0)_5663490332425038265"
depth_path = "/home/zozo/workspaces/carla_ws/anovox/Data/Outputs/th1_5_anomaly/Scenario_3c2cf12b-09ec-4706-ac10-2dd24fc9d394/DEPTH_CAM(0, 0, 1.8)(0, 0, 0)_3736024792582289098"
depth_test_path = "/home/zozo/workspaces/carla_ws/anovox/Data/Outputs/Final_Output_2025_07_03-12_16/Scenario_4428e77f-1d42-4e37-8f2d-3f1f6c5ddc14/DEPTH_CAM(0, 0, 1.8)(0, 0, 0)_5663490332425038265"

# # # Training the dataset
# X_train, y_train = get_carla_data(train_path)
# X_train= preprocess(X_train)
# print(len(X_train), "training samples")

## Display sample train images
# sample_images = X_train[:5]
# sample_labels = y_train[:5]
# show_input_carla_images(sample_images, sample_labels)

# Model Parameters
learning_rate = 0.001
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 20

# Autoencoders
# eval_data_ae, decoded_ae = train_carla_autoencoder(X_train.numpy(), learning_rate, batch_size, epochs)
# original_images = eval_data_ae.cpu().numpy()
# reconstructed_images = decoded_ae.cpu().numpy()
# show_reconstructed_carla_images(original_images, reconstructed_images)

# # VAE
# model = train_carla_vae(X_train, learning_rate, batch_size, epochs)
# torch.save(model.state_dict(), 'carla_cnn_vae.pth')

# Testing the dataset
X_test_unfiltered, y_test = load_carla_test_data(test_path)
depth_data = load_carla_depth_images(depth_path)
X_test = preprocess(X_test_unfiltered)
print(len(X_test), "testing samples")

# Evaluating and displaying the results
model_name = './models/carla_cnn_vae_latent_16.pth'
eval_data_vae, decoded_vae, model = test_trained_vae_model(model_name, X_test)
original_images_rgb = eval_data_vae.cpu().numpy()
reconstructed_images_rgb = decoded_vae.cpu().numpy()
# show_reconstructed_carla_images(original_images_rgb, reconstructed_images_rgb)

# # Calculate pure anomalies using depth data
# error_maps = [np.square(orig - recon).mean(axis=0) for orig, recon in zip(original_images, reconstructed_images)]
# valid_anomalies = validate_anomalies_with_depth(error_maps, depth_data, threshold=0.3, error_threshold=0.1)
# show_pure_anomaly_visualization(original_images, reconstructed_images, error_maps, depth_data, valid_anomalies)



#############################################################################################################################
################################################### CARLA depth VAE ##################################################
#############################################################################################################################

# # Sample Images
# depth_images = load_carla_depth_images_cv2(depth_path)
# view_sample_depth_images(depth_images)

# # Training depth data
# train_depth_dataset = prepare_depth_for_training(train_path)
# trained_model = train_carla_vae_depth(train_depth_dataset, learning_rate=0.001, batch_size=64, num_epochs=10)
# torch.save(trained_model.state_dict(), 'carla_depth_cnn_vae.pth')

# Testing depth data
model_name = './models/carla_depth_cnn_vae_depth_10.pth'
test_depth_dataset = prepare_depth_for_testing(depth_test_path)
eval_data_depth, decoded_depth, model = test_trained_vae_model_depth(model_name, test_depth_dataset)
original_images_depth = eval_data_depth.cpu().numpy()
reconstructed_images_depth = decoded_depth.cpu().numpy()
# show_reconstructed_carla_depth_images(original_images_depth, reconstructed_images_depth)


#############################################################################################################################
############################################## CARLA depth and rgb VAE ###############################################
#############################################################################################################################

show_rgb_and_depth_anomalies(
    original_images_rgb, 
    reconstructed_images_rgb, 
    original_images_depth, 
    reconstructed_images_depth,
    0.05,
    0.05,
    unfiltered_rgb_images=X_test_unfiltered.cpu().numpy()
)
