import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.model_selection import train_test_split
import torch.optim as optim


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(28*28, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 4)
                                     )
        
        self.decoder = nn.Sequential(nn.Linear(4, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, 28*28),
                                     nn.Sigmoid()
                                     )
        
    def encode(self, img):
        return self.encoder(img)
    
    def decode(self, code):
        return self.decoder(code)
    
    def forward(self, img):
        code = self.encode(img)
        out = self.decode(code)
        return out


def transform_image(image):
    transform = torchvision.transforms.ToTensor()
    image = image.reshape(28, 28) # to convert to tensor on next line
    image = transform(image) # tensor for model
    return image.view(image.size(0), -1) # convert to "vector"


# Loading data
train = pd.read_csv("/home/ezhsluny/Documents/python/l4/archive/fashion-mnist_train.csv", nrows=5000)
train_x = train[list(train.columns)[1:]].values
train_y = train['label'].values # extracting labels
train_x = train_x / 255 # normalizing

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2)
train_x = train_x.astype(np.float32) # for avoiding Double-Float conflict in model
val_x = val_x.astype(np.float32)


# Learning parameters
num_epochs = 5
model = Autoencoder()
optimizer = optim.Adam(model.parameters(), lr=0.001)
lossf = nn.MSELoss()

# Learning loop
model.train()
for epoch in range(num_epochs):
    losst = 0.0
    for image in train_x:
        image = transform_image(image)
        optimizer.zero_grad()
        autoencoded = model(image)
        loss = lossf(autoencoded, image)
        loss.backward()
        optimizer.step()
        losst += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {losst/len(train)}")


# Testing
model.eval()
results = []
with torch.no_grad():
    for image in val_x:
        image = transform_image(image)
        autoencoded = model(image)
        results.append(autoencoded.detach().numpy()) # convert to np array


# Result's presentation 
fig = plt.figure(figsize=(10, 7))
for i in range(4):
    image = val_x[i].reshape(28, 28)
    result = results[i].reshape(28, 28)
    table = np.concatenate((image, result), axis=1) # comparison picture
    plt.subplot(2, 2, i + 1)
    plt.imshow(table)

plt.show()


# Interactive visualization
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.3)

initial_code = torch.tensor([0.5, 0.5, 0.5, 0.5])
initial_image = model.decode(initial_code).detach().numpy().reshape(28, 28)
img_plot = ax.imshow(initial_image, cmap='gray')

ax_slider1 = plt.axes([0.1, 0.2, 0.8, 0.05])
ax_slider2 = plt.axes([0.1, 0.15, 0.8, 0.05])
ax_slider3 = plt.axes([0.1, 0.1, 0.8, 0.05])
ax_slider4 = plt.axes([0.1, 0.05, 0.8, 0.05])

slider1 = Slider(ax_slider1, 'Code 1', 0.0, 1.0, valinit=0.5)
slider2 = Slider(ax_slider2, 'Code 2', 0.0, 1.0, valinit=0.5)
slider3 = Slider(ax_slider3, 'Code 3', 0.0, 1.0, valinit=0.5)
slider4 = Slider(ax_slider4, 'Code 4', 0.0, 1.0, valinit=0.5)

def update(val):
    code = torch.tensor([slider1.val, slider2.val, slider3.val, slider4.val], dtype=torch.float32)
    decoded_image = model.decode(code).detach().numpy().reshape(28, 28)
    img_plot.set_data(decoded_image)
    fig.canvas.draw_idle()

slider1.on_changed(update)
slider2.on_changed(update)
slider3.on_changed(update)
slider4.on_changed(update)

plt.show()