'''
1) Скачать датасет   https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
2) Написать нейросеть состоящую из свёрточного блока и полносвязного слоя
3) Используя MSELoss обучить нейронку кодировать изображения одного человека (одного класса) похожим образом
4) С помощью t-SNE визуализировать результаты работы (использовать тестовый датасет)
'''


import torch
import torch.nn as nn
import cv2 as cv
import numpy as np
from torch.nn import functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split


class SiamNet(nn.Module):

    def __init__(self, out_channels):
        super(SiamNet, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3),
                                     nn.BatchNorm2d(64),
                                     nn.LeakyReLU(0.2),
                                     nn.Conv2d(64, 128, kernel_size=3),
                                     nn.BatchNorm2d(128),
                                     nn.LeakyReLU(0.2),
                                     nn.Flatten()
                                     )
        self.sigmoid = nn.Sigmoid()

    def encode(self, img):
        return self.encoder(img)

    def forward(self, input1, input2):
        output1 = self.encode(input1)
        output2 = self.encode(input2)

        distance = F.pairwise_distance(output1, output2)

        return self.sigmoid(distance)


class FaceDataset(Dataset):
    def __init__(self, size, sample_size, data_path='l2/att_faces'):
        self.size = size
        self.sample_size = sample_size
        self.data_path = data_path
        self.data, self.labels = self.load_data()
        self.transform = transforms.ToTensor()

    def load_data(self):
        image = cv.imread('l2/att_faces/s1/1.pgm', 0)
        image = image[::self.size, ::self.size]

        x_same_pair = []
        y_same = []

        for folder in range(40):
            for img in range(int(self.sample_size / 40)):
                ind1, ind2 = np.random.choice(range(1, 11), 2, replace=False)

                img1 = cv.imread(f'l2/att_faces/s{folder + 1}/{ind1}.pgm', 0)
                img2 = cv.imread(f'l2/att_faces/s{folder + 1}/{ind1}.pgm', 0)
                img1 = img1[::self.size, ::self.size]
                img2 = img2[::self.size, ::self.size]
                x_same_pair.append((img1, img2))
                y_same.append(1)

        x_diff_pair = []
        y_diff = []

        for img in range(10):
            for folder in range(int(self.sample_size / 10)):
                ind1, ind2 = np.random.choice(range(1, 41), 2, replace=False)
                img1 = cv.imread(f'l2/att_faces/s{ind1}/{img + 1}.pgm', 0)
                img2 = cv.imread(f'l2/att_faces/s{ind2}/{img + 1}.pgm', 0)
                img1 = img1[::self.size, ::self.size]
                img2 = img2[::self.size, ::self.size]

                x_diff_pair.append((img1, img2))
                y_diff.append(0)

        data = x_same_pair + x_diff_pair
        labels = np.array(y_same + y_diff)

        print(f"Loaded {len(data)} samples")

        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img1, img2 = self.data[idx]
        label = self.labels[idx]

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)


def contrastive_loss(distance, label, margin=1.0):
    loss = torch.mean((label) * torch.pow(distance, 2) +
                      (1 - label) * torch.pow(torch.clamp(margin - distance, min=0.0), 2))
    return loss


# Preparing data
size = 2
sample_size = 1000
dataset = FaceDataset(size, sample_size)

train, test = random_split(dataset, [0.8, 0.2])
train = DataLoader(train, batch_size=32, shuffle=True)
test = DataLoader(test, batch_size=32, shuffle=True)
print('\nDataset splitted')


# Training loop
model = SiamNet(1)
epochs = 3
optimizer = optim.Adam(model.parameters(), lr=0.001)


for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for i, (img1, img2, label) in enumerate(train):
        optimizer.zero_grad()
        distance = model(img1, img2)
        loss = contrastive_loss(distance, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train)}")



# Testing loop
model.eval()
test_loss = 0.0
with torch.no_grad():
    for i, (img1, img2, label) in enumerate(test):
        distance = model(img1, img2) 
        loss = contrastive_loss(distance, label)
        test_loss += loss.item()

print(f"Test Loss: {test_loss/len(test)}")


# Visualization
model.eval()
num_visualizations = 5

indices = np.random.choice(len(test.dataset), num_visualizations, replace=False)
fig, axes = plt.subplots(num_visualizations, 2, figsize=(6, 2 * num_visualizations))

with torch.no_grad():
    for i, idx in enumerate(indices):
        img1, img2, label = test.dataset[idx]

        distance = model(img1.unsqueeze(0), img2.unsqueeze(0))

        img1 = img1.numpy().squeeze()
        img2 = img2.numpy().squeeze()


        axes[i, 0].imshow(img1, cmap='gray')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(img2, cmap='gray')
        axes[i, 1].set_title(f"Distance: {distance.item():.4f}\nLabel: {int(label.item())}")
        axes[i, 1].axis('off')

plt.tight_layout()
plt.show()