import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


TARGET_SYMBOLS = {
    70: '0', 71: '1', 72: '2', 73: '3', 74: '4',
    75: '5', 76: '6', 77: '7', 78: '8', 79: '9',
    196: '+', 195: '-', 513: 'times', 526: 'div',
    184: 'dot', 524: 'pm', 918: '[', 919: ']'
}


def load_filtered_data(data_dir='data'):
    labels_df = pd.read_csv(os.path.join(data_dir, 'hasy-data-labels.csv'))

    # keep only our target symbols
    filtered = labels_df[labels_df['symbol_id'].isin(TARGET_SYMBOLS.keys())].copy()

    # map symbol_id to class index (0..17)
    class_names = sorted(TARGET_SYMBOLS.keys())
    id_to_idx = {sid: i for i, sid in enumerate(class_names)}
    filtered['class_idx'] = filtered['symbol_id'].map(id_to_idx)

    # readable labels for display
    idx_to_name = {i: TARGET_SYMBOLS[sid] for sid, i in id_to_idx.items()}

    return filtered, idx_to_name


def load_images(df, data_dir='data'):
    images = []
    labels = []

    for _, row in df.iterrows():
        img_path = os.path.join(data_dir, row['path'])
        img = Image.open(img_path).convert('L')  # grayscale
        images.append(np.array(img))
        labels.append(row['class_idx'])

    return np.array(images), np.array(labels)


class MathSymbolDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        return img, torch.tensor(label, dtype=torch.long)


def get_dataloaders(images, labels, batch_size=64, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=test_size, random_state=42, stratify=labels
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = MathSymbolDataset(X_train, y_train, transform=transform)
    test_dataset = MathSymbolDataset(X_test, y_test, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader
