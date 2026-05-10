import os
import torch
import numpy as np
from sklearn.metrics import classification_report

from src.dataset import load_filtered_data, load_images, get_dataloaders
from src.model import MathSymbolCNN
from src.train import train_one_epoch, validate
from src.evaluate import get_predictions, plot_confusion_matrix, show_predictions


BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

torch.manual_seed(42)
np.random.seed(42)

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# load and filter dataset
print('Loading dataset...')
df, idx_to_name = load_filtered_data('data')
print(f'  {len(df)} samples across {len(idx_to_name)} classes')

print('Loading images...')
images, labels = load_images(df, 'data')
print(f'  Image shape: {images.shape}')

# dataloaders
train_loader, test_loader = get_dataloaders(images, labels, BATCH_SIZE)

class_names = [idx_to_name[i] for i in range(len(idx_to_name))]
num_classes = len(class_names)

model = MathSymbolCNN(num_classes=num_classes).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(f'\nTraining ({num_classes} classes):')
print('-' * 50)
for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    val_loss, val_acc = validate(
        model, test_loader, criterion, device
    )
    print(
        f'Epoch {epoch+1}/{EPOCHS}  '
        f'train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  '
        f'val_loss={val_loss:.4f}  val_acc={val_acc:.4f}'
    )

torch.save(model.state_dict(), 'models/math_cnn.pt')
print('\nModel saved to models/math_cnn.pt')

# evaluation
print('\nTest Results:')
print('-' * 50)
preds, true_labels = get_predictions(model, test_loader, device)
print(classification_report(true_labels, preds, target_names=class_names))

plot_confusion_matrix(true_labels, preds, class_names)
print('Confusion matrix saved to results/confusion_matrix.png')

show_predictions(model, test_loader, idx_to_name, device)
print('Prediction grid saved to results/predictions.png')
