import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


def get_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(y_true, y_pred, class_names, save_dir='results'):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=7)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/confusion_matrix.png', dpi=120)
    plt.close()


def show_predictions(model, loader, idx_to_name, device, n=16, save_dir='results'):
    model.eval()
    images_shown = []
    preds_shown = []
    labels_shown = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            for i in range(images.size(0)):
                if len(images_shown) >= n:
                    break
                images_shown.append(images[i].cpu())
                preds_shown.append(preds[i].item())
                labels_shown.append(labels[i].item())

            if len(images_shown) >= n:
                break

    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8, 2 * rows))
    axes = axes.flatten()

    for i in range(n):
        img = images_shown[i].squeeze().numpy()
        # undo normalization for display
        img = img * 0.5 + 0.5
        axes[i].imshow(img, cmap='gray')
        pred_name = idx_to_name[preds_shown[i]]
        true_name = idx_to_name[labels_shown[i]]
        color = 'green' if preds_shown[i] == labels_shown[i] else 'red'
        axes[i].set_title(f'pred: {pred_name}\ntrue: {true_name}', fontsize=8, color=color)
        axes[i].axis('off')

    for i in range(n, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/predictions.png', dpi=100)
    plt.close()
