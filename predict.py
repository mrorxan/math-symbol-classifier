import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from src.dataset import TARGET_SYMBOLS, load_filtered_data, load_images
from src.model import MathSymbolCNN


_, idx_to_name = load_filtered_data('data')
num_classes = len(idx_to_name)

model = MathSymbolCNN(num_classes=num_classes)
model.load_state_dict(torch.load('models/math_cnn.pt', map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# grab some random test images to classify
df, _ = load_filtered_data('data')
images, labels = load_images(df, 'data')

np.random.seed(123)
indices = np.random.choice(len(images), size=12, replace=False)

fig, axes = plt.subplots(3, 4, figsize=(8, 6))
axes = axes.flatten()

print('Sample Predictions:')
print('=' * 40)

for i, idx in enumerate(indices):
    img = Image.fromarray(images[idx])
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        pred = output.argmax(dim=1).item()

    pred_name = idx_to_name[pred]
    true_name = idx_to_name[labels[idx]]

    print(f'  True: {true_name:6s}  Predicted: {pred_name:6s}  {"✓" if pred == labels[idx] else "✗"}')

    axes[i].imshow(images[idx], cmap='gray')
    color = 'green' if pred == labels[idx] else 'red'
    axes[i].set_title(f'pred: {pred_name} | true: {true_name}', fontsize=7, color=color)
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('results/sample_predictions.png', dpi=100)
plt.close()
print('\nSaved to results/sample_predictions.png')
