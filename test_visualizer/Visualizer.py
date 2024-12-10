import matplotlib.pyplot as plt

def visualize_predictions(images, labels, predictions, class_names, num_images=16):
    """
    Visualize the model's predictions alongside the true labels.

    Parameters:
    - images: Tensor of images (N, C, H, W).
    - labels: Tensor of true labels (N,).
    - predictions: Tensor of predicted labels (N,).
    - class_names: List of class names corresponding to label indices.
    - num_images: Number of images to display.
    """
    images = images[:num_images]
    labels = labels[:num_images]
    predictions = predictions[:num_images]

    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i, ax in enumerate(axes):
        img = images[i].permute(1, 2, 0)  # Convert (C, H, W) to (H, W, C)
        img = img * 0.5 + 0.5  # Unnormalize if normalized
        ax.imshow(img.numpy())
        ax.axis('off')
        true_label = class_names[labels[i]]
        predicted_label = class_names[predictions[i]]
        ax.set_title(f'T: {true_label}\nP: {predicted_label}', fontsize=8, color='green' if labels[i] == predictions[i] else 'red')
    plt.tight_layout()
    plt.show()

