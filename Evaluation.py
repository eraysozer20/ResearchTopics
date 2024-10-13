import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from PIL import Image

def load_image(image_path):
    """Load an image from a file path."""
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def add_gaussian_noise(image, mean=0, stddev=25):
    # Convert image to numpy array
    image_array = np.array(image)

    # Generate Gaussian noise
    noise = np.random.normal(mean, stddev, image_array.shape)

    # Add the noise to the image
    noisy_image_array = image_array + noise

    # Clip the pixel values to be valid (0-255)
    noisy_image_array = np.clip(noisy_image_array, 0, 255).astype(np.uint8)

    # Convert back to image
    noisy_image = Image.fromarray(noisy_image_array)

    return noisy_image

def mean_squared_error_metric(original, imputed):
    """Calculate Mean Squared Error (MSE) between original and imputed images."""
    return mean_squared_error(original.flatten(), imputed.flatten())

def structural_similarity_index(original, imputed):
    """Calculate Structural Similarity Index (SSIM) between original and imputed images."""
    return ssim(original, imputed)

def peak_signal_to_noise_ratio(original, imputed):
    """Calculate Peak Signal-to-Noise Ratio (PSNR) between original and imputed images."""
    mse_value = mean_squared_error(original.flatten(), imputed.flatten())
    if mse_value == 0:
        return float('inf')  # Return infinity if MSE is zero
    max_pixel_value = 255.0
    return 10 * np.log10(max_pixel_value**2 / mse_value)

def visualize_images(original, imputed):
    """Visualize original and imputed images side by side."""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(original, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Imputed Image')
    plt.imshow(imputed, cmap='gray')
    plt.axis('off')

    plt.show()

def evaluate_images(original_path, imputed_path):
    """Evaluate the imputation performance based on various metrics."""
    original = load_image(original_path)
    imputed = load_image(imputed_path)

    mse_value = mean_squared_error_metric(original, imputed)
    ssim_value = structural_similarity_index(original, imputed)
    psnr_value = peak_signal_to_noise_ratio(original, imputed)

    print(f'Mean Squared Error (MSE): {mse_value:.4f}')
    print(f'Structural Similarity Index (SSIM): {ssim_value:.4f}')
    print(f'Peak Signal-to-Noise Ratio (PSNR): {psnr_value:.4f} dB')

    visualize_images(original, imputed)

# Example usage
# evaluate_images('path/to/original/image.png', 'path/to/imputed/image.png')