import numpy as np
import cv2
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import os


# Step 1: Load the MNIST dataset
def load_mnist_dataset():
    (x_train, _), (_, _) = mnist.load_data()
    return x_train


# Step 2: Skew the Image
def skew_image(image):
    rows, cols = image.shape
    # Define source and destination points for skewing
    src_points = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    dst_points = np.float32(
        [[-5, 5], [cols + 5, -5], [-5, rows + 5], [cols + 5, rows - 5]]
    )
    # Calculate the transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    # Apply the transformation
    skewed_image = cv2.warpPerspective(image, matrix, (cols, rows))
    return skewed_image


# Step 3: Rotate the Image
def rotate_image(image, angle=-20):
    rows, cols = image.shape
    # Calculate the center of the image
    center = (cols / 2, rows / 2)
    # Calculate the rotation matrix
    matrix = cv2.getRotationMatrix2D(center, angle, 1)
    # Apply the rotation
    rotated_image = cv2.warpAffine(image, matrix, (cols, rows))
    return rotated_image


# Step 4: Remove Pixels with Gradient (more on the sides of the image)
def remove_pixels_with_gradient(image, side_width=12):
    rows, cols = image.shape
    # Convert image to float to handle NaN values
    masked_image = image.astype(np.float32).copy()

    # Create a gradient mask for the left side
    left_gradient = np.linspace(0, 1, rows).reshape(rows, 1)
    left_mask = np.tile(left_gradient, (1, side_width))
    left_mask = left_mask > np.random.rand(rows, side_width)
    # Only remove non-black pixels
    masked_image[:, :side_width][
        left_mask & (masked_image[:, :side_width] != 0)
    ] = np.nan

    # Create a slight mask for the top side
    top_mask = np.random.rand(side_width, cols) < 0.2
    # Only remove non-black pixels
    masked_image[:side_width, :][
        top_mask & (masked_image[:side_width, :] != 0)
    ] = np.nan

    # Create a slight mask for the bottom side
    bottom_mask = np.random.rand(side_width, cols) < 0.2
    # Only remove non-black pixels
    masked_image[-side_width:, :][
        bottom_mask & (masked_image[-side_width:, :] != 0)
    ] = np.nan

    return masked_image


# Step 5: Add Moiré Pattern
def add_moire_pattern(image, frequency=10, amplitude=30):
    rows, cols = image.shape
    # Create a 1D sinusoidal pattern
    x = np.linspace(0, frequency * np.pi, cols)
    moire_pattern = amplitude * np.sin(x)
    # Tile the pattern across all rows
    moire_pattern = np.tile(moire_pattern, (rows, 1)).astype(np.uint8)
    # Apply the pattern using OpenCV's addWeighted for blending
    moire_image = cv2.addWeighted(image.astype(np.uint8), 0.8, moire_pattern, 0.2, 0)
    return moire_image


# Main function to apply all transformations and save the dataset
def transform_and_save_mnist_dataset():
    x_train = load_mnist_dataset()
    save_dir = "transformed_mnist"
    os.makedirs(save_dir, exist_ok=True)

    for i, image in enumerate(x_train):
        skewed_image = skew_image(image)
        rotated_image = rotate_image(skewed_image, angle=20)
        masked_image = remove_pixels_with_gradient(rotated_image)
        final_image = add_moire_pattern(masked_image, frequency=15, amplitude=30)
        save_path = os.path.join(save_dir, f"image_{i}.png")
        cv2.imwrite(save_path, final_image)

    # Display a few sample images for verification
    plt.figure(figsize=(10, 5))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.title(f"Image {i}")
        plt.imshow(cv2.imread(os.path.join(save_dir, f"image_{i}.png")), cmap="gray")
    plt.show()


# Run the transformation and save the dataset
transform_and_save_mnist_dataset()
