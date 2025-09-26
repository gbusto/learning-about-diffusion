from datasets import load_dataset
from PIL import Image
import numpy as np

def download_data():
    return load_dataset("ylecun/mnist", split="train")

def get_dataset_image(dataset, index: int):
    return dataset[index]["image"]

def save_image(image: Image.Image, path: str):
    image.save(path)

def show_image(image: Image.Image):
    image.show()

# ----------------------------
# Image <-> array helpers
# ----------------------------
def pil_to_vec_minus1_1(img: Image.Image) -> np.ndarray:
    """
    Convert a PIL image to a flat NumPy vector in [-1, 1].
    Ensures 28x28 grayscale for MNIST.
    """
    img = img.convert("L").resize((28, 28), Image.BILINEAR) # Conver the image to grayscale ("L") and resize it to 28x28
    arr = np.asarray(img, dtype=np.float32) / 255.0         # [0,1]
    arr = arr * 2.0 - 1.0                                   # [-1,1]
    return arr.reshape(-1)                                  # (784,)

def vec_minus1_1_to_pil(vec: np.ndarray) -> Image.Image:
    """
    Convert a flat NumPy vector in [-1,1] back to a 28x28 grayscale PIL image.
    """
    img = (np.clip((vec.reshape(28, 28) + 1.0) * 0.5, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(img, mode="L")

# ----------------------------
# Plain Gaussian noise (no schedule)
# ----------------------------
def add_plain_noise(vec_minus1_1: np.ndarray, sigma: float, rng: np.random.RandomState | None = None) -> np.ndarray:
    """
    Add plain Gaussian noise: x_noisy = x + sigma * N(0,1), then clip to [-1,1].
    Useful for a quick visual demo (not the diffusion way yet).
    """
    rng = rng or np.random.RandomState()
    noise = rng.randn(*vec_minus1_1.shape).astype(np.float32)
    x_noisy = vec_minus1_1 + sigma * noise
    return np.clip(x_noisy, -1.0, 1.0)

if __name__ == "__main__":
    mnist_train_dataset = download_data()
    image = get_dataset_image(mnist_train_dataset, 1)

    vectorized_image = pil_to_vec_minus1_1(image)

    for i in range(1, 20):
        noisy_image_vec = add_plain_noise(vectorized_image, sigma=i/10)
        noisy_image_pil = vec_minus1_1_to_pil(noisy_image_vec)
        save_image(noisy_image_pil, f"./tmp/noisy_image_{i}.png")

    show_image(image)
