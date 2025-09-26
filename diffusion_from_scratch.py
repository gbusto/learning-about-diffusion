from datasets import load_dataset
from PIL import Image
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import os
import math

# ====== Constants ======
IMAGE_SIZE = (28, 28)                 # MNIST size
VECTOR_LENGTH = IMAGE_SIZE[0] * IMAGE_SIZE[1]
GRID_COLS = 8                         # max images per row when saving a grid


# ====== Dataset helpers ======

def load_mnist_train():
    """Download the MNIST training split and return a Hugging Face Dataset."""
    return load_dataset("mnist", split="train")

def get_mnist_image(dataset, index: int) -> Image.Image:
    """Return a single MNIST example as a PIL.Image (grayscale)."""
    return dataset[index]["image"]


# ====== Image <-> Vector ([-1, 1]) ======

def pil_to_vec_minus1_1(img: Image.Image) -> np.ndarray:
    """
    Convert a PIL image to a flat NumPy vector in [-1, 1].
    Ensures 28x28 grayscale.
    """
    img = img.convert("L").resize(IMAGE_SIZE, Image.BILINEAR)
    arr_0_1 = np.asarray(img, dtype=np.float32) / 255.0       # [0,1]
    arr_m11 = (arr_0_1 * 2.0) - 1.0                           # [-1,1]
    return arr_m11.reshape(VECTOR_LENGTH)                      # (784,)

def vec_minus1_1_to_pil(vec: np.ndarray) -> Image.Image:
    """
    Convert a flat vector in [-1,1] back to a 28x28 grayscale PIL image.
    """
    h, w = IMAGE_SIZE
    arr_m11 = vec.reshape(h, w)
    arr_0_1 = np.clip((arr_m11 + 1.0) * 0.5, 0.0, 1.0)        # [-1,1] -> [0,1]
    arr_u8  = (arr_0_1 * 255.0).astype(np.uint8)
    return Image.fromarray(arr_u8, mode="L")


# ====== Grid saver ======

def save_pil_grid(images: List[Image.Image], path: str, cols: int = GRID_COLS) -> None:
    """
    Save a list of 28x28 PIL images into a grid PNG, wrapping after `cols` images.
    """
    if not images:
        raise ValueError("No images to save.")

    w, h = IMAGE_SIZE
    rows = math.ceil(len(images) / cols)
    canvas = Image.new("L", (cols * w, rows * h))

    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        canvas.paste(im, (c * w, r * h))

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    canvas.save(path)
    print(f"Saved grid to {path}")


# ====== Diffusion scheduler ======

@dataclass
class DiffusionSchedule:
    """
    Holds all fixed per-timestep numbers for forward noising (and later for DDIM).
    - betas[t]  : how much noise is added at step t
    - alphas[t] : 1 - betas[t]
    - alpha_bar[t] (ā_t): cumulative product of alphas up to t
                           ("how much clean signal survives by step t")
    """
    betas: np.ndarray        # shape (T,)
    alphas: np.ndarray       # shape (T,)
    alpha_bar: np.ndarray    # shape (T,)

    def alpha_bar_at(self, t: int) -> float:
        """Return ā_t as a plain float for a given timestep t."""
        return float(self.alpha_bar[t])


def build_linear_schedule(num_steps: int,
                          beta_start: float = 1e-4,
                          beta_end: float = 2e-2) -> DiffusionSchedule:
    """
    Create a simple linear beta schedule and the derived arrays needed for diffusion.
    """
    betas = np.linspace(beta_start, beta_end, num_steps, dtype=np.float32)     # (T,)
    alphas = 1.0 - betas                                                       # (T,)
    alpha_bar = np.cumprod(alphas, axis=0).astype(np.float32)                  # (T,)
    return DiffusionSchedule(betas=betas, alphas=alphas, alpha_bar=alpha_bar)


# ====== Forward noising (q-sample) ======

def forward_noisy_sample(x0_vec: np.ndarray,
                         timestep: int,
                         schedule: DiffusionSchedule,
                         noise: Optional[np.ndarray] = None,
                         clip_for_viz: bool = True) -> np.ndarray:
    """
    Create a noisy version x_t of a clean image vector x0 at a specific timestep.

      x_t = sqrt(ā_t) * x0 + sqrt(1 - ā_t) * eps,   where eps ~ N(0, I)

    Args:
        x0_vec:    clean image vector in [-1,1], shape (784,)
        timestep:  integer t in [0, T-1]
        schedule:  DiffusionSchedule containing alpha_bar
        noise:     optional fixed noise vector (same shape as x0_vec) for reproducible ladders
        clip_for_viz: clip to [-1,1] before returning (nice for saving PNGs)

    Returns:
        x_t_vec: noisy image vector, same shape as x0_vec
    """
    a_bar_t = schedule.alpha_bar_at(timestep)                  # ā_t
    sqrt_a_bar = np.sqrt(a_bar_t)                              # sqrt(ā_t)
    sqrt_one_minus_a_bar = np.sqrt(max(1.0 - a_bar_t, 0.0))    # sqrt(1 - ā_t)

    if noise is None:
        noise = np.random.randn(*x0_vec.shape).astype(np.float32)

    x_t_vec = (sqrt_a_bar * x0_vec) + (sqrt_one_minus_a_bar * noise)

    if clip_for_viz:
        x_t_vec = np.clip(x_t_vec, -1.0, 1.0)

    return x_t_vec


# ====== Progression builders ======

def evenly_spaced_timesteps(total_steps: int, count: int) -> np.ndarray:
    """
    Pick `count` integers evenly spaced from 0 to total_steps-1 (inclusive).
    Example: total_steps=100, count=8 -> array([ 0, 14, 28, 42, 56, 70, 85, 99])
    """
    return np.linspace(0, total_steps - 1, count, dtype=int)

def make_clean_to_noisy_tiles(x0_vec: np.ndarray,
                              schedule: DiffusionSchedule,
                              tiles_count: int = 16,
                              reuse_same_noise: bool = True) -> List[Image.Image]:
    """
    Build a list of images from nearly clean (small t) to very noisy (large t).
    If `reuse_same_noise` is True, the *same* noise vector is reused at each step
    for a smoother visual progression.
    """
    T = len(schedule.alpha_bar)
    timesteps = evenly_spaced_timesteps(T, tiles_count)

    base_noise = None
    if reuse_same_noise:
        base_noise = np.random.randn(*x0_vec.shape).astype(np.float32)

    pil_images: List[Image.Image] = []
    for t in timesteps:
        x_t = forward_noisy_sample(
            x0_vec=x0_vec,
            timestep=int(t),
            schedule=schedule,
            noise=base_noise,
            clip_for_viz=True
        )
        pil_images.append(vec_minus1_1_to_pil(x_t))

    return pil_images


# ====== Main demo ======

if __name__ == "__main__":
    if not os.path.exists("./tmp"):
        os.makedirs("./tmp")

    # 1) Load one or more MNIST images
    ds = load_mnist_train()
    img0 = get_mnist_image(ds, 0)
    img1 = get_mnist_image(ds, 1)
    img2 = get_mnist_image(ds, 2)

    # 2) Convert to vectors in [-1,1]
    x0_vec = pil_to_vec_minus1_1(img0)
    x1_vec = pil_to_vec_minus1_1(img1)
    x2_vec = pil_to_vec_minus1_1(img2)

    # 3) Build a simple linear schedule
    NUM_STEPS = 100  # keep small for visualization
    schedule = build_linear_schedule(num_steps=NUM_STEPS, beta_start=1e-4, beta_end=2e-2)

    # 4) Make and save a single clean→noisy grid for one digit
    tiles = make_clean_to_noisy_tiles(x0_vec, schedule, tiles_count=16, reuse_same_noise=True)
    save_pil_grid(tiles, "./tmp/mnist_clean_to_noisy.png", cols=GRID_COLS)

    # 5) (Optional) Make a 3-row grid: 3 different digits, each across 8 steps
    all_tiles: List[Image.Image] = []
    for vec in (x0_vec, x1_vec, x2_vec):
        row_tiles = make_clean_to_noisy_tiles(vec, schedule, tiles_count=8, reuse_same_noise=True)
        all_tiles.extend(row_tiles)
    save_pil_grid(all_tiles, "./tmp/mnist_three_rows.png", cols=GRID_COLS)
