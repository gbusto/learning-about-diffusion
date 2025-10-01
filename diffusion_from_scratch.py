from datasets import load_dataset
from PIL import Image
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import os
import math
import argparse

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

def get_mnist_images(dataset, start_index: int, end_index: int) -> List[Image.Image]:
    """Return a list of MNIST examples as PIL.Images (grayscale)."""
    return [dataset[i]["image"] for i in range(start_index, end_index)]


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
        # Helpful for visualizing noise being added in a nice way;
        #   basically, forward noising is just turning up the opacity of the "noise"
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


# ====== Generate training data (returns arrays ready for saving/training) ======

def generate_training_data(
    schedule: DiffusionSchedule,
    images: List[Image.Image],
    versions_per_image: int = 10,
    rng: Optional[np.random.RandomState] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Produce parallel arrays:
      X_t : (N, 784) float32  — noisy inputs
      t   : (N,)    int64     — integer timesteps in [0, T-1]
      X0  : (N, 784) float32  — clean targets

    N = len(images) * versions_per_image
    """
    rng = rng or np.random.RandomState()
    T = len(schedule.alpha_bar)

    X_t_list: list[np.ndarray] = []
    t_list:  list[int]         = []
    X0_list: list[np.ndarray]  = []

    for img in images:
        x0_vec = pil_to_vec_minus1_1(img)  # (784,), [-1,1]
        for _ in range(versions_per_image):
            t_int = int(rng.randint(0, T))
            x_t_vec = forward_noisy_sample(
                x0_vec=x0_vec,
                timestep=t_int,
                schedule=schedule,
                noise=None,            # fresh Gaussian noise inside
                clip_for_viz=False     # don't clip during training data creation
            )
            X_t_list.append(x_t_vec)
            t_list.append(t_int)
            X0_list.append(x0_vec)

    # Convert to arrays in the right dtypes/shapes for saving and training
    X_t = np.stack(X_t_list).astype(np.float32)   # (N, 784)
    t   = np.asarray(t_list, dtype=np.int64)      # (N,)
    X0  = np.stack(X0_list).astype(np.float32)    # (N, 784)
    return X_t, t, X0


# ====== Main: build schedule, generate arrays, save .npz ======

if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42424242, help="RNG seed")
    parser.add_argument("--num-steps", type=int, default=100, help="diffusion steps T")
    parser.add_argument("--num-images", type=int, default=1000, help="how many MNIST images to use")
    parser.add_argument("--versions-per-image", type=int, default=10, help="noisy examples per image")
    parser.add_argument("--out", type=str, default="./tmp/training_data.npz", help="output .npz path")
    parser.add_argument("--preview", type=int, default=0, help="optional: save 2*preview tiles [noisy, clean]")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rng = np.random.RandomState(args.seed)

    # 1) Load data
    ds = load_mnist_train()
    images = get_mnist_images(ds, 0, args.num_images)

    # 2) Build schedule
    schedule = build_linear_schedule(num_steps=args.num_steps, beta_start=1e-4, beta_end=2e-2)

    # 3) Generate arrays ready for saving / training
    X_t, t, X0 = generate_training_data(
        schedule=schedule,
        images=images,
        versions_per_image=args.versions_per_image,
        rng=rng
    )

    # 4) Save a single .npz with named arrays
    np.savez(args.out, X_t=X_t, t=t, X0=X0)
    print(f"Saved {args.out}")
    print(f"  X_t: {X_t.shape} {X_t.dtype}  | t: {t.shape} {t.dtype}  | X0: {X0.shape} {X0.dtype}")

    # 5) (optional) quick visual sanity check: [noisy, clean] pairs
    if args.preview > 0:
        k = min(args.preview, X_t.shape[0])
        tiles: list[Image.Image] = []
        for i in range(k):
            tiles.append(vec_minus1_1_to_pil(np.clip(X_t[i], -1.0, 1.0)))
            tiles.append(vec_minus1_1_to_pil(np.clip(X0[i], -1.0, 1.0)))
        save_pil_grid(tiles, "./tmp/training_preview.png", cols=2)
