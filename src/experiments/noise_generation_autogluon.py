from cleanlab import noise_generation
import torchvision
import numpy as np
from pathlib import Path
import pandas as pd
from autogluon.vision import ImageDataset


def generate_noisy_labels_autogluon_image_dataset(
    image_folder: str,
    frac_zero_noise_rates: float = 0.4,
    noise_amount: float = 0.2,
    verbose: int = 1,
):
    """Create class conditional noisy labels for an image dataset

    Code adapted from: https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/cifar_create_label_errors.py

    See paper "Confident Learning: Estimating Uncertainty in Dataset Labels" by Northcutt et al. (refer to as "CL paper")
    https://arxiv.org/abs/1911.00068

    Parameters
    ----------
    image_folder : str
      Path to images in the same format expected by the `torchvision.datasets.ImageFolder()` method.

    frac_zero_noise_rates : float, default=0.4
      AKA sparsity as defined in CL paper.

    noise_amount : float, default=0.2
      Noise amount as defined in CL paper.

    verbose : int, default=1
      Set this = 0 to suppress all print statements.


    Returns
    -------
    dataset_with_noisy_labels : gluoncv.auto.data.dataset.ImageClassificationDataset
      Autogluon dataset with image paths and noisy labels.

    """

    # Read data from image folder
    dataset = torchvision.datasets.ImageFolder(
        root=image_folder,
    )

    # labels
    y = dataset.targets

    # number of classes
    K = len(dataset.classes)

    # Generate class-conditional noise
    nm = noise_generation.generate_noise_matrix_from_trace(
        K=K,
        trace=int(K * (1 - noise_amount)),
        valid_noise_matrix=False,
        frac_zero_noise_rates=frac_zero_noise_rates,
        seed=0,
    )

    # noise matrix is valid if diagonal maximizes row and column
    assert all((nm.argmax(axis=0) == range(K)) & (nm.argmax(axis=1) == range(K)))

    # Create noisy labels
    np.random.seed(seed=0)
    s = noise_generation.generate_noisy_labels(y, nm)

    # Check accuracy of s and y
    if verbose:
        print("Generated noisy labels with config:")
        print(f"  noise_amount         : {noise_amount}")
        print(f"  frac_zero_noise_rates: {frac_zero_noise_rates}")
        print("Accuracy of noisy labels (s) and true labels (y):", sum(s == y) / len(s))
        print(f"Classes: {dataset.classes}")

    # Create DataFrame with file paths and noisy labels
    # Use "image" and "label" column names to follow AutoGluon convention
    image_file_paths = np.array(dataset.imgs)[:, 0]
    df = pd.DataFrame(
        {"image": image_file_paths, "label": s}  # file paths  # noisy label
    )
    df["image"] = df.image.map(lambda f: str(Path(f).resolve()))

    # Convert to AutoGluon ImageClassificationDataset
    df = ImageDataset(df, classes=dataset.classes)

    return df
