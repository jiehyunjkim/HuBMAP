from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple

import monai
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import tifffile
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from monai.data import CSVDataset
from monai.data import DataLoader
from monai.data import ImageReader
from sklearn.model_selection import StratifiedKFold
from tqdm.notebook import tqdm

KAGGLE_DIR = Path("/") / "/Users/jiehyun/kaggle"

INPUT_DIR = KAGGLE_DIR / "input"
OUTPUT_DIR = KAGGLE_DIR / "working"

COMPETITION_DATA_DIR = INPUT_DIR / "hubmap-organ-segmentation"

TRAIN_PREPARED_CSV_PATH = "train_prepared.csv"
VAL_PRED_PREPARED_CSV_PATH = "val_pred_prepared.csv"
TEST_PREPARED_CSV_PATH = "test_prepared.csv"

N_SPLITS = 4
RANDOM_SEED = 2022
SPATIAL_SIZE = 1024
VAL_FOLD = 0
NUM_WORKERS = 2
BATCH_SIZE = 16
MODEL = "unet"
LOSS = "dice"
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.0
FAST_DEV_RUN = False
GPUS = 1
MAX_EPOCHS = 10
PRECISION = 16
DEBUG = False

#change it later
DEVICE = "cpu"
THRESHOLD = 0.5

def add_path_to_df(df: pd.DataFrame, data_dir: Path, type_: str, stage: str) -> pd.DataFrame:
    ending = ".tiff" if type_ == "image" else ".npy"
    
    dir_ = str(data_dir / f"{stage}_{type_}s") if type_ == "image" else f"{stage}_{type_}s"
    df[type_] = dir_ + "/" + df["id"].astype(str) + ending
    return df


def add_paths_to_df(df: pd.DataFrame, data_dir: Path, stage: str) -> pd.DataFrame:
    df = add_path_to_df(df, data_dir, "image", stage)
    df = add_path_to_df(df, data_dir, "mask", stage)
    return df


def create_folds(df: pd.DataFrame, n_splits: int, random_seed: int) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    for fold, (_, val_idx) in enumerate(skf.split(X=df, y=df["organ"])):
        df.loc[val_idx, "fold"] = fold

    return df


def prepare_data(data_dir: Path, stage: str, n_splits: int, random_seed: int) -> None:
    df = pd.read_csv(data_dir / f"{stage}.csv")
    df = add_paths_to_df(df, data_dir, stage)

    if stage == "train":
        df = create_folds(df, n_splits, random_seed)

    filename = f"{stage}_prepared.csv"
    df.to_csv(filename, index=False)

    print(f"Created {filename} with shape {df.shape}")

    return df

train_df = prepare_data(COMPETITION_DATA_DIR, "train", N_SPLITS, RANDOM_SEED)
test_df = prepare_data(COMPETITION_DATA_DIR, "test", N_SPLITS, RANDOM_SEED)

#print(train_df)
#print(test_df)

def rle2mask(mask_rle: str, shape: Tuple[int, int]) -> np.ndarray:
    """
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return
    Returns numpy array, 1 - mask, 0 - background
    Source: https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


def save_array(file_path: str, array: np.ndarray) -> None:
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(file_path, array)


def save_masks(df: pd.DataFrame) -> None:
    for row in tqdm(df.itertuples(), total=len(df)):
        mask = rle2mask(row.rle, shape=(row.img_width, row.img_height))
        save_array(row.mask, mask)

save_masks(train_df)


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

class TIFFImageReader(ImageReader):
    def read(self, data: str) -> np.ndarray:
        image = tifffile.imread(data)
        print(image.shape)
        image = rgb2gray(image)
        print(image.shape)
        return image

    def get_data(self, img: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        return img, {"spatial_shape": np.asarray(img.shape), "original_channel_dim": -1}

    def verify_suffix(self, filename: str) -> bool:
        return ".tiff" in filename

class LitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_csv_path: str,
        test_csv_path: str,
        spatial_size: int,
        val_fold: int,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.train_df = pd.read_csv(train_csv_path)
        self.test_df = pd.read_csv(test_csv_path)

        self.train_transform, self.val_transform, self.test_transform = self._init_transforms()
        
    def _init_transforms(self) -> Tuple[Callable, Callable, Callable]:
        spatial_size = (self.hparams.spatial_size, self.hparams.spatial_size)
        train_transform = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=["image"], reader=TIFFImageReader),
                monai.transforms.AddChanneld(keys=["image"]),
                monai.transforms.ScaleIntensityd(keys=["image"]),
                monai.transforms.LoadImaged(keys=["mask"]),
                monai.transforms.AddChanneld(keys=["mask"]),
                monai.transforms.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
                monai.transforms.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
                monai.transforms.RandRotate90d(keys=["image", "mask"], prob=0.75, spatial_axes=(0, 1)),
                monai.transforms.OneOf(
                    [
                        #monai.transforms.RandGaussianNoised(keys=["image"], prob=1.0, std=0.1),
                        monai.transforms.RandAdjustContrastd(keys=["image"], prob=1.0, gamma=(0.5, 4.5)),
                        monai.transforms.RandShiftIntensityd(keys=["image"], prob=1.0, offsets=(0.1, 0.2)),
                        monai.transforms.RandHistogramShiftd(keys=["image"], prob=1.0),
                    ]
                ),
                monai.transforms.OneOf(
                    [
                        monai.transforms.RandGridDistortiond(keys=["image", "mask"], prob=0.5, distort_limit=0.2),
                        #monai.transforms.RandZoomd(keys=["image", "mask"], prob=0.5, spatial_size=spatial_size,  mode="nearest"),

                    ]
                ),
                monai.transforms.Resized(keys=["image", "mask"], spatial_size=spatial_size, mode="nearest"),
                monai.transforms.ToTensord(keys=["image", "mask"]),
            ]
        )

        val_transform = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=["image"], reader=TIFFImageReader),
                monai.transforms.AddChanneld(keys=["image"]),
                monai.transforms.ScaleIntensityd(keys=["image"]),
                monai.transforms.LoadImaged(keys=["mask"]),
                monai.transforms.AddChanneld(keys=["mask"]),
                #monai.transforms.RandZoomd(keys=["image", "mask"], prob=0.5, spatial_size=spatial_size,  mode="nearest"),
                monai.transforms.Resized(keys=["image", "mask"], spatial_size=spatial_size, mode="nearest"),
                monai.transforms.ToTensord(keys=["image", "mask"]),
            ]
        )

        test_transform = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=["image"], reader=TIFFImageReader),
                monai.transforms.AddChanneld(keys=["image"]),
                monai.transforms.ScaleIntensityd(keys=["image"]),
                monai.transforms.Resized(keys=["image"], spatial_size=spatial_size, mode="nearest"),
                monai.transforms.ToTensord(keys=["image"]),
            ]
        )

        return train_transform, val_transform, test_transform

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            train_df = self.train_df[self.train_df.fold != self.hparams.val_fold].reset_index(drop=True)
            val_df = self.train_df[self.train_df.fold == self.hparams.val_fold].reset_index(drop=True)

            self.train_dataset = self._dataset(train_df, transform=self.train_transform)
            self.val_dataset = self._dataset(val_df, transform=self.val_transform)

        if stage == "test" or stage is None:
            self.test_dataset = self._dataset(self.test_df, transform=self.test_transform)

    def _dataset(self, df: pd.DataFrame, transform: Callable) -> CSVDataset:
        return CSVDataset(src=df, transform=transform)

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset)

    def _dataloader(self, dataset: CSVDataset, train: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

def show_image(title: str, image: np.ndarray, mask: Optional[np.ndarray] = None):
    plt.title(title)
    plt.imshow(image)

    if mask is not None:
        plt.imshow(mask, alpha=0.2)

    plt.tight_layout()
    plt.axis("off")


def show_batch(batch: Dict, nrows: int, show_mask: bool = True):
    fig, _ = plt.subplots(figsize=(3 * nrows, 3 * nrows))

    for idx, _ in enumerate(batch["image"]):
        plt.subplot(nrows, nrows, idx + 1)

        title = batch["id"][idx].numpy()
        image = np.transpose(batch["image"][idx].numpy(), axes=(1, 2, 0))
        mask = np.transpose(batch["mask"][idx].numpy(), axes=(1, 2, 0)) if show_mask else None

        show_image(title, image, mask)


nrows = 1

data_module = LitDataModule(
    train_csv_path=TRAIN_PREPARED_CSV_PATH,
    test_csv_path=TEST_PREPARED_CSV_PATH,
    spatial_size=SPATIAL_SIZE,
    val_fold=VAL_FOLD,
    batch_size=nrows ** 2,
    num_workers=NUM_WORKERS,
)
data_module.setup()

train_batch = next(iter(data_module.train_dataloader()))
show_batch(train_batch, nrows)