import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
import pytorch_lightning as pl
class CustomData(Dataset) :
  def __init__(self, path: str, transform: transforms.Compose) -> None:
    self.path = list(path.glob('*/*'))
    self.transform = transform
  def load_image(self, idx) -> Image.Image:
     image = Image.open(self.path[idx])
     return image
  def __len__(self) -> int:
    return len(self.path)
  def __getitem__(self, idx) -> torch.tensor:
     image = self.transform(self.load_image(idx))
     return image
class DataModule(pl.LightningDataModule) :
  def __init__(self, train_dir, test_dir, transform, batch_size, num_workers) :
    super().__init__()
    self.train_dir = train_dir
    self.test_dir = test_dir
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.transform = transform
  def prepare_data(self) :
    CustomData(self.train_dir, self.transform)
    CustomData(self.test_dir, self.transform)

  def setup(self, stage) :
    entire_ds = CustomData(self.train_dir, self.transform)
    print('---------------', len(entire_ds), '-----------------')
    self.train_ds, self.val_ds = random_split(entire_ds, [40000, 10000])
    self.test_ds = CustomData(self.test_dir, self.transform)
  def train_dataloader(self) :
    return DataLoader(dataset = self.train_ds,
                              batch_size = self.batch_size,
                              num_workers = self.num_workers,
                              shuffle = True)
  def val_dataloader(self) :
    return DataLoader(dataset = self.val_ds,
                              batch_size = self.batch_size,
                              num_workers = self.num_workers,
                              shuffle = False)
  def test_dataloader(self) :
    return DataLoader(dataset = self.test_ds,
                              batch_size = self.batch_size,
                              num_workers = self.num_workers,
                              shuffle = False)