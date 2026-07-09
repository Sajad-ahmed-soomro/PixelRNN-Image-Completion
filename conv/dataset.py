# dataset.py
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import os

def default_transforms(imsize=256):
    return T.Compose([
        T.Resize((imsize, imsize)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

class PairedDataset(Dataset):
    """
    Expects each file in root_dir to be a paired image: [left | right]
    Left can be grayscale edge map or RGB; right is color photo.
    """
    def __init__(self, root_dir, transforms=None):
        super().__init__()
        self.root_dir = root_dir
        self.files = sorted([
            os.path.join(root_dir, f) for f in os.listdir(root_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        if len(self.files) == 0:
            raise RuntimeError(f"No image files found in {root_dir}")
        self.transforms = transforms if transforms is not None else default_transforms()

    def __len__(self):
        return len(self.files)

    def _open_pair(self, path):
        img = Image.open(path)
        img = img.convert("RGB")  # convert both halves to RGB for consistency
        w, h = img.size
        w2 = w // 2
        left = img.crop((0, 0, w2, h))
        right = img.crop((w2, 0, w, h))
        return left, right

    def __getitem__(self, idx):
        path = self.files[idx]
        left, right = self._open_pair(path)
        if self.transforms:
            left = self.transforms(left)
            right = self.transforms(right)
        return {'A': left, 'B': right}
