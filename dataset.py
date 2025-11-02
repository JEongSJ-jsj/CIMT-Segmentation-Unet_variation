from pathlib import Path
import cv2, numpy as np
from torch.utils.data import Dataset
import albumentations as A

class CIMTDataset(Dataset):
    def __init__(self, data_root, split='train', img_size=(256,256)):
        data_root = Path(data_root)
        self.images_dir = data_root/'images'
        self.masks_dir  = data_root/'masks'
        list_file = data_root/(f'{split}.txt')
        if list_file.exists():
            self.ids = [x.strip() for x in list_file.read_text(encoding='utf-8').splitlines() if x.strip()]
        else:
            self.ids = [p.stem for p in sorted(self.images_dir.glob('*.png'))]

        self.transform = A.Compose([
            A.Resize(img_size[1], img_size[0]),
            A.HorizontalFlip(p=0.5 if split=='train' else 0.0),
            A.RandomRotate90(p=0.2 if split=='train' else 0.0),
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        img_path = self.images_dir / f"{id_}.png"
        mask_path = self.masks_dir / f"{id_}.png"

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        assert img is not None and mask is not None, f"Missing {id_}"

        aug = self.transform(image=img, mask=mask)
        img, mask = aug['image'], aug['mask']
        img = (img/255.0).astype('float32')[None, ...]  # (1,H,W)
        mask = (mask>127).astype('float32')[None, ...]
        return img, mask
