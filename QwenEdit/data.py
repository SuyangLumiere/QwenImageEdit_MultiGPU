import torch
import random
import math
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

def get_image(path):
    img = Image.open(path).convert("RGB")
    pass

    return img

def list_imgs(path):
    path = Path(path) if isinstance(path, str) else path
    imgs = [p for p in path.rglob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".gif"]]

    return imgs

def path_done_well(*paths, mk_dir=False):

    if not mk_dir:
        path = (p if isinstance(p, Path) else Path(p) for p in paths)

        return path
    else:
        path = [p.resolve() if isinstance(p, Path) else Path(p).resolve() for p in paths]
        for p in path:
            try:
                p.mkdir(exist_ok=True)
            except Exception as e:
                print(f"Failed to create directory {p}: {e}")
        return tuple(path)

class PreprocessDataset(Dataset):
    def __init__(self, txt_cache_dir=None, img_cache_dir=None, ctrl_cache_dir=None, **args):
        super().__init__()

        self.txt_cache_dir, self.img_cache_dir, self.ctrl_cache_dir = path_done_well(
            txt_cache_dir, img_cache_dir, ctrl_cache_dir
        )

        try:
            self.pairs = pd.DataFrame({
                "text": sorted(self.txt_cache_dir.glob("*.pt")),
                "img": sorted(self.img_cache_dir.glob("*.pt")),
                "ctrl": sorted(self.ctrl_cache_dir.glob("*.pt"))
            })
        except Exception as e:
            print(f"Error loading dataset pairs: {e}")
            print(f"text_len: {len(self.txt_cache_dir.glob('*.pt'))}")
            print(f"img_len: {len(self.img_cache_dir.glob('*.pt'))}")
            print(f"ctrl_len: {len(self.ctrl_cache_dir.glob('*.pt'))}")

    def __len__(self):
        return self.pairs.shape[0]

    def __getitem__(self, idx):
        try:
            dat = self.pairs.iloc[idx]
            img = torch.load(dat["img"].resolve())
            txt = torch.load(dat["text"].resolve())
            ctrl = torch.load(dat["ctrl"].resolve())

            return img, txt['prompt_embeds'], txt['prompt_embeds_mask'], ctrl
        except Exception as e:
            print(e)
            new_idx = random.randint(0, len(self)-1)
            return self.__getitem__(new_idx)


def loader(train_batch_size, num_workers, **args):
    dataset = PreprocessDataset(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)
