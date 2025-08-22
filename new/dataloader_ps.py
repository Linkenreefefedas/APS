# file: preprocess_voc_binseg.py
import os
import argparse
import hashlib
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ===== VOC 類別名稱（1..20） =====
VOC_ID2NAME = {
    1:'aeroplane', 2:'bicycle', 3:'bird', 4:'boat', 5:'bottle',
    6:'bus', 7:'car', 8:'cat', 9:'chair', 10:'cow',
    11:'diningtable', 12:'dog', 13:'horse', 14:'motorbike', 15:'person',
    16:'pottedplant', 17:'sheep', 18:'sofa', 19:'train', 20:'tvmonitor'
}
BACKGROUND_IDS = {0, 255}  # 0=背景, 255=ignore

IMG_EXT = ".jpg"  # VOC JPEGImages
MSK_EXT = ".png"  # VOC SegmentationClass

def stable_cache_key(path: str) -> str:
    p = os.path.abspath(path)
    h = hashlib.sha1(p.encode('utf-8')).hexdigest()[:16]
    stem = Path(path).stem
    return f"{stem}__{h}"

def _load_mask_to_class_ids(mask_path: str) -> np.ndarray:
    """
    讀取語意分割圖並回傳 class_id 的 2D numpy array。
    L/P（索引或調色盤）→ 直接拿像素值；RGB → 唯一顏色映射到 id。
    """
    im = Image.open(mask_path)
    if im.mode in ("L", "P"):
        arr = np.array(im, dtype=np.int32)
        return arr
    else:
        arr_rgb = np.array(im.convert("RGB"), dtype=np.uint8)
        H, W, _ = arr_rgb.shape
        flat = arr_rgb.reshape(-1, 3)
        _, inv_idx = np.unique(flat, axis=0, return_inverse=True)
        return inv_idx.reshape(H, W).astype(np.int32)

def _save_img(img_arr: np.ndarray, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(img_arr).save(out_path, quality=95)

def _save_bin_mask(bin_mask: np.ndarray, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray((bin_mask.astype(np.uint8) * 255)).save(out_path)

def _random_crop_coords(H, W, ch, cw, rng: random.Random):
    """回傳 (y0, x0) 左上角座標，保證 0<=y0<=H-ch, 0<=x0<=W-cw。"""
    if H == ch and W == cw:
        return 0, 0
    y0 = 0 if H == ch else rng.randint(0, H - ch)
    x0 = 0 if W == cw else rng.randint(0, W - cw)
    return y0, x0

def _crop_image_and_mask(img_np: np.ndarray, cls_np: np.ndarray, ch: int, cw: int, rng: random.Random):
    """對 RGB 影像與 class_id mask 做同位置隨機裁切。"""
    H, W = img_np.shape[:2]
    if H < ch or W < cw:
        return None, None  # 太小不裁
    y0, x0 = _random_crop_coords(H, W, ch, cw, rng)
    img_c = img_np[y0:y0+ch, x0:x0+cw, ...]
    cls_c = cls_np[y0:y0+ch, x0:x0+cw]
    return img_c, cls_c

def _iter_names_from_txt(txt_path: str):
    with open(txt_path, "r") as f:
        names = [ln.strip() for ln in f.readlines() if ln.strip()]
    return names

def _present_classes(class_ids: np.ndarray):
    # 排除背景/忽略
    vals = np.unique(class_ids)
    return sorted(int(v) for v in vals if v not in BACKGROUND_IDS)

def _row_record(split, sample_id, img_path, mask_path, cid, H, W):
    ckey = stable_cache_key(img_path)
    return {
        "split": split,
        "sample_id": sample_id,
        "image_path": os.path.abspath(img_path),
        "mask_path": os.path.abspath(mask_path),
        "cat_id": int(cid),
        "cat_name": VOC_ID2NAME.get(int(cid), f"class_{int(cid)}"),
        "height": int(H),
        "width": int(W),
        "cache_key": ckey,
        "cache_name": ckey + ".npz",
    }

def preprocess_voc_binseg(
    image_dir: str,
    mask_dir: str,
    traval_txt: str,
    out_root: str,
    train_ratio: float = 0.8,
    seed: int = 2025,
    train_augment_mult: int = 0,
    crop_size: str = "320,480",
):
    """
    讀取單一 txt 名單 -> 內部切 8:2（可調），train 可做增量（隨機裁切）。
    - train/val 都會輸出「原圖對應的二元 mask」樣本
    - 當 train_augment_mult > 0 時，會另外對 train 做裁切增量，並輸出新影像與對應二元 mask
    """
    rng = random.Random(seed)
    os.makedirs(out_root, exist_ok=True)

    # 解析 crop size
    try:
        ch, cw = [int(x) for x in crop_size.split(",")]
    except Exception:
        raise ValueError("--crop-size 格式需為 'H,W'，例如 '320,480'")

    # 讀名字並切分
    names_all = _iter_names_from_txt(traval_txt)
    if len(names_all) == 0:
        print("No filenames found in traval txt.")
        return

    rng.shuffle(names_all)
    n_train = int(round(len(names_all) * train_ratio))
    names_train = names_all[:n_train]
    names_val   = names_all[n_train:]

    print(f"[Split] total={len(names_all)} | train={len(names_train)} | val={len(names_val)} | ratio={train_ratio}")

    rows = {"train": [], "val": []}

    # 路徑準備
    out_mask_train = os.path.join(out_root, "train", "masks")
    out_mask_val   = os.path.join(out_root, "val",   "masks")
    out_img_aug    = os.path.join(out_root, "train_aug", "images")  # 裁切後新影像
    out_mask_aug   = os.path.join(out_root, "train_aug", "masks")   # 裁切後新二元 mask

    os.makedirs(out_mask_train, exist_ok=True)
    os.makedirs(out_mask_val,   exist_ok=True)
    if train_augment_mult > 0:
        os.makedirs(out_img_aug, exist_ok=True)
        os.makedirs(out_mask_aug, exist_ok=True)

    def process_one(stem: str, split: str):
        img_path = os.path.join(image_dir, stem + IMG_EXT)
        msk_path = os.path.join(mask_dir,  stem + MSK_EXT)
        if not (os.path.exists(img_path) and os.path.exists(msk_path)):
            return

        # 讀原圖大小與內容
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        img_np = np.array(img, dtype=np.uint8)

        # class_id mask
        cls_np = _load_mask_to_class_ids(msk_path)
        present = _present_classes(cls_np)
        if not present:
            return

        # 1) 先輸出「原圖 → 每類一張二元 mask」
        for cid in present:
            bin_m = (cls_np == cid).astype(np.uint8)
            if bin_m.sum() == 0:
                continue
            mask_name = f"{stem}__cat{cid}.png"
            if split == "train":
                out_bin_path = os.path.join(out_mask_train, mask_name)
            else:
                out_bin_path = os.path.join(out_mask_val,   mask_name)
            _save_bin_mask(bin_m, out_bin_path)

            rows[split].append(_row_record(
                split=split,
                sample_id=f"{stem}__cat{cid}",
                img_path=img_path,
                mask_path=out_bin_path,
                cid=cid,
                H=H, W=W
            ))

        # 2) 若是 train 且要做增量，做 train_augment_mult 次隨機裁切
        if split == "train" and train_augment_mult > 0:
            if H < ch or W < cw:
                return  # 太小就不裁

            for k in range(train_augment_mult):
                img_c, cls_c = _crop_image_and_mask(img_np, cls_np, ch, cw, rng)
                if img_c is None:
                    continue

                # 儲存裁切後新影像
                stem_aug = f"{stem}__aug{k}"
                img_aug_path = os.path.join(out_img_aug, stem_aug + IMG_EXT)
                _save_img(img_c, img_aug_path)

                present_c = _present_classes(cls_c)
                if not present_c:
                    continue

                for cid in present_c:
                    bin_c = (cls_c == cid).astype(np.uint8)
                    if bin_c.sum() == 0:
                        continue
                    mask_aug_name = f"{stem_aug}__cat{cid}.png"
                    mask_aug_path = os.path.join(out_mask_aug, mask_aug_name)
                    _save_bin_mask(bin_c, mask_aug_path)

                    rows["train"].append(_row_record(
                        split="train",
                        sample_id=f"{stem_aug}__cat{cid}",
                        img_path=img_aug_path,   # 注意：使用新影像路徑
                        mask_path=mask_aug_path,
                        cid=cid,
                        H=ch, W=cw
                    ))

    # 跑 train / val
    for s in tqdm(names_train, desc="Processing train (orig + aug)"):
        process_one(s, "train")
    for s in tqdm(names_val, desc="Processing val (orig only)"):
        process_one(s, "val")

    # 存檔
    for split in ("train", "val"):
        df = pd.DataFrame(rows[split])
        out_csv  = os.path.join(out_root, f"manifest_{split}.csv")
        out_xlsx = os.path.join(out_root, f"manifest_{split}.xlsx")
        df.to_csv(out_csv, index=False)
        df.to_excel(out_xlsx, index=False)
        print(f"[{split}] samples={len(df)}  →  {out_csv}")
    print("Done ✅")


import json
import random
from collections import defaultdict
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

# ======= PASCAL-5i folds (class NAMES must match manifest cat_name) =======
PASCAL5I_FOLDS: Dict[int, List[str]] = {
    0: ['aeroplane','bicycle','bird','boat','bottle'],
    1: ['bus','car','cat','chair','cow'],
    2: ['diningtable','dog','horse','motorbike','person'],
    3: ['pottedplant','sheep','sofa','train','tvmonitor'],
}


def build_psv2_index(manifest_csv: str, fold: int, role: str = "novel", save_json: str | None = None) -> Dict[int, Any]:
    """Build an index for episodic one-shot sampling on PASCAL-5i manifests.

    Args:
        manifest_csv: path to manifest_{train|val}.csv from preprocess_voc_binseg.py
        fold: 0/1/2/3 (see PASCAL5I_FOLDS)
        role: "novel" → keep just the 5 classes in the fold;
              "base"  → keep the remaining 15 classes
        save_json: optional path to dump the index (debug/inspection)

    Returns: dict[int, dict]
        {
          cat_id: {
            "cat_name": str,
            "by_image": { image_path: [row_dict, row_dict, ...], ... }
          }, ...
        }
        Only classes with rows coming from >= 2 distinct images are kept,
        so that support/query can be different images.
    """
    assert fold in (0, 1, 2, 3), "fold must be 0,1,2,3"
    assert role in ("novel", "base"), "role must be 'novel' or 'base'"

    df = pd.read_csv(manifest_csv)
    fold_names = set(PASCAL5I_FOLDS[fold])

    if role == "novel":
        df = df[df["cat_name"].isin(fold_names)]
    else:
        df = df[~df["cat_name"].isin(fold_names)]

    index: Dict[int, Any] = {}
    for cat_id, g in df.groupby("cat_id"):
        by_image = defaultdict(list)
        # cat_name should be consistent within a cat_id
        cat_name = g["cat_name"].iloc[0]
        for _, r in g.iterrows():
            by_image[r["image_path"]].append(dict(r))
        if len(by_image) >= 2:  # need at least 2 images for one-shot (sup/query)
            index[int(cat_id)] = {
                "cat_name": cat_name,
                "by_image": {ip: rows for ip, rows in by_image.items()}
            }

    if save_json:
        with open(save_json, "w") as f:
            json.dump(index, f)
        print(f"Saved index → {save_json} (classes: {len(index)})")
    else:
        print(f"Built index (classes: {len(index)})")

    return index


class _EpisodeBase(Dataset):
    """Shared utilities for episodic one-shot datasets."""

    def __init__(self, index: Dict[int, Any], seed: int = 2025, transform=None):
        assert isinstance(index, dict) and len(index) > 0
        self.index = index
        self.cats: List[int] = list(index.keys())
        self.rng = random.Random(seed)
        self.transform = transform  # if you plug Albumentations etc., keep image/mask in-sync

    @staticmethod
    def _load_row(row: Dict[str, Any]):
        img = Image.open(row["image_path"]).convert("RGB")
        m   = Image.open(row["mask_path"]).convert("L")
        img_t = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float() / 255.0
        m_np  = (np.array(m) > 127).astype(np.uint8)  # bin mask → {0,1}
        m_t   = torch.from_numpy(m_np).long()
        meta = {
            "image_path": row["image_path"],
            "mask_path":  row["mask_path"],
            "cat_id":     int(row["cat_id"]),
            "cat_name":   row["cat_name"],
            "sample_id":  row["sample_id"],
        }
        return img_t, m_t, meta


class OneShotPSV2Random(_EpisodeBase):
    """Training split: uniformly sample one-shot episodes at random.

    The distribution is uniform over classes. For the chosen class, pick two
    different images (support/query) and then pick a random row from each image.
    """

    def __init__(self, index: Dict[int, Any], episodes: int = 10000, seed: int = 2025, transform=None):
        super().__init__(index=index, seed=seed, transform=transform)
        self.episodes = episodes
        self.paths_per_cat: Dict[int, List[str]] = {cid: list(index[cid]["by_image"].keys()) for cid in self.cats}

    def __len__(self):
        return self.episodes

    def __getitem__(self, idx):
        cid = self.rng.choice(self.cats)
        pack = self.index[cid]
        cat_name = pack["cat_name"]
        p1, p2 = self.rng.sample(self.paths_per_cat[cid], 2)  # two distinct images
        r_sup = self.rng.choice(pack["by_image"][p1])
        r_qry = self.rng.choice(pack["by_image"][p2])

        sup_img, sup_m, sup_meta = self._load_row(r_sup)
        qry_img, qry_m, qry_meta = self._load_row(r_qry)

        # Optional user transform: keep image/mask aligned externally if applied
        # (left as a hook; not applied here to avoid KPI changes)
        return {
            "cat_id": cid,
            "cat_name": cat_name,
            "support": {"image": sup_img, "mask": sup_m, "meta": sup_meta},
            "query":   {"image": qry_img, "mask": qry_m, "meta": qry_meta},
        }


class OneShotPSV2RoundRobin(_EpisodeBase):
    """Validation split: cover each class and image-pairs in a round-robin plan.

    For each class, we generate (approximately) disjoint image pairs. If the
    number of images is odd, the last image pairs with the first (ring).
    """

    def __init__(self, index: Dict[int, Any], seed: int = 2025, shuffle_classes: bool = True):
        super().__init__(index=index, seed=seed, transform=None)
        self.shuffle_classes = shuffle_classes
        self.plan = self._build_plan()

    def _build_plan(self):
        plan = []
        cat_ids = list(self.index.keys())
        if self.shuffle_classes:
            self.rng.shuffle(cat_ids)

        for cid in cat_ids:
            pack = self.index[cid]
            cat_name = pack["cat_name"]
            paths = list(pack["by_image"].keys())
            if len(paths) < 2:
                continue
            self.rng.shuffle(paths)

            # Pair neighbors; if odd, last pairs with first
            pairs = []
            if len(paths) % 2 == 0:
                for i in range(0, len(paths), 2):
                    pairs.append((paths[i], paths[i + 1]))
            else:
                for i in range(0, len(paths) - 1, 2):
                    pairs.append((paths[i], paths[i + 1]))
                pairs.append((paths[-1], paths[0]))

            for (ps, pq) in pairs:
                plan.append({
                    "cat_id": cid,
                    "cat_name": cat_name,
                    "sup_path": ps,
                    "qry_path": pq,
                })

        if self.shuffle_classes:
            self.rng.shuffle(plan)
        return plan

    def __len__(self):
        return len(self.plan)

    def _load_from_image(self, pack, image_path):
        rows = pack["by_image"][image_path]
        r = self.rng.choice(rows)
        return self._load_row(r)

    def __getitem__(self, idx):
        ep = self.plan[idx]
        cid = ep["cat_id"]
        pack = self.index[cid]

        sup_img, sup_m, sup_meta = self._load_from_image(pack, ep["sup_path"])
        qry_img, qry_m, qry_meta = self._load_from_image(pack, ep["qry_path"])

        return {
            "cat_id": cid,
            "cat_name": ep["cat_name"],
            "support": {"image": sup_img, "mask": sup_m, "meta": sup_meta},
            "query":   {"image": qry_img, "mask": qry_m, "meta": qry_meta},
        }

    def reset(self):
        self.plan = self._build_plan()


def parse_args():
    p = argparse.ArgumentParser("VOC → binary masks + 8:2 split + train augmentation (random crop)")
    p.add_argument("--image-dir", type=str, required=True, help="VOC JPEGImages 資料夾")
    p.add_argument("--mask-dir",  type=str, required=True, help="VOC SegmentationClass 資料夾")
    p.add_argument("--traval-txt", type=str, required=True, help="同時含 train/val 的 txt（逐行檔名，不含副檔名）")
    p.add_argument("--out-root",  type=str, default="/content/FFS_binseg")
    p.add_argument("--train-ratio", type=float, default=0.8, help="train 比例（預設 0.8）")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--train-augment-mult", type=int, default=0, help="train 每張圖做幾次隨機裁切（0=不做）")
    p.add_argument("--crop-size", type=str, default="320,480", help="H,W（預設 320,480）")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    preprocess_voc_binseg(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        traval_txt=args.traval_txt,
        out_root=args.out_root,
        train_ratio=args.train_ratio,
        seed=args.seed,
        train_augment_mult=args.train_augment_mult,
        crop_size=args.crop_size,
    )
