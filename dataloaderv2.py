# file: preprocess_coco_binseg.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils

import hashlib

COCO_ROOT = "/content/COCO2017"
ANN = {
    "train": os.path.join(COCO_ROOT, "annotations", "instances_train2017.json"),
    "val":   os.path.join(COCO_ROOT, "annotations", "instances_val2017.json"),
}
IMAGES = {
    "train": os.path.join(COCO_ROOT, "train2017"),
    "val":   os.path.join(COCO_ROOT, "val2017"),
}
OUT_ROOT = "/content/COCO2017_binseg"

def stable_cache_key(path: str) -> str:
    p = os.path.abspath(path)
    h = hashlib.sha1(p.encode('utf-8')).hexdigest()[:16]
    stem = Path(path).stem
    return f"{stem}__{h}"

def _ann_to_mask(ann, h, w):
    seg = ann.get("segmentation", None)
    if seg is None: return np.zeros((h,w), np.uint8)
    if isinstance(seg, list):
        rles = maskUtils.frPyObjects(seg, h, w)
        rle = maskUtils.merge(rles)
    elif isinstance(seg, dict):
        rle = seg
    else:
        return np.zeros((h,w), np.uint8)
    m = maskUtils.decode(rle)
    return (m.astype(np.uint8) > 0).astype(np.uint8)

def _compose_binary_mask_for_class(coco, img_info, cat_id, include_crowd=False):
    h, w = img_info["height"], img_info["width"]
    ann_ids = coco.getAnnIds(imgIds=[img_info["id"]], catIds=[cat_id])
    anns = coco.loadAnns(ann_ids)
    if not include_crowd:
        anns = [a for a in anns if int(a.get("iscrowd",0)) == 0]
    m = np.zeros((h,w), np.uint8)
    for a in anns:
        m |= _ann_to_mask(a, h, w)
    return (m > 0).astype(np.uint8)

def _save_png(arr, path):
    arr = (arr.astype(np.uint8))*255
    Image.fromarray(arr).save(path)

from pathlib import Path  # 若原檔已 import 可移除

def preprocess_split(
    split: str = "train",
    include_crowd: bool = False,
    shard_count: int = 5,
    shard_idx: int = 0,
):
    """
    將 COCO 該 split 切成 shard_count 份，只處理第 shard_idx 份。
    - shard_idx 從 0 開始，需滿足 0 <= shard_idx < shard_count
    - 切分規則：依 COCO image_id 做取模 (img_id % shard_count == shard_idx)
    """
    assert shard_count >= 1, "--shard-count 必須 >= 1"
    assert 0 <= shard_idx < shard_count, "--shard-idx 必須介於 [0, shard_count)"

    os.makedirs(OUT_ROOT, exist_ok=True)
    out_mask_dir = os.path.join(OUT_ROOT, split, "masks")
    os.makedirs(out_mask_dir, exist_ok=True)

    coco = COCO(ANN[split])
    id2name = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    rows = []
    img_ids = coco.getImgIds()

    # 只挑選本 shard 要處理的影像（用 id 取模，穩定且可平行）
    selected_ids = [iid for iid in img_ids if (int(iid) % shard_count) == shard_idx]

    for img in tqdm(coco.loadImgs(selected_ids),
                    desc=f"Preprocessing {split} [shard {shard_idx}/{shard_count}]"):
        img_id = img["id"]; file_name = img["file_name"]
        h, w = img["height"], img["width"]

        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        if not include_crowd:
            anns = [a for a in anns
                    if int(a.get("iscrowd", 0)) == 0 and a.get("segmentation")]

        present_cats = sorted({a["category_id"] for a in anns})
        if not present_cats:
            continue

        for cid in present_cats:
            m = _compose_binary_mask_for_class(coco, img, cid, include_crowd)
            if m.sum() == 0:
                continue
            stem = Path(file_name).stem
            mask_name = f"{stem}__cat{cid}.png"
            mask_path = os.path.join(out_mask_dir, mask_name)
            _save_png(m, mask_path)

            img_path = os.path.join(IMAGES[split], file_name)
            rows.append({
                "split": split,
                "sample_id": f"{stem}__cat{cid}",
                "image_path": img_path,
                "mask_path": mask_path,
                "cat_id": cid,
                "cat_name": id2name[cid],
                "height": h,
                "width": w,
                "cache_key": stable_cache_key(img_path),
                "cache_name": stable_cache_key(img_path) + ".npz",
            })

    # 每個 shard 產生獨立清單，之後好合併
    out_csv = os.path.join(
        OUT_ROOT, f"manifest_{split}.shard{shard_idx}-of-{shard_count}.csv"
    )
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[{split}] shard {shard_idx}/{shard_count} samples: {len(rows)} → {out_csv}")
    return out_csv



# file: coco20i_indexer.py
import json
from collections import defaultdict
import pandas as pd

COCO20I_FOLDS = {
    0: ['person','airplane','boat','parking meter','dog','elephant','backpack','suitcase',
        'sports ball','skateboard','wine glass','spoon','sandwich','hot dog','chair',
        'dining table','mouse','microwave','refrigerator','scissors'],
    1: ['bicycle','bus','traffic light','bench','horse','bear','umbrella','frisbee','kite',
        'surfboard','cup','bowl','orange','pizza','couch','toilet','remote','oven','book','teddy bear'],
    2: ['car','train','fire hydrant','bird','sheep','zebra','handbag','skis','baseball bat',
        'tennis racket','fork','banana','broccoli','donut','potted plant','tv','keyboard',
        'toaster','clock','hair drier'],
    3: ['motorcycle','truck','stop sign','cat','cow','giraffe','tie','snowboard','baseball glove',
        'bottle','knife','apple','carrot','cake','bed','laptop','cell phone','sink','vase','toothbrush'],
}

def build_coco20i_index(manifest_csv: str, fold: int, role: str = "novel", save_json: str = None):
    assert fold in (0,1,2,3)
    assert role in ("novel","base")

    df = pd.read_csv(manifest_csv)
    fold_names = set(COCO20I_FOLDS[fold])

    if role == "novel":
        df = df[df["cat_name"].isin(fold_names)]
    else:
        df = df[~df["cat_name"].isin(fold_names)]

    index = {}
    for cat_id, g in df.groupby("cat_id"):
        by_image = defaultdict(list)
        cat_name = g["cat_name"].iloc[0]
        for _, r in g.iterrows():
            by_image[r["image_path"]].append(dict(r))
        if len(by_image) >= 2:
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

# file: episodic_coco20i_random.py
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class OneShotCOCO20iRandom(Dataset):
    """
    Training split：隨機均勻抽樣 one-shot episodes。
    依賴 build_coco20i_index() 輸出的 index（已保證每類 ≥2 張不同影像）。
    """
    def __init__(self, index: dict, episodes: int = 10000, seed: int = 2025, transform=None):
        assert isinstance(index, dict) and len(index) > 0
        self.index = index
        self.cats = list(index.keys())
        self.episodes = episodes
        self.rng = random.Random(seed)
        self.transform = transform
        self.paths_per_cat = {cid: list(index[cid]["by_image"].keys()) for cid in self.cats}

    def __len__(self):
        return self.episodes

    def _load(self, row):
        img = Image.open(row["image_path"]).convert("RGB")
        m   = Image.open(row["mask_path"]).convert("L")
        img_t = torch.from_numpy(np.array(img).transpose(2,0,1)).float()/255.0
        m_t   = torch.from_numpy((np.array(m) > 127).astype(np.uint8)).long()
        meta = {
            "image_path": row["image_path"],
            "mask_path": row["mask_path"],
            "cat_id": int(row["cat_id"]),
            "cat_name": row["cat_name"],
            "sample_id": row["sample_id"],
        }
        if self.transform:
            # 可放 Albumentations 等（務必同步處理 image/mask）
            pass
        return img_t, m_t, meta

    def __getitem__(self, idx):
        cid = self.rng.choice(self.cats)
        pack = self.index[cid]
        cat_name = pack["cat_name"]
        p1, p2 = self.rng.sample(self.paths_per_cat[cid], 2)
        r_sup = self.rng.choice(pack["by_image"][p1])
        r_qry = self.rng.choice(pack["by_image"][p2])

        sup_img, sup_m, sup_meta = self._load(r_sup)
        qry_img, qry_m, qry_meta = self._load(r_qry)

        return {
            "cat_id": cid,
            "cat_name": cat_name,
            "support": {"image": sup_img, "mask": sup_m, "meta": sup_meta},
            "query":   {"image": qry_img, "mask": qry_m, "meta": qry_meta},
        }

# file: episodic_coco20i_roundrobin.py
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class OneShotCOCO20iRoundRobin(Dataset):
    """
    Validation split：round-robin 覆蓋取樣 one-shot episodes。
    每輪（__len__）是事先建立好的完整覆蓋計畫；可 reset() 於每個 epoch 重排。
    """
    def __init__(self, index: dict, seed: int = 2025, shuffle_classes: bool = True):
        assert isinstance(index, dict) and len(index) > 0
        self.index = index
        self.rng = random.Random(seed)
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

            # 相鄰配對，奇數個則最後一張配回第一張
            pairs = []
            if len(paths) % 2 == 0:
                for i in range(0, len(paths), 2):
                    pairs.append((paths[i], paths[i+1]))
            else:
                for i in range(0, len(paths)-1, 2):
                    pairs.append((paths[i], paths[i+1]))
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
        from PIL import Image
        img = Image.open(r["image_path"]).convert("RGB")
        m   = Image.open(r["mask_path"]).convert("L")
        import numpy as np, torch
        img_t = torch.from_numpy(np.array(img).transpose(2,0,1)).float()/255.0
        m_t   = torch.from_numpy((np.array(m) > 127).astype(np.uint8)).long()
        meta = {
            "image_path": r["image_path"],
            "mask_path": r["mask_path"],
            "cat_id": int(r["cat_id"]),
            "cat_name": r["cat_name"],
            "sample_id": r["sample_id"],
        }
        return img_t, m_t, meta

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

if __name__ == "__main__":
    # preprocess_split("train", include_crowd=False)
    # preprocess_split("val",   include_crowd=False)
    preprocess_split("train", include_crowd=False, shard_count=5, shard_idx=0)
    preprocess_split("val", include_crowd=False, shard_count=5, shard_idx=0)

    preprocess_split("train", include_crowd=False, shard_count=5, shard_idx=1)
    preprocess_split("val", include_crowd=False, shard_count=5, shard_idx=1)

    preprocess_split("train", include_crowd=False, shard_count=5, shard_idx=2)
    preprocess_split("val", include_crowd=False, shard_count=5, shard_idx=2)

    preprocess_split("train", include_crowd=False, shard_count=5, shard_idx=3)
    preprocess_split("val", include_crowd=False, shard_count=5, shard_idx=3)

    preprocess_split("train", include_crowd=False, shard_count=5, shard_idx=4)
    preprocess_split("val", include_crowd=False, shard_count=5, shard_idx=4)
    
