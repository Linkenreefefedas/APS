# =============================================================
# File: supseg_dataset.py
# Description: Dataset adapter that consumes your meta-episode source (e.g., COCO-20i) and
#              reuses cached SAM2/DINO features from your existing cache directory.
#              This avoids the global `args` leak and keeps everything local to the dataset.
# =============================================================
from dataclasses import dataclass
from typing import List
import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# Reuse helpers from your toolbox train.py (uploaded):
from train import cache_key, _sam2_embed_from_cached, sobel_edge_hint, support_mask_to_grid

@dataclass
class EpisodeItem:
    support: dict  # expects keys: image(np.float32 CHW 0..1), mask(np.float32 HW 0/1), meta{image_path}
    query: dict    # same structure as support

class SupEpisodeAdapter(Dataset):
    def __init__(self, base_dataset, cache_dir: str, cache_long: int = 1024,
                 kmax_f: int = 1, kmax_b: int = 1):
        self.base = base_dataset
        self.cache_dir = cache_dir
        self.cache_long = int(cache_long)
        self.kmax_f = int(kmax_f)
        self.kmax_b = int(kmax_b)

    def __len__(self):
        return len(self.base)

    def _load_cached(self, img_path: str):
        key = cache_key(img_path)
        pt = os.path.join(self.cache_dir, f"{key}.pt")
        assert os.path.isfile(pt), f"Cache not found: {pt}"
        data = torch.load(pt, map_location="cpu")
        sam = _sam2_embed_from_cached(data["sam2"])   # [Cs,Hf,Wf] float32
        dng = data["dng"].to(torch.float32)           # [Cd,Gh,Gw]
        return sam, dng, data

    def __getitem__(self, i):
        ep = self.base[i]
        sup = ep['support']; qry = ep['query']
        sup_img = (sup['image'].numpy().transpose(1,2,0) * 255.0).astype(np.uint8)
        qry_img = (qry['image'].numpy().transpose(1,2,0) * 255.0).astype(np.uint8)
        sup_mask = (sup['mask'].numpy().astype(np.uint8) * 255)
        qry_mask = (qry['mask'].numpy().astype(np.uint8) * 255)
        sup_path = sup['meta']['image_path']; qry_path = qry['meta']['image_path']

        sam_r, dn_r, _ = self._load_cached(sup_path)
        sam_t, dn_t, _ = self._load_cached(qry_path)

        # Prototypes on support
        C,Hr,Wr = sam_r.shape
        m_small = cv2.resize(sup_mask, (Wr,Hr), interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
        m_small = torch.from_numpy(m_small)
        fg = (m_small>0.5).float(); bg = (m_small<=0.5).float()
        proto_fg_sam = (sam_r * fg.unsqueeze(0)).sum((1,2)) / (fg.sum()+1e-8)
        proto_bg_sam = (sam_r * bg.unsqueeze(0)).sum((1,2)) / (bg.sum()+1e-8)
        proto_fg_sam = F.normalize(proto_fg_sam, dim=0)
        proto_bg_sam = F.normalize(proto_bg_sam, dim=0)

        Cd,Gh,Gw = dn_r.shape
        m_small2 = cv2.resize(sup_mask, (Gw,Gh), interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
        m_small2 = torch.from_numpy(m_small2)
        fg2 = (m_small2>0.5).float(); bg2=(m_small2<=0.5).float()
        proto_fg_dn = (dn_r * fg2.unsqueeze(0)).sum((1,2)) / (fg2.sum()+1e-8)
        proto_bg_dn = (dn_r * bg2.unsqueeze(0)).sum((1,2)) / (bg2.sum()+1e-8)
        proto_fg_dn = F.normalize(proto_fg_dn, dim=0)
        proto_bg_dn = F.normalize(proto_bg_dn, dim=0)

        # Upsample DINO similarity for explainability inputs (optional, usable as extra_maps)
        # Edges as hint
        Ht,Wt = sam_t.shape[-2:]  # 64x64 for SAM, we'll down later
        edge = sobel_edge_hint(qry_img, (Ht,Wt))  # [Ht,Wt]
        S32 = support_mask_to_grid(sup_mask, cache_long=self.cache_long, gridH=16, gridW=16)
        S_up = F.interpolate(S32.unsqueeze(0).unsqueeze(0), size=(Ht,Wt), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        extra = torch.stack([edge, S_up], dim=0).unsqueeze(0)   # [1,2,Ht,Wt] (will be resized inside model)

        sample = {
            'sam_t': sam_t.unsqueeze(0),    # [1,256,64,64]
            'dn_t': dn_t.unsqueeze(0),                   # [768,32,32]
            'proto_fg_sam': proto_fg_sam.unsqueeze(0),
            'proto_bg_sam': proto_bg_sam.unsqueeze(0),
            'proto_fg_dn':  proto_fg_dn.unsqueeze(0),
            'proto_bg_dn':  proto_bg_dn.unsqueeze(0),
            'tgt_rgb': qry_img,             # HWC uint8
            'tgt_mask': qry_mask,           # HW  uint8
            'tgt_path': qry_path,
            'extra_maps': extra             # [1,2,64,64]
        }
        return sample

