# train_rl.py
# ---- SAM2 快取 I/O 猴子補丁：讓任何版本都有 set_image_from_cache / export_cache ----
def _monkey_patch_sam2_cache_io():
    import functools, types, torch
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except Exception:
        # 你的匯入別名若不同，改這行
        from sam2_image_predictor import SAM2ImagePredictor

    if hasattr(SAM2ImagePredictor, "set_image_from_cache") and hasattr(SAM2ImagePredictor, "export_cache"):
        return  # 已有就不重覆貼

    def _as_hw_tuple(val):
        import numpy as np
        if val is None: return None
        if isinstance(val, int): return (int(val), int(val))
        if isinstance(val, (tuple, list)):
            if len(val) >= 2: return (int(val[-2]), int(val[-1]))
            if len(val) == 1:  return (int(val[0]), int(val[0]))
            return None
        if isinstance(val, torch.Tensor):
            v = val.detach().cpu().flatten().tolist()
            if len(v) >= 2: return (int(v[-2]), int(v[-1]))
            if len(v) == 1:  return (int(v[0]), int(v[0]))
            return None
        try:
            import numpy as np
            if isinstance(val, np.ndarray):
                v = val.flatten().tolist()
                if len(v) >= 2: return (int(v[-2]), int(v[-1]))
                if len(v) == 1:  return (int(v[0]), int(v[0]))
        except Exception:
            pass
        return None

    @torch.no_grad()
    def set_image_from_cache(self, cache: dict) -> None:
        assert isinstance(cache, dict), "cache must be a dict"

        # 1) 取 embedding 與 pyramid
        img_key = None
        for k in ("image_embed", "image_embeddings", "image_features", "vision_feats"):
            if k in cache: img_key = k; break
        if img_key is None:
            raise KeyError("cache missing image embedding (image_embed / image_embeddings / ...)")

        image_embed = cache[img_key]
        high_res = cache.get("high_res_feats", None)
        if high_res is None:
            raise KeyError("cache missing 'high_res_feats' (list of tensors). Please rebuild cache with export_cache().")

        dev = self.device
        to_dev_fp32 = lambda x: (x if isinstance(x, torch.Tensor) else torch.as_tensor(x)).to(dev).to(torch.float32)

        # squeeze 可能的 batch 維
        if isinstance(image_embed, list):
            image_embed = image_embed[0]
        if isinstance(image_embed, torch.Tensor) and image_embed.dim() == 5 and image_embed.size(0) == 1:
            image_embed = image_embed[0]
        image_embed = to_dev_fp32(image_embed)

        assert isinstance(high_res, (list, tuple)) and len(high_res) > 0, "high_res_feats must be non-empty list"
        high_res = [to_dev_fp32(t) for t in high_res]

        # 2) 尺寸 → (H, W)
        orig_hw = (_as_hw_tuple(cache.get("original_size"))
                   or _as_hw_tuple(cache.get("orig_size"))
                   or _as_hw_tuple(cache.get("original_size_hw")))
        if orig_hw is None:
            s = int(getattr(self.model, "image_size", 1024)); orig_hw = (s, s)

        # 3) 寫回 predictor 狀態
        self._features = {"image_embed": image_embed, "high_res_feats": high_res}
        self._orig_hw = [tuple(orig_hw)]
        self._is_image_set = True
        self._is_batch = False

        # 4) 補丁 transforms（支援 normalize 等 kwargs 與 int → (H,W)）
        tr = getattr(self, "_transforms", None)
        if tr is not None and hasattr(tr, "transform_coords") and not getattr(tr, "_patched_accepts_kwargs", False):
            old = tr.transform_coords
            @functools.wraps(old)
            def wrapped(coords, *args, **kwargs):
                if 'orig_hw' in kwargs:
                    ohw = _as_hw_tuple(kwargs['orig_hw'])
                    if ohw is not None: kwargs['orig_hw'] = ohw
                elif len(args) >= 1:
                    ohw = _as_hw_tuple(args[0])
                    if ohw is not None: args = (ohw,) + args[1:]
                return old(coords, *args, **kwargs)
            tr.transform_coords = wrapped
            tr._patched_accepts_kwargs = True

    @torch.no_grad()
    def export_cache(self) -> dict:
        if not getattr(self, "_is_image_set", False) or self._features is None:
            raise RuntimeError("Call set_image(...) before export_cache().")
        def to_cpu_fp16(x: torch.Tensor): return x.detach().to("cpu").contiguous().to(torch.float16)

        # image_embed / high_res_feats
        img_key = "image_embed" if "image_embed" in self._features else \
                  ("image_embeddings" if "image_embeddings" in self._features else None)
        if img_key is None:
            raise KeyError("predictor._features lacks image embedding key")
        pack = {
            "image_embed": to_cpu_fp16(self._features[img_key]),
        }
        hrs = self._features.get("high_res_feats", None)
        assert isinstance(hrs, list) and len(hrs) > 0, "high_res_feats missing in predictor._features"
        pack["high_res_feats"] = [to_cpu_fp16(t) for t in hrs]

        # sizes
        if isinstance(getattr(self, "_orig_hw", None), list) and len(self._orig_hw) > 0:
            ohw = _as_hw_tuple(self._orig_hw[0])
        else:
            s = int(getattr(self.model, "image_size", 1024)); ohw = (s, s)
        pack["original_size"] = tuple(ohw)
        pack["orig_size"]     = tuple(ohw)
        pack["input_size"]    = tuple(ohw)

        # optional padded size
        pinp = getattr(self._transforms, "padded_input_image_size", None)
        pinp = _as_hw_tuple(pinp) or ohw
        pack["padded_input_image_size"] = tuple(pinp)

        # meta（你要記錄的欄位）
        pack["meta"] = {
            "sizes_saved_as_tuple": True,
            "original_size_hw": tuple(ohw),
            "input_size_hw": tuple(ohw),
            "has_high_res_feats": True,
            "high_res_feats_levels": len(pack["high_res_feats"]),
        }
        return pack

    # 掛到類別上
    SAM2ImagePredictor.set_image_from_cache = set_image_from_cache
    SAM2ImagePredictor.export_cache = export_cache

_monkey_patch_sam2_cache_io()
import os, math, csv, json, time, random, hashlib
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import cv2
from PIL import Image
from torch.distributions import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate

from transformers import AutoImageProcessor, AutoModel  # HF DINOv3
from tqdm import tqdm
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import functools

from dataloader_ps import build_psv2_index as build_coco20i_index
from dataloader_ps import OneShotPSV2Random as OneShotCOCO20iRandom
from dataloader_ps import OneShotPSV2RoundRobin as OneShotCOCO20iRoundRobin

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------ I/O utils ------------------------------

def load_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path)
    assert bgr is not None, f"Image not found: {path}"
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def load_gray(path: str) -> np.ndarray:
    g = cv2.imread(path, 0)
    assert g is not None, f"Mask not found: {path}"
    return g

def norm01_t(x: torch.Tensor) -> torch.Tensor:
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-6)

class AutocastCtx:
    def __init__(self, enabled: bool): self.enabled = enabled
    def __enter__(self):
        return torch.autocast("cuda", dtype=torch.bfloat16) if (self.enabled and device.type=="cuda") else torch.cuda.amp.autocast(enabled=False)
    def __exit__(self, exc_type, exc, tb):
        return False

def cache_key(img_path: str) -> str:
    p = os.path.abspath(img_path)
    h = hashlib.sha1(p.encode('utf-8')).hexdigest()[:16]
    stem = os.path.splitext(os.path.basename(img_path))[0]
    return f"{stem}__{h}"

def scan_images(dirs, exts, recursive=True):
    exts = {e.lower().strip() for e in exts}
    out = []
    for root in dirs:
        root = os.path.abspath(root)
        if recursive:
            for r, _, files in os.walk(root):
                for f in files:
                    if os.path.splitext(f)[1].lower() in exts:
                        out.append(os.path.join(r, f))
        else:
            for f in os.listdir(root):
                p = os.path.join(root, f)
                if os.path.isfile(p) and os.path.splitext(f)[1].lower() in exts:
                    out.append(p)
    return sorted(set(out))

def save_checkpoint(path, model, opt, sch, epoch, best_iou, lc):
    state = {
        'model': model.state_dict(),
        'optimizer': opt.state_dict(),
        'scheduler': sch.state_dict(),
        'epoch': int(epoch),
        'best_iou': float(best_iou),
        'lc': float(lc)
    }
    torch.save(state, path)

def collate_keep_rgb_mask_as_list(batch):
    keys_stack = [
        'sam_t','dn_t',
        'proto_fg_sam','proto_bg_sam','proto_fg_dn','proto_bg_dn',
        'x_in',
        'H_fg_star','H_bg_star',
        'K_fg_star','K_bg_star','rgb_in'
    ]
    batch_core = [{k:v for k,v in b.items() if k in keys_stack} for b in batch]
    out = default_collate(batch_core)
    if 'tgt_rgb' in batch[0]:
        out['tgt_rgb']  = [b['tgt_rgb']  for b in batch]
    if 'tgt_mask' in batch[0]:
        out['tgt_mask'] = [b['tgt_mask'] for b in batch]
    if 'tgt_path' in batch[0]:
        out['tgt_path'] = [b['tgt_path'] for b in batch]  # 用於從快取取回 predictor 特徵
    return out

def _to_bchw(t, name="tensor"):
    if t is None:
        raise ValueError(f"{name} is None")
    while t.dim() >= 5 and t.shape[0] == 1:
        t = t.squeeze(0)
    if t.dim() == 3:
        t = t.unsqueeze(0)
    elif t.dim() == 4:
        pass
    elif t.dim() == 5:
        B, D1, D2, H, W = t.shape
        t = t.reshape(B, D1 * D2, H, W)
    else:
        raise ValueError(f"{name}: unexpected ndim={t.dim()}, shape={tuple(t.shape)}")
    return t

def _force_size_attrs(predictor, hw_tuple):
    """把 SAM2 predictor 需要的尺寸別名一口氣補齊成 (H,W)。"""
    H, W = int(hw_tuple[0]), int(hw_tuple[1])
    hw = (H, W)
    # list 形式（許多分支實際讀這個）
    try:
        predictor._orig_hw = [hw]
    except Exception:
        pass
    # 常見別名
    for name in ("original_size", "orig_size", "_original_size"):
        try: setattr(predictor, name, hw)
        except Exception: pass
    for name in ("input_size", "_input_size", "padded_input_image_size"):
        try: setattr(predictor, name, hw)
        except Exception: pass

# ---- Letterbox helpers ----
def resize_letterbox_rgb(rgb: np.ndarray, fixed_size: int = 1024) -> tuple[np.ndarray, dict]:
    H, W = rgb.shape[:2]
    if max(H, W) == 0:
        raise ValueError("Invalid image size")
    interp = cv2.INTER_AREA if (H >= fixed_size and W >= fixed_size) else cv2.INTER_CUBIC
    out = cv2.resize(rgb, (fixed_size, fixed_size), interpolation=interp)
    meta = dict(top=0, left=0, out_h=fixed_size, out_w=fixed_size,
                scale=1.0, orig_h=H, orig_w=W)
    return out, meta

def resize_letterbox_mask(mask: np.ndarray, fixed_size: int = 1024, k=None) -> np.ndarray:
    out = cv2.resize(mask, (fixed_size, fixed_size), interpolation=cv2.INTER_NEAREST)
    return out

# ----------------------- DINOv3 HF features (grid) ---------------------
_DINOV3_CACHE = {}
def _get_dinov3_hf(model_id: str):
    if model_id not in _DINOV3_CACHE:
        proc = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device)
        model.eval()
        _DINOV3_CACHE[model_id] = (proc, model)
    return _DINOV3_CACHE[model_id]

@torch.inference_mode()
def get_grid_feats_dinov3_hf(image_rgb: np.ndarray, model_id: str) -> torch.Tensor:
    proc, model = _get_dinov3_hf(model_id)
    pil = Image.fromarray(image_rgb)
    inputs = proc(images=pil, return_tensors="pt", do_center_crop=False, do_resize=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs)
    x = out.last_hidden_state
    assert x.dim() in (3,4)
    if x.dim()==4:
        grid = x[0].permute(1,2,0).contiguous()
        return F.normalize(grid, dim=-1).to(torch.float32)
    Hp, Wp = inputs["pixel_values"].shape[-2:]
    psize = int(getattr(getattr(model, "config", None), "patch_size", 16))
    Gh, Gw = Hp//psize, Wp//psize
    M = Gh*Gw
    toks = x[0, -M:, :]
    grid = toks.view(Gh, Gw, -1).contiguous()
    return F.normalize(grid, dim=-1).to(torch.float32)

# ------------------------------ SAM2 predictor / cache -----------------

def sam2_build_image_predictor(cfg_yaml: str, ckpt_path: str):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model = build_sam2(cfg_yaml, ckpt_path).to(device)
    return SAM2ImagePredictor(model)

def _get_predictor_features_dict(predictor) -> dict:
    """
    從 SAM2ImagePredictor 把 set_image 後的特徵抽出成 CPU/FP16 字典，
    並把尺寸鍵一律正規化為 (H, W) tuple。
    """
    feats = None
    for attr in ("_features", "features"):
        if hasattr(predictor, attr):
            d = getattr(predictor, attr)
            if isinstance(d, dict):
                feats = d; break
    if feats is None:
        raise RuntimeError("SAM2 predictor has no feature dict (_features/features)")

    want_keys = {
        "image_embeddings", "image_embed", "image_features", "vision_feats",
        "high_res_feats",   # ★ 新增
        "image_pe", "positional_encoding",
        "input_size", "original_size", "orig_size"
    }

    def _as_hw_tuple(val, fallback=None):
        import numpy as np, torch as _torch
        if val is None: return fallback
        if isinstance(val, int): return (int(val), int(val))
        if isinstance(val, (tuple, list)):
            if len(val) >= 2: return (int(val[-2]), int(val[-1]))
            if len(val) == 1:  return (int(val[0]), int(val[0]))
            return fallback
        if isinstance(val, _torch.Tensor):
            v = val.detach().cpu().flatten().tolist()
            if len(v) >= 2: return (int(v[-2]), int(v[-1]))
            if len(v) == 1:  return (int(v[0]), int(v[0]))
            return fallback
        try:
            import numpy as np
            if isinstance(val, np.ndarray):
                v = val.flatten().tolist()
                if len(v) >= 2: return (int(v[-2]), int(v[-1]))
                if len(v) == 1:  return (int(v[0]), int(v[0]))
        except Exception:
            pass
        return fallback

    out = {}
    for k, v in feats.items():
        if k not in want_keys:
            continue
        if k == "high_res_feats":
            # 保持 pyramid list 形式
            out[k] = [t.detach().to("cpu").contiguous().to(torch.float16) for t in v]
            continue
        # 其他鍵照舊
        if isinstance(v, (list, tuple)) and len(v) > 0 and hasattr(v[0], "shape"):
            v = v[0]
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().to("cpu").contiguous().to(torch.float16)
        else:
            out[k] = v

    # 尺寸鍵 → (H, W)
    orig_hw = _as_hw_tuple(out.get("original_size")) or _as_hw_tuple(out.get("orig_size"))
    inp_hw  = _as_hw_tuple(out.get("input_size"), fallback=orig_hw)
    if orig_hw is not None:
        out["original_size"] = tuple(orig_hw)
        out["orig_size"]     = tuple(orig_hw)
    if inp_hw is not None:
        out["input_size"]    = tuple(inp_hw)
    return out



def _sam2_embed_from_cached(feats: dict) -> torch.Tensor:
    """
    從快取字典取出影像 embedding，回傳 [C,Hf,Wf]、float32、L2-normalized
    """
    cand = ["image_embeddings", "image_embed", "image_features", "vision_feats"]
    x = None
    for k in cand:
        if k in feats and isinstance(feats[k], torch.Tensor):
            x = feats[k]; break
    if x is None:
        raise KeyError(f"No image embedding found in cached feats; looked for {cand}")
    if x.dim() == 4 and x.size(0) == 1:
        x = x[0]
    elif x.dim() != 3:
        raise ValueError(f"Unexpected SAM2 embedding shape: {tuple(x.shape)}")
    x = x.to(torch.float32)
    x = F.normalize(x, dim=0)
    return x  # [C,Hf,Wf]

def _apply_cached_feats_to_predictor(predictor, feats_cached: dict, fallback_hw=None):
    """
    將快取回填到 predictor，並把所有尺寸欄位統一成 (H, W)。
    同步覆蓋 predictor._features / predictor.features 與物件屬性。
    """
    def _as_hw_tuple(val, fallback=None):
        import numpy as np
        if val is None: return fallback
        if isinstance(val, int): return (int(val), int(val))
        if isinstance(val, (tuple, list)):
            if len(val) >= 2: return (int(val[-2]), int(val[-1]))
            if len(val) == 1:  return (int(val[0]), int(val[0]))
            return fallback
        if isinstance(val, torch.Tensor):
            v = val.detach().cpu().flatten().tolist()
            if len(v) >= 2: return (int(v[-2]), int(v[-1]))
            if len(v) == 1:  return (int(v[0]), int(v[0]))
            return fallback
        try:
            import numpy as np
            if isinstance(val, np.ndarray):
                v = val.flatten().tolist()
                if len(v) >= 2: return (int(v[-2]), int(v[-1]))
                if len(v) == 1:  return (int(v[0]), int(v[0]))
        except Exception:
            pass
        return fallback

    dev = next(predictor.model.parameters()).device

    # 1) 搬到正確裝置/精度
    feats = {}
    for k, v in feats_cached.items():
        if isinstance(v, torch.Tensor):
            feats[k] = v.to(dev).to(torch.float32)
        else:
            feats[k] = v

    # 2) 尺寸一律 (H, W)
    orig_hw = (_as_hw_tuple(feats_cached.get("original_size"))
               or _as_hw_tuple(feats_cached.get("orig_size"))
               or _as_hw_tuple(feats.get("original_size"))
               or _as_hw_tuple(feats.get("orig_size"))
               or _as_hw_tuple(fallback_hw))
    inp_hw  = (_as_hw_tuple(feats_cached.get("input_size"))
               or _as_hw_tuple(feats.get("input_size"))
               or orig_hw)

    if orig_hw is not None:
        feats["original_size"] = tuple(orig_hw)
        feats["orig_size"]     = tuple(orig_hw)
    if inp_hw is not None:
        feats["input_size"]    = tuple(inp_hw)

    # 3) 回填 features 容器
    predictor._features = feats
    if hasattr(predictor, "features") and isinstance(getattr(predictor, "features"), dict):
        predictor.features = feats

    # 4) 同步屬性
    def _safe_set(obj, name, val):
        try: setattr(obj, name, tuple(val))
        except Exception: pass
    if orig_hw is not None:
        for name in ("original_size", "orig_size", "_original_size", "_orig_hw"):
            _safe_set(predictor, name, orig_hw)
    if inp_hw is not None:
        for name in ("input_size", "_input_size"):
            _safe_set(predictor, name, inp_hw)

    # 5) 標記為已 set image
    if hasattr(predictor, "_is_image_set"):
        predictor._is_image_set = True
    else:
        try: predictor.is_image_set = True
        except Exception: pass


def _patch_predictor_transforms(predictor):
    """
    對 predictor._transforms.transform_coords 做容錯包裝：
    - 接受 *args/**kwargs（包含 normalize= 等）
    - 將 orig_hw（無論是位置參數或關鍵字）正規化成 (H,W)
    """
    tr = getattr(predictor, "_transforms", None)
    if tr is None or not hasattr(tr, "transform_coords"):
        return
    if getattr(tr, "_patched_accepts_kwargs", False):
        return

    old = tr.transform_coords

    def _as_hw_tuple(val):
        import numpy as np, torch
        if val is None: return None
        if isinstance(val, int): return (int(val), int(val))
        if isinstance(val, (tuple, list)):
            if len(val) >= 2: return (int(val[-2]), int(val[-1]))
            if len(val) == 1:  return (int(val[0]), int(val[0]))
            return None
        if isinstance(val, torch.Tensor):
            v = val.detach().cpu().flatten().tolist()
            if len(v) >= 2: return (int(v[-2]), int(v[-1]))
            if len(v) == 1:  return (int(v[0]), int(v[0]))
            return None
        if 'numpy' in str(type(val)):
            v = np.array(val).flatten().tolist()
            if len(v) >= 2: return (int(v[-2]), int(v[-1]))
            if len(v) == 1:  return (int(v[0]), int(v[0]))
            return None
        return None

    @functools.wraps(old)
    def wrapped(coords, *args, **kwargs):
        # kw 形式
        if 'orig_hw' in kwargs:
            kwargs['orig_hw'] = _as_hw_tuple(kwargs['orig_hw']) or kwargs['orig_hw']
            return old(coords, *args, **kwargs)
        # 位置參數形式：預期第一個是 orig_hw
        if len(args) >= 1:
            ohw = _as_hw_tuple(args[0]) or args[0]
            args = (ohw,) + args[1:]
        return old(coords, *args, **kwargs)

    tr.transform_coords = wrapped
    tr._patched_accepts_kwargs = True


# ------------------------------ Prototypes & sims ----------------------
def compute_proto_and_sims_from_cache(ref_pt: str, ref_mask: np.ndarray,
                                      tgt_pt: str,
                                      cache_dir: str,
                                      use_bg_proto: bool=True,
                                      tau: float=0.2):
    """
    從 .pt 快取取 SAM2 / DINO，計算原型與相似度（不經過任何編碼器）
    回傳：dict 與先前 compute_proto_and_sims 相容
    """
    def _load_pt(pth):
        key = cache_key(pth)
        data = torch.load(os.path.join(cache_dir, f"{key}.pt"), map_location="cpu")
        sam2 = data["sam2"]; dng = data["dng"].to(torch.float32)  # [Cd,Gh,Gw]
        sam = _sam2_embed_from_cached(sam2)                       # [Cs,Hf,Wf]
        meta = data.get("meta", {})
        return sam, dng, meta, sam2

    sam_r, dn_r, meta_r, _ = _load_pt(ref_pt)
    sam_t, dn_t, meta_t, _ = _load_pt(tgt_pt)

    # SAM2 相似度
    C, Hf, Wf = sam_r.shape
    m_small = cv2.resize(ref_mask, (Wf, Hf), interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
    m_small = torch.from_numpy(m_small)
    fg = (m_small>0.5).float(); bg = (m_small<=0.5).float()
    proto_fg_sam = (sam_r * fg.unsqueeze(0)).sum((1,2)) / (fg.sum()+1e-8)
    proto_fg_sam = F.normalize(proto_fg_sam, dim=0)
    sim_fg_sam = (sam_t * proto_fg_sam.view(-1,1,1)).sum(0)
    if use_bg_proto:
        if bg.sum()<1:
            inv = (1.0-m_small).flatten(); topk = torch.topk(inv, k=min(10, inv.numel()))[1]
            bg = torch.zeros_like(inv); bg[topk]=1.0; bg = bg.view(Hf,Wf)
        proto_bg_sam = (sam_r * bg.unsqueeze(0)).sum((1,2)) / (bg.sum()+1e-8)
        proto_bg_sam = F.normalize(proto_bg_sam, dim=0)
        sim_sam = torch.sigmoid((sim_fg_sam - (sam_t * proto_bg_sam.view(-1,1,1)).sum(0))/tau)
    else:
        sim_sam = (sim_fg_sam+1.0)/2.0

    # DINO 相似度
    Cd, Gh, Gw = dn_r.shape
    m_small2 = cv2.resize(ref_mask, (Gw, Gh), interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
    m_small2 = torch.from_numpy(m_small2)
    fg2 = (m_small2>0.5).float(); bg2 = (m_small2<=0.5).float()
    proto_fg_dn = (dn_r * fg2.unsqueeze(0)).sum((1,2))/(fg2.sum()+1e-8)
    proto_fg_dn = F.normalize(proto_fg_dn, dim=0)
    sim_fg_dn = (dn_t * proto_fg_dn.view(-1,1,1)).sum(0)
    if use_bg_proto:
        proto_bg_dn = (dn_r * bg2.unsqueeze(0)).sum((1,2))/(bg2.sum()+1e-8)
        proto_bg_dn = F.normalize(proto_bg_dn, dim=0)
        sim_dn = torch.sigmoid((sim_fg_dn - (dn_t * proto_bg_dn.view(-1,1,1)).sum(0))/tau)
        
    else:
        sim_dn = (sim_fg_dn+1.0)/2.0

    return {
        "sam_feat_ref": sam_r, "sam_feat_tgt": sam_t,
        "dino_feat_ref": dn_r.permute(1,2,0), "dino_feat_tgt": dn_t.permute(1,2,0),
        "proto_fg_sam": proto_fg_sam, "proto_bg_sam": proto_bg_sam if use_bg_proto else None,
        "proto_fg_dn": proto_fg_dn, "proto_bg_dn": proto_bg_dn if use_bg_proto else None,
        "sim_sam": sim_sam, "sim_dino": sim_dn
    }

# ------------------------------ Input assembly ------------------------
def sobel_edge_hint(image_rgb: np.ndarray, size_hw: Tuple[int,int]) -> torch.Tensor:
    Ht, Wt = size_hw
    g = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy)
    mag = (mag - mag.min()) / (mag.max()-mag.min()+1e-6)
    mag = cv2.resize(mag.astype(np.float32), (Wt, Ht), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(mag).to(torch.float32)

# ------------------------------ PointPromptNet-B ----------------------
class DSConv(nn.Module):
    def __init__(self, c, act=True):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.GELU() if act else nn.Identity()
    def forward(self, x):
        x = self.dw(x); x = self.pw(x); x = self.bn(x); return self.act(x)

class Block(nn.Module):
    def __init__(self, c):
        super().__init__(); self.c1=DSConv(c); self.c2=DSConv(c)
    def forward(self, x):
        return self.c2(self.c1(x)) + x

class Down(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.GELU(),
            nn.Conv2d(cout, cout, 3, 2, 1, groups=cout, bias=False),
            nn.Conv2d(cout, cout, 1, 1, 0, bias=False), nn.BatchNorm2d(cout), nn.GELU()
        )
    def forward(self, x): return self.proj(x)

class ProtoCrossAttn(nn.Module):
    def __init__(self, c=192, heads=4, mlp_ratio=2.0):
        super().__init__()
        self.c, self.h = c, heads
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.o = nn.Linear(c, c, bias=False)
        self.mlp = nn.Sequential(nn.Linear(c, int(c*mlp_ratio)), nn.GELU(), nn.Linear(int(c*mlp_ratio), c))
        self.norm_q = nn.LayerNorm(c); self.norm_x = nn.LayerNorm(c)
    def forward(self, x, proto_tokens):
        B,C,H,W = x.shape
        xs = x.permute(0,2,3,1).reshape(B, H*W, C)
        q = self.q(self.norm_q(proto_tokens))
        k = self.k(self.norm_x(xs)); v = self.v(self.norm_x(xs))
        def split(t): return t.view(B, -1, self.h, C//self.h).transpose(1,2)
        qh, kh, vh = split(q), split(k), split(v)
        attn = (qh @ kh.transpose(-2,-1)) / (C//self.h)**0.5
        attn = attn.softmax(dim=-1)
        out = (attn @ vh).transpose(1,2).reshape(B, 2, C)
        out = self.o(out) + proto_tokens
        out = out + self.mlp(out)
        return x, out


class SE(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        m = max(1, ch // r)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, m, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(m, ch, 1, bias=True),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.fc(x)

class ResidualBlock(nn.Module):
    """
    輕量殘差：可選 depthwise 可分離卷積 + SE
    保持空間大小不變；通道數維持 ch。
    """
    def __init__(self, ch, use_se=True, dilation=1):
        super().__init__()
        padding = dilation

        self.conv1 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=padding, dilation=dilation, groups=ch, bias=False),
            nn.BatchNorm2d(ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.SiLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=padding, dilation=dilation, groups=ch, bias=False),
            nn.BatchNorm2d(ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
        )

        self.se = SE(ch) if use_se else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.se(h)
        return self.act(x + h)


class PointPromptNetB(nn.Module):
    def __init__(self, c_in=100, c0=192, c1=256, c2=384, 
                 kmax_f=8, kmax_b=8,
                 sam_c=256, dino_c=768, proj_dim=48,rgb_proj_ch=16):
        super().__init__()
        self.proj_sam = nn.Conv2d(sam_c, proj_dim, 1, bias=False)
        self.proj_dn  = nn.Conv2d(dino_c, proj_dim, 1, bias=False)
        
        # 新增：RGB 輕量分支（3→16）
        rgb_proj_ch = getattr(self, "rgb_proj_ch", 16)  # 若你有放到 __init__ 參數就直接用那個
        self.rgb_stem = nn.Sequential(
            nn.Conv2d(3, rgb_proj_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(rgb_proj_ch),
            nn.SiLU(inplace=True),
            # 深度：重複幾個殘差塊（可調整 blocks、dilation）
            ResidualBlock(rgb_proj_ch, use_se=True, dilation=1),
            ResidualBlock(rgb_proj_ch, use_se=True, dilation=1),
            ResidualBlock(rgb_proj_ch, use_se=True, dilation=2),  # 帶一點擴張看更大感受野
        )
        
        self.proto_lin = nn.Linear(proj_dim*2, c0)
        self.in_proj = nn.Sequential(
                    nn.Conv2d(c_in + rgb_proj_ch, c0, 1, bias=False),
                    nn.BatchNorm2d(c0),
                    nn.GELU()
                )
        self.enc0 = nn.Sequential(Block(c0), Block(c0))
        self.down1 = Down(c0, c1); self.enc1 = nn.Sequential(Block(c1), Block(c1))
        self.down2 = Down(c1, c2); self.enc2 = nn.Sequential(Block(c2), Block(c2))
        self.bot  = Block(c2)
        self.pxattn = ProtoCrossAttn(c=c0, heads=4, mlp_ratio=2.0)
        self.up1  = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                  nn.Conv2d(c2, c1, 1, bias=False), nn.BatchNorm2d(c1), nn.GELU(), Block(c1))
        self.up0  = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                  nn.Conv2d(c1, c0, 1, bias=False), nn.BatchNorm2d(c0), nn.GELU(), Block(c0))
        # heat heads
        self.fg_head = nn.Conv2d(c0, 1, 1)
        self.bg_head = nn.Conv2d(c0, 1, 1)
        # 16×16 grid 三分類 head（0=pos,1=neu,2=neg）
        self.grid_head = nn.Conv2d(c0, 3, 1)

        self.kmax_f = kmax_f
        self.kmax_b = kmax_b

        for m in [self.proj_sam, self.proj_dn]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
        nn.init.xavier_uniform_(self.proto_lin.weight)

    def map_proto(self, proto_fg_sam, proto_bg_sam, proto_fg_dn, proto_bg_dn):
        Ws = self.proj_sam.weight.view(self.proj_sam.out_channels, -1)
        Wd = self.proj_dn.weight.view(self.proj_dn.out_channels, -1)
        fg_s = (Ws @ proto_fg_sam.unsqueeze(-1)).squeeze(-1)
        bg_s = (Ws @ proto_bg_sam.unsqueeze(-1)).squeeze(-1)
        fg_d = (Wd @ proto_fg_dn.unsqueeze(-1)).squeeze(-1)
        bg_d = (Wd @ proto_bg_dn.unsqueeze(-1)).squeeze(-1)
        fg = torch.cat([fg_s, fg_d], dim=-1)
        bg = torch.cat([bg_s, bg_d], dim=-1)
        fg = self.proto_lin(fg); bg = self.proto_lin(bg)
        return fg, bg

    def forward(self, x_in, sam_feat_tgt, dino_feat_tgt,
                proto_fg_sam, proto_bg_sam, proto_fg_dn, proto_bg_dn, rgb_in=None):
    
        x_in  = _to_bchw(x_in, "x_in")
    
        # --- SAM 特徵整形 ---
        if sam_feat_tgt.dim() == 3:
            sam_in = sam_feat_tgt.unsqueeze(0)
        elif sam_feat_tgt.dim() == 4:
            sam_in = sam_feat_tgt
        else:
            raise ValueError(f"sam_feat_tgt ndim={sam_feat_tgt.dim()}")
    
        # --- DINO 特徵整形 ---
        if dino_feat_tgt.dim() == 3:
            if dino_feat_tgt.shape[-1] in (256, 384, 768):
                dn_in = dino_feat_tgt.permute(2,0,1).unsqueeze(0)
            elif dino_feat_tgt.shape[0] in (256, 384, 768):
                dn_in = dino_feat_tgt.unsqueeze(0)
            else:
                raise ValueError(f"dino_feat_tgt shape not recognized: {tuple(dino_feat_tgt.shape)}")
        elif dino_feat_tgt.dim() == 4:
            if dino_feat_tgt.shape[1] in (256, 384, 768):
                dn_in = dino_feat_tgt
            elif dino_feat_tgt.shape[-1] in (256, 384, 768):
                dn_in = dino_feat_tgt.permute(0,3,1,2)
            else:
                raise ValueError(f"dino_feat_tgt shape not recognized: {tuple(dino_feat_tgt.shape)}")
        else:
            raise ValueError(f"dino_feat_tgt ndim={dino_feat_tgt.dim()}")
    
        # --- 投影到共同通道 ---
        sam_p = self.proj_sam(sam_in)
        dn_p  = self.proj_dn(dn_in)
    
        H, W = x_in.shape[-2], x_in.shape[-1]
        if sam_p.shape[-2:] != (H, W):
            sam_p = F.interpolate(sam_p, size=(H, W), mode='bilinear', align_corners=False)
        if dn_p.shape[-2:] != (H, W):
            dn_p  = F.interpolate(dn_p,  size=(H, W), mode='bilinear', align_corners=False)
        x_in = _to_bchw(x_in, "x_in(BCHW-check)")
    
        # ★ RGB 分支（可選）：對齊→輕量投影→與其他特徵 concat
        if rgb_in is not None:
            rgb_in = _to_bchw(rgb_in, "rgb_in")  # 支援 [3,H,W] 或 [B,3,H,W]
            if rgb_in.shape[-2:] != (H, W):
                rgb_in = F.interpolate(rgb_in, size=(H, W), mode='bilinear', align_corners=False)
            rgb_feat = self.rgb_stem(rgb_in)      # [B,16,H,W]
            x_cat = torch.cat([sam_p, dn_p, x_in, rgb_feat], dim=1)
        else:
            x_cat = torch.cat([sam_p, dn_p, x_in], dim=1)
    
        # --- 後續與原來一致 ---
        x = self.in_proj(x_cat)
        s0 = self.enc0(x)
        s1 = self.enc1(self.down1(s0))
        s2 = self.enc2(self.down2(s1))
        z  = self.bot(s2)
        u1 = self.up1(z) + s1
        u0 = self.up0(u1) + s0
    
        fg_tok, bg_tok = self.map_proto(proto_fg_sam, proto_bg_sam, proto_fg_dn, proto_bg_dn)
        protos = torch.stack([fg_tok, bg_tok], dim=1)
        u0, _ = self.pxattn(u0, protos)
    
        H_fg = self.fg_head(u0)        # logits
        H_bg = self.bg_head(u0)        # logits
        G    = self.grid_head(u0)      # [B,3,Hf,Wf] → 下到 16×16
        G    = F.interpolate(G, size=(16,16), mode='bilinear', align_corners=False)
        return H_fg, H_bg, G

# ------------------------------ GT builders ---------------------------
def distance_transform_heatmap(mask: np.ndarray, size_hw: Tuple[int,int]) -> torch.Tensor:
    Ht, Wt = size_hw
    m = (cv2.resize(mask, (Wt,Ht), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
    if m.max()==0:
        return torch.zeros(Ht, Wt, dtype=torch.float32)
    dist = cv2.distanceTransform(m, distanceType=cv2.DIST_L2, maskSize=3)
    dist = dist / (dist.max()+1e-6)
    dist = cv2.GaussianBlur(dist, (0,0), sigmaX=1.0)
    return torch.from_numpy(dist.astype(np.float32))

def outer_ring(mask: np.ndarray, size_hw: Tuple[int,int], r: int=3) -> torch.Tensor:
    Ht, Wt = size_hw
    m = (cv2.resize(mask, (Wt,Ht), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
    dil = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1)))
    ring = (dil & (~m)).astype(np.float32)
    return torch.from_numpy(ring)

def bg_distance_map(mask: np.ndarray, size_hw: Tuple[int,int]) -> np.ndarray:
    """背景到物體邊界的歐氏距離（像素，於 letterbox 尺寸）。"""
    Ht, Wt = size_hw
    m = (cv2.resize(mask, (Wt, Ht), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
    bg = (m == 0).astype(np.uint8)
    dist = cv2.distanceTransform(bg, distanceType=cv2.DIST_L2, maskSize=3)
    return dist  # float32, 單位：px

def make_band_gate(mask: np.ndarray, size_hw: Tuple[int,int], dmin_px: int, dmax_px: int) -> torch.Tensor:
    """僅保留 [dmin, dmax] 之間的背景區域；回傳 torch.float32 0/1。"""
    dist = bg_distance_map(mask, size_hw)
    band = ((dist >= float(dmin_px)) & (dist <= float(dmax_px))).astype(np.float32)
    return torch.from_numpy(band)


def count_targets(mask: np.ndarray) -> int:
    m = (mask>127).astype(np.uint8)
    n, labels = cv2.connectedComponents(m, connectivity=8)
    return max(1, n-1)

def build_gt(target_mask: np.ndarray,
             sim_fg_outside: torch.Tensor,
             size_hw: Tuple[int,int], kmax_f:int=8, kmax_b:int =8,
             alpha:float=0.5, beta:float=0.5,
             ring_r:int = 3,
             band_dmin_px:int = -1, band_dmax_px:int = 999999):
    device = sim_fg_outside.device
    Ht, Wt = size_hw

    # 前景 soft heat
    H_fg_star = distance_transform_heatmap(target_mask, size_hw).to(device)

    # 外圈 & outside
    ring    = outer_ring(target_mask, size_hw, r=int(ring_r)).to(device)           # [Ht,Wt] 0/1
    outside = torch.clamp(sim_fg_outside, 0, 1)                                    # [Ht,Wt] 0..1

    # 帶狀 band gating（只保留 [dmin,dmax]）
    if int(band_dmin_px) >= 0:
        band = make_band_gate(target_mask, size_hw, int(band_dmin_px), int(band_dmax_px)).to(device)
        ring    = ring * band
        outside = outside * band

    H_bg_star = torch.clamp(alpha*ring + beta*outside, 0, 1)

    # k*（維持原本邏輯）
    m_small = (cv2.resize(target_mask, (Wt, Ht), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
    n, _ = cv2.connectedComponents(m_small, connectivity=8)
    K_fg_star = min(max(1, n-1), kmax_f)
    cnts = cv2.findContours(m_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    perim = sum(cv2.arcLength(c, True) for c in cnts) if cnts else 0.0
    K_bg_star = int(np.clip(round(perim/200.0), 0, kmax_b))
    return H_fg_star, H_bg_star, K_fg_star, K_bg_star

# === Support mask → 16x16 occupancy grid (0..1) ===
@torch.inference_mode()
def support_mask_to_grid(mask_np: np.ndarray,
                         cache_long: int,
                         gridH: int = 16,
                         gridW: int = 16) -> torch.Tensor:
    """
    輸入:
      - mask_np: 支援 mask，numpy HW (0..255)
      - cache_long: letterbox 尺寸（需與快取一致）
    回傳:
      - [gridH, gridW] 的 torch.float32 tensor（每格為前景佔比 0..1）
    """
    # 1) letterbox 到 cache_long
    m_lb = resize_letterbox_mask(mask_np, fixed_size=cache_long)  # [H,W] uint8(0..255)
    m_bin = (m_lb > 127).astype(np.float32)                       # [H,W] {0,1}

    # 2) 做一次整數平均池化 → 16x16 佔比
    kh = cache_long // gridH
    kw = cache_long // gridW
    assert cache_long % gridH == 0 and cache_long % gridW == 0, \
        f"cache_long={cache_long} 必須能被 gridH={gridH}, gridW={gridW} 整除"

    x = torch.from_numpy(m_bin)[None, None]  # [1,1,H,W]
    g = F.avg_pool2d(x, kernel_size=(kh, kw), stride=(kh, kw))[0, 0]  # [gridH, gridW] 0..1
    return g.to(torch.float32)


# ------------------------------ Point picking -------------------------
def nms_peaks(H: torch.Tensor, K: int, radius: int=2, thresh: float=0.1) -> List[Tuple[int,int]]:
    Hc = H.clone(); Hc[Hc<thresh] = 0.0
    pts=[]; Hf, Wf = Hc.shape
    for _ in range(int(K)):
        idx = torch.argmax(Hc)
        v = float(Hc.view(-1)[idx])
        if v<=0: break
        r = int(idx // Wf); c = int(idx % Wf)
        pts.append((r,c))
        r0,r1=max(0,r-radius),min(Hf,r+radius+1)
        c0,c1=max(0,c-radius),min(Wf,c+radius+1)
        Hc[r0:r1,c0:c1]=0.0
    return pts

def grid_to_xy(rc: Tuple[int,int], Hgrid:int, Wgrid:int, H:int, W:int):
    r,c = rc
    x = int((c+0.5)*W/Wgrid)
    y = int((r+0.5)*H/Hgrid)
    return [x,y]

# ------------------------------ Dataset & cache -----------------------
@dataclass
class Episode:
    target: str
    target_mask: str
    ref: str
    ref_mask: str
    split: str

def sam2_build_and_extract_dict(cfg_yaml: str, ckpt_path: str, rgb_lb: np.ndarray) -> dict:
    """一次性建 predictor → set_image → 直接導出完整快取包（for cache）"""
    predictor = sam2_build_image_predictor(cfg_yaml, ckpt_path)
    with AutocastCtx(True):
        predictor.set_image(rgb_lb)
    return predictor.export_cache()  # ← 改這裡


class EpisodeDataset(Dataset):
    def __init__(self, episodes: List[Episode], cache_dir: str, dinov3_id: str,
                 sam2_cfg: str, sam2_ckpt: str, build_cache: bool=False, use_autocast: bool=True,
                 cache_long: int = 1024, cache_multiple: int = 16):
        self.episodes = [e for e in episodes]
        self.cache_dir = cache_dir
        self.dinov3_id = dinov3_id
        self.use_autocast = use_autocast
        os.makedirs(cache_dir, exist_ok=True)
        self.predictor = sam2_build_image_predictor(sam2_cfg, sam2_ckpt)
        self.sam2_cfg = sam2_cfg
        self.sam2_ckpt = sam2_ckpt
        self.cache_long = int(cache_long)
        self.cache_multiple = int(cache_multiple)
        if build_cache:
            self._build_all()

    def _key(self, img_path: str) -> str:
        return cache_key(img_path)

    def _build_one(self, target: str):
        key = self._key(target)
        out = os.path.join(self.cache_dir, f"{key}.pt")
        if os.path.exists(out):
            return
        rgb_raw = load_rgb(target)
        rgb_lb, meta = resize_letterbox_rgb(rgb_raw, self.cache_long)

        # SAM2 predictor.set_image → 抽特徵字典
        with AutocastCtx(True):
            self.predictor.set_image(rgb_lb)
        sam2_feats = self.predictor.export_cache()
        # --- 尺寸正規化與記錄（保存時寫成 tuple） ---
        orig_hw = sam2_feats.get("original_size") or sam2_feats.get("orig_size")
        inp_hw  = sam2_feats.get("input_size") or orig_hw
        if isinstance(orig_hw, int): orig_hw = (orig_hw, orig_hw)
        if isinstance(inp_hw,  int): inp_hw  = (inp_hw, inp_hw)
        if orig_hw is not None:
            sam2_feats["original_size"] = tuple(orig_hw)
            sam2_feats["orig_size"]     = tuple(orig_hw)
        if inp_hw is not None:
            sam2_feats["input_size"]    = tuple(inp_hw)

        # DINO grid
        dng = get_grid_feats_dinov3_hf(rgb_lb, self.dinov3_id).permute(2,0,1)  # [Cd,Gh,Gw] float32

        meta_out = {
            "out_h": int(meta['out_h']), "out_w": int(meta['out_w']),
            "top": int(meta['top']), "left": int(meta['left']),
            "scale": float(meta['scale']),
            "orig_h": int(meta['orig_h']), "orig_w": int(meta['orig_w']),
        }
        
        if isinstance(sam2_feats, dict) and "meta" in sam2_feats:
            m2 = sam2_feats["meta"]
            # 你要追蹤的欄位：sizes_saved_as_tuple / original_size_hw / input_size_hw
            meta_out.update({
                "sizes_saved_as_tuple": bool(m2.get("sizes_saved_as_tuple", True)),
                "original_size_hw": tuple(m2.get("original_size_hw", sam2_feats.get("original_size", (meta['out_h'], meta['out_w'])))),
                "input_size_hw":    tuple(m2.get("input_size_hw",    sam2_feats.get("input_size",    (meta['out_h'], meta['out_w'])))),
                "has_high_res_feats": True,
                "high_res_feats_levels": int(m2.get("high_res_feats_levels", len(sam2_feats.get("high_res_feats", []))))
            })
        else:
            # 萬一某些分支沒帶 meta，也保底記錄
            ohw = tuple(sam2_feats.get("original_size", (meta['out_h'], meta['out_w'])))
            ihw = tuple(sam2_feats.get("input_size",    ohw))
            meta_out.update({
                "sizes_saved_as_tuple": True,
                "original_size_hw": ohw,
                "input_size_hw":    ihw,
                "has_high_res_feats": bool(sam2_feats.get("high_res_feats", [])),
                "high_res_feats_levels": len(sam2_feats.get("high_res_feats", [])) if sam2_feats.get("high_res_feats", []) else 0
            })
        
        data = {
            "sam2": sam2_feats,           # ← 直接存 export_cache() 的打包結果
            "dng": dng.half().cpu(),      # 省空間
            "meta": meta_out
        }
        torch.save(data, out)
        
    def _build_all(self):
        uniq = set()
        for e in self.episodes:
            uniq.add(e.target); uniq.add(e.ref)
        uniq = sorted(list(uniq))
        for p in tqdm(uniq, total=len(uniq), desc="Caching images", unit="img"):
            self._build_one(p)
        print(f"Cache built for {len(uniq)} images → {self.cache_dir}")

    def _load_cached(self, path: str):
        key = self._key(path)
        pt = os.path.join(self.cache_dir, f"{key}.pt")
        if not os.path.isfile(pt):
            raise FileNotFoundError(f"Cache not found: {pt}")
        data = torch.load(pt, map_location="cpu")
        sam = _sam2_embed_from_cached(data["sam2"])  # [Cs,Hf,Wf] float32
        dng = data["dng"].to(torch.float32)          # [Cd,Gh,Gw]
        return sam, dng  # [Cs,Hf,Wf], [Cd,Gh,Gw]

    def __len__(self): return len(self.episodes)

    def __getitem__(self, i):
        e = self.episodes[i]
        tgt_rgb = load_rgb(e.target); ref_rgb = load_rgb(e.ref)
        tgt_mask = load_gray(e.target_mask); ref_mask = load_gray(e.ref_mask)
        sam_t, dn_t = self._load_cached(e.target)
        sam_r, dn_r = self._load_cached(e.ref)

        C,Hf,Wf = sam_r.shape
        m_small = cv2.resize(ref_mask, (Wf,Hf), interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
        m_small = torch.from_numpy(m_small)
        fg = (m_small>0.5).float(); bg = (m_small<=0.5).float()
        proto_fg_sam = (sam_r * fg.unsqueeze(0)).sum((1,2))/(fg.sum()+1e-8)
        proto_fg_sam = F.normalize(proto_fg_sam, dim=0)
        sim_fg_sam = (sam_t * proto_fg_sam.view(-1,1,1)).sum(0)
        if bg.sum()<1:
            inv = (1.0-m_small).flatten(); topk=torch.topk(inv, k=min(10, inv.numel()))[1]
            bg = torch.zeros_like(inv); bg[topk]=1.0; bg = bg.view(Hf,Wf)
        proto_bg_sam = (sam_r * bg.unsqueeze(0)).sum((1,2))/(bg.sum()+1e-8)
        proto_bg_sam = F.normalize(proto_bg_sam, dim=0)
        sim_bg_sam = (sam_t * proto_bg_sam.view(-1,1,1)).sum(0)
        sim_sam = torch.sigmoid((sim_fg_sam - sim_bg_sam)/0.2).to(torch.float32)

        Cd, Gh, Gw = dn_r.shape
        m_small2 = cv2.resize(ref_mask, (Gw,Gh), interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
        m_small2 = torch.from_numpy(m_small2)
        fg2 = (m_small2>0.5).float(); bg2 = (m_small2<=0.5).float()
        proto_fg_dn = (dn_r * fg2.unsqueeze(0)).sum((1,2))/(fg2.sum()+1e-8)
        proto_fg_dn = F.normalize(proto_fg_dn, dim=0)
        sim_fg_dn = (dn_t * proto_fg_dn.view(-1, 1, 1)).sum(0)  # -> [Gh, Gw]
        
        proto_bg_dn = (dn_r * bg2.unsqueeze(0)).sum((1,2))/(bg2.sum()+1e-8)
        proto_bg_dn = F.normalize(proto_bg_dn, dim=0)
        sim_bg_dn = (dn_t * proto_bg_dn.view(-1, 1, 1)).sum(0)  # -> [Gh, Gw]
        
        sim_dn = torch.sigmoid((sim_fg_dn - sim_bg_dn)/0.2).to(torch.float32)
        sim_dn_up = F.interpolate(sim_dn.unsqueeze(0).unsqueeze(0), size=(Hf,Wf), mode='bicubic', align_corners=False).squeeze(0).squeeze(0)

        edge = sobel_edge_hint(tgt_rgb, (Hf,Wf))
        # -- 新增：支援 mask 顯式提示（單張支援） --
        S_cond32 = support_mask_to_grid(ref_mask, cache_long=args.cache_long, gridH=16, gridW=16)  # [16,16]
        S_cond_up = F.interpolate(S_cond32.unsqueeze(0).unsqueeze(0), size=(Hf, Wf),
                                  mode='bilinear', align_corners=False).squeeze(0).squeeze(0)      # [Hf,Wf]
        
        x_in = torch.stack([sim_sam, sim_dn_up, edge, S_cond_up], dim=0).unsqueeze(0)  # [1,3,Hf,Wf]

        sim_fg_outside = torch.clamp(sim_sam - (torch.from_numpy((cv2.resize(tgt_mask,(Wf,Hf), interpolation=cv2.INTER_NEAREST)>127).astype(np.float32))), 0, 1)
        H_fg_star, H_bg_star, K_fg_star, K_bg_star = build_gt(tgt_mask, sim_fg_outside, (Hf,Wf))
        sample = {
            'sam_t': sam_t.unsqueeze(0), 'dn_t': dn_t,
            'proto_fg_sam': proto_fg_sam.unsqueeze(0), 'proto_bg_sam': proto_bg_sam.unsqueeze(0),
            'proto_fg_dn': proto_fg_dn.unsqueeze(0), 'proto_bg_dn': proto_bg_dn.unsqueeze(0),
            'x_in': x_in,
            'H_fg_star': H_fg_star.unsqueeze(0), 'H_bg_star': H_bg_star.unsqueeze(0),
            'K_fg_star': torch.tensor(K_fg_star, dtype=torch.long),
            'K_bg_star': torch.tensor(K_bg_star, dtype=torch.long),
            'tgt_rgb': tgt_rgb, 'tgt_mask': tgt_mask,
            'tgt_path': e.target
        }
        return sample

class COCO20iEpisodeAdapter(Dataset):
    def __init__(self, base_dataset, cache_dir: str, cache_long: int = 1024, cache_multiple: int = 16,
                 kmax_f: int = 8, kmax_b: int = 8,
                 hbg_alpha: float = 0.5, hbg_beta: float = 0.5,
                 hbg_ring_r: int = 3,
                 hbg_band_dmin: int = -1, hbg_band_dmax: int = 999999):
        self.base = base_dataset
        self.cache_dir = cache_dir
        self.cache_long = int(cache_long)
        self.cache_multiple = int(cache_multiple)
        self.kmax_f = int(kmax_f)
        self.kmax_b = int(kmax_b)
        self.hbg_alpha = float(hbg_alpha)
        self.hbg_beta  = float(hbg_beta)
        self.hbg_ring_r = int(hbg_ring_r)
        self.hbg_band_dmin = int(hbg_band_dmin)
        self.hbg_band_dmax = int(hbg_band_dmax)


    def __len__(self): return len(self.base)

    def _load_cached(self, img_path: str):
        key = cache_key(img_path)
        pt = os.path.join(self.cache_dir, f"{key}.pt")
        if not os.path.isfile(pt):
            raise FileNotFoundError(f"Cache not found: {pt}")
        data = torch.load(pt, map_location="cpu")
        sam = _sam2_embed_from_cached(data["sam2"])  # CPU float32
        dng = data["dng"].to(torch.float32)          # CPU float32
        out_h = int(data["meta"]["out_h"]); out_w = int(data["meta"]["out_w"])
        return sam, dng, (out_h, out_w)

    def __getitem__(self, i):
        ep = self.base[i]
        sup = ep['support']; qry = ep['query']
        sup_img = (sup['image'].numpy().transpose(1,2,0) * 255.0).astype(np.uint8)
        qry_img = (qry['image'].numpy().transpose(1,2,0) * 255.0).astype(np.uint8)
        sup_mask = (sup['mask'].numpy().astype(np.uint8) * 255)
        qry_mask = (qry['mask'].numpy().astype(np.uint8) * 255)
        sup_path = sup['meta']['image_path']
        qry_path = qry['meta']['image_path']

        sam_r, dn_r, _ = self._load_cached(sup_path)
        sam_t, dn_t, _ = self._load_cached(qry_path)

        Cr, Hr, Wr = sam_r.shape
        Ct, Ht, Wt = sam_t.shape
        m_small = cv2.resize(sup_mask, (Wr, Hr), interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
        m_small = torch.from_numpy(m_small)
        fg = (m_small > 0.5).float()
        bg = (m_small <= 0.5).float()
        proto_fg_sam = (sam_r * fg.unsqueeze(0)).sum((1,2)) / (fg.sum()+1e-8)
        proto_fg_sam = F.normalize(proto_fg_sam, dim=0)
        sim_fg_sam = (sam_t * proto_fg_sam.view(-1,1,1)).sum(0)
        if bg.sum() < 1:
            inv = (1.0-m_small).flatten(); topk = torch.topk(inv, k=min(10, inv.numel()))[1]
            bg = torch.zeros_like(inv); bg[topk]=1.0; bg = bg.view(Hr,Wr)
        proto_bg_sam = (sam_r * bg.unsqueeze(0)).sum((1,2)) / (bg.sum()+1e-8)
        proto_bg_sam = F.normalize(proto_bg_sam, dim=0)
        sim_bg_sam = (sam_t * proto_bg_sam.view(-1,1,1)).sum(0)
        sim_sam = torch.sigmoid((sim_fg_sam - sim_bg_sam)/0.2).to(torch.float32)

        Cd, Gh, Gw = dn_r.shape
        m_small2 = cv2.resize(sup_mask, (Gw,Gh), interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
        m_small2 = torch.from_numpy(m_small2)
        fg2 = (m_small2>0.5).float(); bg2 = (m_small2<=0.5).float()
        proto_fg_dn = (dn_r * fg2.unsqueeze(0)).sum((1,2))/(fg2.sum()+1e-8)
        proto_fg_dn = F.normalize(proto_fg_dn, dim=0)
        sim_fg_dn = (dn_t * proto_fg_dn.view(-1, 1, 1)).sum(0)  # -> [Gh, Gw]
        proto_bg_dn = (dn_r * bg2.unsqueeze(0)).sum((1,2))/(bg2.sum()+1e-8)
        proto_bg_dn = F.normalize(proto_bg_dn, dim=0)
        sim_bg_dn = (dn_t * proto_bg_dn.view(-1, 1, 1)).sum(0)  # -> [Gh, Gw]
        sim_dn = torch.sigmoid((sim_fg_dn - sim_bg_dn)/0.2).to(torch.float32)
        sim_dn_up = F.interpolate(sim_dn.unsqueeze(0).unsqueeze(0),
                                  size=(Ht,Wt), mode='bicubic', align_corners=False
                                 ).squeeze(0).squeeze(0)

        edge = sobel_edge_hint(qry_img, (Ht, Wt))
        S_cond32 = support_mask_to_grid(sup_mask, cache_long=self.cache_long, gridH=16, gridW=16)  # [16,16]
        S_cond_up = F.interpolate(S_cond32.unsqueeze(0).unsqueeze(0), size=(Ht, Wt),
                                  mode='bilinear', align_corners=False).squeeze(0).squeeze(0)       # [Ht,Wt]
        x_in = torch.stack([sim_sam, sim_dn_up, edge, S_cond_up], dim=0).unsqueeze(0)  # [1,3,Ht,Wt]

        sim_fg_outside = torch.clamp(
            sim_sam - torch.from_numpy(
                (cv2.resize(qry_mask,(Wt,Ht), interpolation=cv2.INTER_NEAREST) > 127).astype(np.float32)
            ),
            0, 1
        )
        H_fg_star, H_bg_star, K_fg_star, K_bg_star = build_gt(
            qry_mask, sim_fg_outside, (Ht, Wt),
            kmax_f=self.kmax_f, kmax_b=self.kmax_b,
            alpha=self.hbg_alpha, beta=self.hbg_beta,
            ring_r=self.hbg_ring_r,
            band_dmin_px=self.hbg_band_dmin, band_dmax_px=self.hbg_band_dmax
        )
        rgb_hw = cv2.resize(qry_img, (Wt, Ht), interpolation=cv2.INTER_LINEAR)  # HxW x 3
        rgb_t  = torch.from_numpy(rgb_hw).permute(2,0,1).float() / 255.0            # [3,H,W]
        
        rgb_mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        rgb_std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        rgb_in = (rgb_t - rgb_mean) / rgb_std  

        return {
            'sam_t': sam_t.unsqueeze(0), 'dn_t': dn_t,
            'proto_fg_sam': proto_fg_sam.unsqueeze(0), 'proto_bg_sam': proto_bg_sam.unsqueeze(0),
            'proto_fg_dn':  proto_fg_dn.unsqueeze(0),  'proto_bg_dn':  proto_bg_dn.unsqueeze(0),
            'x_in': x_in,
            'H_fg_star': H_fg_star.unsqueeze(0), 'H_bg_star': H_bg_star.unsqueeze(0),
            'K_fg_star': torch.tensor(K_fg_star, dtype=torch.long),
            'K_bg_star': torch.tensor(K_bg_star, dtype=torch.long),
            'tgt_rgb': qry_img, 'tgt_mask': qry_mask,
            'tgt_path': qry_path, 'rgb_in': rgb_in
        }

# ------------------------------ Losses --------------------------------
class FocalBCEWithLogits(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', eps=1e-6):
        super().__init__()
        self.a = alpha
        self.g = gamma
        self.red = reduction
        self.eps = eps
    def forward(self, logits, target, mask=None):
        ce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        p = torch.sigmoid(logits)
        p_t = p * target + (1 - p) * (1 - target)
        alpha_t = self.a * target + (1 - self.a) * (1 - target)
        mod = (1 - p_t).clamp_min(self.eps) ** self.g
        loss = alpha_t * mod * ce
        if mask is not None:
            loss = loss * mask
            denom = mask.sum().clamp_min(1.0)
            return loss.sum() / denom
        if self.red == 'mean':  return loss.mean()
        if self.red == 'sum':   return loss.sum()
        return loss

def prep_target_like(logits, target):
    if target.dim() == 5 and target.size(2) == 1: target = target.squeeze(2)
    if target.dim() == 3: target = target.unsqueeze(1)
    assert target.dim() == 4, f"target ndim should be 4, got {target.dim()} with shape {tuple(target.shape)}"
    Ht, Wt = target.shape[-2:]; Hl, Wl = logits.shape[-2:]
    if (Ht, Wt) != (Hl, Wl):
        target = F.interpolate(target.float(), size=(Hl, Wl), mode='bilinear', align_corners=False)
        target = target.clamp_(0.0, 1.0)
    return target

# ------------------------------ RL helpers -----------------------------
def build_points_from_actions(a, P, Kfg, Kbg):
    """
    a: [B,16,16] 動作 (0=pos,1=neu,2=neg)
    P: [B,3,16,16] 其 softmax 機率，用來排序
    回傳前景/背景點（只回 batch 單張 B=1 的 (r,c) 清單）
    """
    assert a.dim()==3 and a.shape[0]==1
    A = a[0].detach().cpu().numpy()  # 32x32
    Ppos = P[0,0].detach().cpu().numpy()
    Pneg = P[0,2].detach().cpu().numpy()
    pos_idx = np.argwhere(A==0)
    neg_idx = np.argwhere(A==2)
    if len(pos_idx)>0:
        scores = Ppos[pos_idx[:,0], pos_idx[:,1]]
        order = (-scores).argsort()[:int(Kfg)]
        pos_idx = pos_idx[order]
    if len(neg_idx)>0:
        scores = Pneg[neg_idx[:,0], neg_idx[:,1]]
        order = (-scores).argsort()[:int(Kbg)]
        neg_idx = neg_idx[order]
    pts_fg = [(int(r),int(c)) for r,c in pos_idx]
    pts_bg = [(int(r),int(c)) for r,c in neg_idx]
    return pts_fg, pts_bg

@torch.inference_mode()
def sam2_reward_from_points_cached(pts_fg, pts_bg,
                                   gridH, gridW,
                                   tgt_rgb_raw, tgt_mask_raw,
                                   predictor,
                                   tgt_path: str,
                                   cache_dir: str,
                                   cache_long: int = 1024):
    """
    用 .pt/.npz 快取直接回填 predictor 特徵，避免 set_image。
    """
    pts = pts_fg + pts_bg
    if len(pts) == 0:
        return 0.0, None
    labels = [1]*len(pts_fg) + [0]*len(pts_bg)

    # 1) 用 letterbox 尺寸做座標換算 & 作為 fallback_hw
    tgt_rgb_lb, _ = resize_letterbox_rgb(tgt_rgb_raw, cache_long)
    H_img, W_img = tgt_rgb_lb.shape[:2]
    xy = np.array([grid_to_xy((r, c), gridH, gridW, H_img, W_img) for (r, c) in pts], dtype=np.float32)

    # 2) 從快取回填 predictor 狀態（優先 .pt，否則 .npz，最後才 fallback set_image）
    key = cache_key(tgt_path)
    pt_path  = os.path.join(cache_dir, f"{key}.pt")
    npz_path = os.path.join(cache_dir, f"{key}.npz")

    feats_cached = None
    if os.path.isfile(pt_path):
        data = torch.load(pt_path, map_location="cpu")
        if "sam2" in data and isinstance(data["sam2"], dict):
            feats_cached = data["sam2"]
        elif "sam" in data:  # 舊式：只存 embedding
            feats_cached = {"image_embeddings": data["sam"]}
        else:
            # 直接當作 feats 字典
            feats_cached = data
    elif os.path.isfile(npz_path):
        data = np.load(npz_path)
        if "sam" in data:
            sam = torch.from_numpy(data["sam"]).to(torch.float16)  # [C,Hf,Wf]
            feats_cached = {"image_embeddings": sam}
        # 若也存了尺寸資訊，可一併補上
    # 回填 or fallback
    if feats_cached is not None:
        predictor.set_image_from_cache(data["sam2"])
    else:
        # 沒快取才真正編碼（訓練時不建議發生）
        with AutocastCtx(True):
            predictor.set_image(tgt_rgb_lb)
    
    _patch_predictor_transforms(predictor)

    # 3) 解碼 → 取 best mask → 計 IoU
    _force_size_attrs(predictor, (H_img, W_img))   # ★ 補齊所有尺寸欄位
    
    masks, scores, _ = predictor.predict(point_coords=xy.astype(np.float32),
                                         point_labels=np.array(labels, dtype=np.int32),
                                         multimask_output=True)
    j = int(np.argmax(scores))
    m = masks[j].astype(np.uint8)

    gt_lb = resize_letterbox_mask(tgt_mask_raw, cache_long)
    gt = (gt_lb > 127).astype(np.uint8)
    inter = (m > 0) & (gt > 0)
    uni   = (m > 0) | (gt > 0)
    iou = float(inter.sum()) / float(uni.sum() + 1e-6)
    return iou, {'xy': xy, 'labels': labels}

@torch.inference_mode()
def sam2_reward_from_points_cached_batch(
    pts_fg_list, pts_bg_list,      # 長度 = B；每個元素是 [(r,c), ...]
    gridH, gridW,
    tgt_rgb_list, tgt_mask_list,   # 長度 = B；numpy HWC / HW
    tgt_path_list,                 # 長度 = B；每張圖的原始路徑（用來找對應 cache .pt）
    predictor,                     # SAM2ImagePredictor（共用一個實例）
    cache_dir: str,
    cache_long: int = 1024,
):
    """
    只用 .pt 快取回填 predictor 狀態（set_image_from_cache）
    不做 batched encoder，逐張圖直接解碼 → 計 IoU
    """
    B = len(tgt_rgb_list)
    if B == 0:
        return [], []

    ious = []
    infos = []

    for b in range(B):
        rgb_raw  = tgt_rgb_list[b]
        mask_raw = tgt_mask_list[b]
        tgt_path = tgt_path_list[b]

        # 若沒有點，定義 IoU=0
        pts_fg = pts_fg_list[b]
        pts_bg = pts_bg_list[b]
        pts    = pts_fg + pts_bg
        if len(pts) == 0:
            ious.append(0.0)
            infos.append({'xy': None, 'labels': None})
            continue

        # 用 letterbox 尺寸做網格→像素座標換算（不會觸發 encoder）
        rgb_lb, _ = resize_letterbox_rgb(rgb_raw, cache_long)
        H_img, W_img = rgb_lb.shape[:2]
        xy = np.array(
            [grid_to_xy(rc, gridH, gridW, H_img, W_img) for rc in pts],
            dtype=np.float32
        )
        labels = np.array([1]*len(pts_fg) + [0]*len(pts_bg), dtype=np.int32)

        # 從 .pt 快取回填（不跑 encoder）
        key = cache_key(tgt_path)
        pt_path = os.path.join(cache_dir, f"{key}.pt")
        if not os.path.isfile(pt_path):
            # 若缺檔，維持與單圖 cached 版本相同的回退策略：給 0 分
            ious.append(0.0)
            infos.append({'xy': None, 'labels': None})
            continue

        data = torch.load(pt_path, map_location="cpu")
        # 直接把 full cache dict 回填
        predictor.set_image_from_cache(data["sam2"])
        _patch_predictor_transforms(predictor)
        _force_size_attrs(predictor, (H_img, W_img))  # 尺寸護欄

        # 解碼（multimask）
        masks, scores, _ = predictor.predict(
            point_coords=xy.astype(np.float32),
            point_labels=labels.astype(np.int32),
            multimask_output=True
        )
        j = int(np.argmax(scores))
        m = masks[j].astype(np.uint8)

        # 計 IoU（與 letterbox 後的 GT）
        gt_lb = resize_letterbox_mask(mask_raw, cache_long)
        gt = (gt_lb > 127).astype(np.uint8)

        inter = (m > 0) & (gt > 0)
        union = (m > 0) | (gt > 0)
        iou = float(inter.sum()) / float(union.sum() + 1e-6)

        ious.append(iou)
        infos.append({'xy': xy, 'labels': labels})

    return ious, infos


@torch.inference_mode()
def sam2_reward_from_points_batched(
    pts_fg_list, pts_bg_list,  # 長度 = B；每個元素是 [(r,c), ...]
    gridH, gridW,
    tgt_rgb_list, tgt_mask_list,  # 長度 = B；numpy HWC / HW
    predictor,                    # SAM2ImagePredictor
    cache_long: int = 1024,
):
    """
    使用 sam2_image_predictor.py 的 batch API：
      - set_image_batch([...])
      - predict_batch(point_coords_batch=[...], point_labels_batch=[...])
    直接對一個 batch 的影像同時計算 masks，回傳每張 IoU。
    """
    B = len(tgt_rgb_list)
    if B == 0:
        return [], []

    # 1) 先把影像做 letterbox（同訓練流程），並把 grid rc → xy（像素座標）
    rgb_lb_list = []
    coords_batch, labels_batch = [], []
    for b in range(B):
        rgb_raw = tgt_rgb_list[b]
        rgb_lb, _ = resize_letterbox_rgb(rgb_raw, cache_long)
        H_img, W_img = rgb_lb.shape[:2]
        rgb_lb_list.append(rgb_lb)

        pts_fg = pts_fg_list[b]; pts_bg = pts_bg_list[b]
        pts = pts_fg + pts_bg
        if len(pts) == 0:
            coords_batch.append(None)   # 沒點：給 None
            labels_batch.append(None)
            continue
        xy = np.array([grid_to_xy(rc, gridH, gridW, H_img, W_img) for rc in pts], dtype=np.float32)
        labels = np.array([1]*len(pts_fg) + [0]*len(pts_bg), dtype=np.int32)
        coords_batch.append(xy)
        labels_batch.append(labels)

    # 2) 批次做 embedding 與解碼（注意：這裡會重新跑影像 encoder）
    predictor.reset_predictor()
    predictor.set_image_batch(rgb_lb_list)

    masks_list, scores_list, _ = predictor.predict_batch(
        point_coords_batch=coords_batch,
        point_labels_batch=labels_batch,
        multimask_output=True,
        return_logits=False,
        normalize_coords=True,
    )

    # 3) 計算每張 IoU（與 letterbox 後的 GT）
    ious = []
    infos = []
    for b in range(B):
        gt_lb = resize_letterbox_mask(tgt_mask_list[b], cache_long)
        gt = (gt_lb > 127).astype(np.uint8)

        if coords_batch[b] is None:
            ious.append(0.0)
            infos.append({'xy': None, 'labels': None})
            continue

        # 取最佳分支
        s = scores_list[b]
        j = int(np.argmax(s))
        m = masks_list[b][j].astype(np.uint8)

        inter = (m > 0) & (gt > 0)
        union = (m > 0) | (gt > 0)
        iou = float(inter.sum()) / float(union.sum() + 1e-6)
        ious.append(iou)
        infos.append({'xy': coords_batch[b], 'labels': labels_batch[b]})

    return ious, infos

@torch.inference_mode()
def build_gt32_and_dist32(tgt_mask_list, cache_long: int, gridH: int, gridW: int, device):
    """
    輸入:
        - tgt_mask_list: 長度 B 的 list，每個是 numpy HW (0..255)
        - cache_long: letterbox 尺寸（例如 512）
        - gridH, gridW: 策略網格尺寸（例如 16,16）
    輸出:
        - gt32:  [B, gridH, gridW]，每格的「前景佔比」(0..1)
        - dist32:[B, gridH, gridW]，每格的「到前景邊界的平均距離(px)」
    """
    kh, kw = cache_long // gridH, cache_long // gridW
    assert cache_long % gridH == 0 and cache_long % gridW == 0, \
        f"cache_long={cache_long} 必須能被 gridH={gridH}, gridW={gridW} 整除"

    gt_list, d_list = [], []
    for m in tgt_mask_list:
        # 1) letterbox 到 cache_long×cache_long
        gt_lb = resize_letterbox_mask(m, cache_long)  # [H,W] uint8 (0..255)

        # 2) 前景二值 & 背景距離圖（px）
        gt_bin = (gt_lb > 127).astype(np.float32)            # [H,W] {0,1}
        bg     = (gt_lb <= 127).astype(np.uint8)
        dmap   = cv2.distanceTransform(bg, cv2.DIST_L2, 3).astype(np.float32)  # [H,W] float32 (px)

        # 3) 兩個通道打包，做一次整數平均池化 → grid 尺寸
        x = torch.from_numpy(np.stack([gt_bin, dmap], axis=0))[None, ...]  # [1,2,H,W]
        pooled = F.avg_pool2d(x, kernel_size=(kh, kw), stride=(kh, kw), padding=0)  # [1,2,gridH,gridW]

        gt_list.append(pooled[0, 0])   # 前景佔比
        d_list.append(pooled[0, 1])    # 平均距離(px)

    gt32   = torch.stack(gt_list, dim=0).to(device=device, dtype=torch.float32)   # [B,gridH,gridW]
    dist32 = torch.stack(d_list, dim=0).to(device=device, dtype=torch.float32)    # [B,gridH,gridW]
    return gt32, dist32

# ------------------------------ Train / Eval --------------------------
def train_loop(args):
    start_epoch = 0
    best_iou = 0.0
    lc = 0.0  # 不再使用 L_cnt

    # === 建 COCO-20i 索引 ===
    assert args.manifest_train is not None, "Please set --manifest-train (manifest_train.csv)"
    idx_train = build_coco20i_index(args.manifest_train, fold=args.coco20i_fold, role=args.role)

    base_train = OneShotCOCO20iRandom(index=idx_train, episodes=args.episodes, seed=2025)
    ds = COCO20iEpisodeAdapter(
        base_train, args.cache_dir,
        cache_long=args.cache_long, cache_multiple=args.cache_multiple,
        kmax_f=args.kmax_f, kmax_b=args.kmax_b,
        hbg_alpha=args.hbg_alpha, hbg_beta=args.hbg_beta,
        hbg_ring_r=args.hbg_ring_r,
        hbg_band_dmin=args.hbg_band_dmin, hbg_band_dmax=args.hbg_band_dmax
    )
    dl_torch = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=True, drop_last=True,
        collate_fn=collate_keep_rgb_mask_as_list,
    )
    # eval data
    idx_val = build_coco20i_index(args.manifest_val, fold=args.coco20i_fold, role=args.role)
    base_val = OneShotCOCO20iRandom(index=idx_val, episodes=args.val_samples, seed=2025)
    ds_val = COCO20iEpisodeAdapter(
        base_val, args.cache_dir,
        cache_long=args.cache_long, cache_multiple=args.cache_multiple,
        kmax_f=args.kmax_f, kmax_b=args.kmax_b,
        hbg_alpha=args.hbg_alpha, hbg_beta=args.hbg_beta,
        hbg_ring_r=args.hbg_ring_r,
        hbg_band_dmin=args.hbg_band_dmin, hbg_band_dmax=args.hbg_band_dmax
    )
    eval_loader = DataLoader(ds_val, batch_size=1, shuffle=False, collate_fn=collate_keep_rgb_mask_as_list)

    # === 建模型（c_in=proj_dim*2 + 3） ===
    probe = ds[0]
    sam_C = probe['sam_t'].shape[1]
    dino_C = probe['dn_t'].shape[0]
    x_in_ch = probe['x_in'].shape[1]  # 4
    proj_dim = 48
    c_in = proj_dim*2 + x_in_ch  # 96 + 4 = 100
    model = PointPromptNetB(c_in=c_in, kmax_f=args.kmax_f, kmax_b=args.kmax_b, sam_c=sam_C, dino_c=dino_C, proj_dim=proj_dim).to(device)
    base_lr = args.lr
    policy_params = list(model.grid_head.parameters())
    rest_params   = [p for n,p in model.named_parameters() if not n.startswith('grid_head')]
    opt = torch.optim.AdamW(
        [{'params': policy_params, 'lr': 5.0 * base_lr},   # grid/policy head 放大 5×
            {'params': rest_params,   'lr': base_lr}],
        weight_decay=args.wd
    )

    total_steps = args.all_epochs * max(1, len(dl_torch))
    warmup_steps = int(0.1 * total_steps)
    sch = SequentialLR(
        opt,
        schedulers=[
            LinearLR(opt, start_factor=1e-6/args.lr, end_factor=1.0, total_iters=warmup_steps),
            CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=1e-6)
        ],
        milestones=[warmup_steps]
    )
    bce = FocalBCEWithLogits(alpha=0.75, gamma=2.0)

    global_step = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'], strict=False)
        if 'optimizer' in ckpt: opt.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt: sch.load_state_dict(ckpt['scheduler'])
        best_iou = float(ckpt.get('best_iou', 0.0))
        start_epoch = int(ckpt.get('epoch', -1)) + 1
        print(f"[resume] from {args.resume} → epoch {start_epoch}, step {global_step}, best_iou {best_iou:.4f}")

    if args.torch_compile:
        model = torch.compile(model, mode='reduce-overhead')

    # 共用 SAM2 predictor（僅作 decoder，用快取回填，不再編碼）
    train_loop._sam2_pred = sam2_build_image_predictor(args.sam2_cfg, args.sam2_ckpt)
    # ---- simple linear scheduler helpers ----
    def lerp(a, b, t): return a + (b - a) * t
    def sched(start, end, epoch, total_epochs):
        if total_epochs <= 1: return end
        t = max(0.0, min(1.0, epoch / (total_epochs - 1)))
        return lerp(start, end, t)
    
    for epoch in range(start_epoch, args.epochs):
        global_step = 0
        w_heat       = sched(args.w_heat_start,       args.w_heat_end,       epoch, args.epochs)
        lambda_pg    = sched(args.lambda_pg_start,    args.lambda_pg_end,    epoch, args.epochs)
        entropy_beta = sched(args.entropy_beta_start, args.entropy_beta_end, epoch, args.epochs)
        tau          = sched(args.tau_start,          args.tau_end,          epoch, args.epochs)
        adv_scale    = sched(args.adv_scale_start,    args.adv_scale_end,    epoch, args.epochs)
        sample_n_f   = sched(float(args.sample_n_start), float(args.sample_n_end), epoch, args.epochs)
        sample_n     = int(max(1, round(sample_n_f)))

        model.train(); t0=time.time()
        for it, batch in enumerate(tqdm(dl_torch, desc=f"Epoch {epoch}", unit="batch")):
            # 取 batch
            x_in   = batch['x_in'].to(device)
            sam_t  = batch['sam_t'].to(device).squeeze(1)
            dn_t   = batch['dn_t'].to(device)
            Hfg_s  = batch['H_fg_star'].to(device)
            Hbg_s  = batch['H_bg_star'].to(device)
            Kfg_s  = batch['K_fg_star'].to(device)
            Kbg_s  = batch['K_bg_star'].to(device)
            proto_fg_sam = batch['proto_fg_sam'].to(device).squeeze(1)
            proto_bg_sam = batch['proto_bg_sam'].to(device).squeeze(1)
            proto_fg_dn  = batch['proto_fg_dn'].to(device).squeeze(1)
            proto_bg_dn  = batch['proto_bg_dn'].to(device).squeeze(1)
            
            # ★ 取 rgb_in（可能不存在，安全地取）
            rgb_in = batch.get('rgb_in', None)
            if rgb_in is not None:
                rgb_in = rgb_in.to(device)
            
            # 前向
            H_fg, H_bg, G = model(
                x_in, sam_t, dn_t,
                proto_fg_sam, proto_bg_sam, proto_fg_dn, proto_bg_dn,
                rgb_in=rgb_in   # ★ 傳進去
            )

            # 像素級熱度損失
            H_fg = H_fg.float(); H_bg = H_bg.float()
            Hfg_s = prep_target_like(H_fg, Hfg_s.float())
            Hbg_s = prep_target_like(H_bg, Hbg_s.float())
            L_heat = bce(H_fg, Hfg_s) + bce(H_bg, Hbg_s)

            # === RL：SCST（整批） ===
            B = x_in.size(0)
            gridH = gridW = 16

            # 溫度取樣：logits / tau（數值護欄）
            tau = float(max(1e-3, tau))  # 防止極小溫度
            G = torch.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0)  # 清掉 NaN/Inf
            logits_tau = (G / tau).clamp(-50, 50).permute(0,2,3,1).contiguous().view(B*gridH*gridW, 3)
            dist = Categorical(logits=logits_tau)
            
            # 排序/取點用的機率圖也基於護欄後的 logits
            P_tau = torch.softmax(((G / tau).clamp(-50, 50)).detach(), dim=1)  # [B,3,16,16]

            # greedy 基準（也基於 tau 的分佈）
            a_greedy = (G / tau).argmax(dim=1)

            # 供排序/取點（也用 tau）
            P_tau = torch.softmax(G.detach() / tau, dim=1)  # [B,3,16,16]

            # H_fg_star/H_bg_star 的 bonus（16×16）
            Hfg32 = F.interpolate(Hfg_s, size=(gridH,gridW), mode='area').squeeze(1)
            Hbg32 = F.interpolate(Hbg_s, size=(gridH,gridW), mode='area').squeeze(1)
            alpha_bonus = float(getattr(args, 'bonus_alpha', 0.5))
            iou_gamma   = float(getattr(args, 'iou_gamma', 1.5))
            
            with torch.no_grad():
                gt32, dist32 = build_gt32_and_dist32(batch['tgt_mask'], args.cache_long, gridH, gridW, device)
            
            
            # 將整個 batch 的點集建好（helper）
            neg_min_dist_px = int(getattr(args, 'neg_min_dist_px', 0))
            neg_close_penalty = float(getattr(args, 'neg_close_penalty', -0.2))
            neg_on_fg_penalty   = float(getattr(args, 'neg_on_fg_penalty', -0.2))
            neg_on_fg_thresh    = float(getattr(args, 'neg_on_fg_thresh', 0.25))

            def bonus_for(b, pts_fg, pts_bg):
                # H* 加分（向量化）
                if len(pts_fg):
                    idx_fg = torch.tensor(pts_fg, device=device, dtype=torch.long).T  # [2, Nf]
                    bf = Hfg32[b, idx_fg[0], idx_fg[1]].mean()
                else:
                    bf = torch.tensor(0.0, device=device)
            
                if len(pts_bg):
                    idx_bg = torch.tensor(pts_bg, device=device, dtype=torch.long).T  # [2, Nb]
                    bb = Hbg32[b, idx_bg[0], idx_bg[1]].mean()
                else:
                    bb = torch.tensor(0.0, device=device)
            
                bonus = alpha_bonus * (bf + bb)
            
                # 懲罰 A：距離
                if len(pts_bg) and neg_min_dist_px > 0:
                    close_frac = (dist32[b, idx_bg[0], idx_bg[1]] < neg_min_dist_px).float().mean()
                    bonus = bonus + neg_close_penalty * close_frac
            
                # 懲罰 B：on-FG
                if len(pts_bg) and neg_on_fg_penalty < 0:
                    onfg_frac = (gt32[b, idx_bg[0], idx_bg[1]] >= neg_on_fg_thresh).float().mean()
                    bonus = bonus + neg_on_fg_penalty * onfg_frac
            
                # 回傳 python float 或保留 tensor 都可；若後面要和 numpy 相加再 .item()
                return bonus
            
            def build_pts_lists(A, Pfull):
                pts_fg_list, pts_bg_list = [], []
                for b in range(B):
                    pts_fg, pts_bg = build_points_from_actions(A[b:b+1], Pfull[b:b+1], args.kmax_f, args.kmax_b)
                    pts_fg_list.append(pts_fg)
                    pts_bg_list.append(pts_bg)
                return pts_fg_list, pts_bg_list
            
            # ---- 多重取樣（前期 2、後期 1）----
            logp_samples   = []  # list of [B]
            R_sample_runs  = []  # list of [B]
            R_greedy_runs  = []  # list of [B]

            for _ in range(sample_n):
                a_sample = dist.sample().view(B, gridH, gridW)

                # 只對「抽到的動作」取 log_prob（避免訊號被 32x32 平均稀釋）
                logP_tau = F.log_softmax((G / tau).clamp(-50,50), dim=1)                  # [B,3,16,16]
                logP_a   = logP_tau.gather(1, a_sample.unsqueeze(1)).squeeze(1)           # [B,16,16]
                logp_samples.append(logP_a.mean(dim=(1,2)))                                # [B]

                # 建點
                pts_fg_s_list, pts_bg_s_list = build_pts_lists(a_sample, P_tau)
                pts_fg_g_list, pts_bg_g_list = build_pts_lists(a_greedy, P_tau)

                # 準備整批影像 / GT
                tgt_rgb_list  = batch['tgt_rgb']   # list 長度 B
                tgt_mask_list = batch['tgt_mask']  # list 長度 B

                pred = train_loop._sam2_pred  # 共用 predictor

                # batched decode：抽樣/貪婪各解一次
                # 需要 batch['tgt_path']（collate 已保留成 list）
                tgt_path_list = batch['tgt_path']
                
                R_s_list, _ = sam2_reward_from_points_cached_batch(
                    pts_fg_s_list, pts_bg_s_list, gridH, gridW,
                    tgt_rgb_list, tgt_mask_list, tgt_path_list,
                    pred, cache_dir=args.cache_dir, cache_long=args.cache_long
                )
                R_g_list, _ = sam2_reward_from_points_cached_batch(
                    pts_fg_g_list, pts_bg_g_list, gridH, gridW,
                    tgt_rgb_list, tgt_mask_list, tgt_path_list,
                    pred, cache_dir=args.cache_dir, cache_long=args.cache_long
                )

                R_sample_t = []
                R_greedy_t = []
                for b in range(B):
                    Rs = (max(0.0, float(R_s_list[b])) ** iou_gamma) + bonus_for(b, pts_fg_s_list[b], pts_bg_s_list[b])
                    Rg = (max(0.0, float(R_g_list[b])) ** iou_gamma) + bonus_for(b, pts_fg_g_list[b], pts_bg_g_list[b])
                    R_sample_t.append(Rs)
                    R_greedy_t.append(Rg)

                R_sample_runs.append(torch.tensor(R_sample_t, device=device))   # [B]
                R_greedy_runs.append(torch.tensor(R_greedy_t, device=device))   # [B]

            # 多重取樣 → 平均 reward 與 logp
            logp_sample = torch.stack(logp_samples,  dim=0).mean(dim=0)  # [B]
            R_sample    = torch.stack(R_sample_runs, dim=0).mean(dim=0)  # [B]
            R_greedy    = torch.stack(R_greedy_runs, dim=0).mean(dim=0)  # [B]

            # 優勢（[B]）
            advantage = (R_sample - R_greedy).detach()                 # [B]
            if bool(getattr(args, 'adv_norm', True)) and advantage.numel() > 1:
                std = advantage.std(unbiased=False).clamp_min(1e-6)    # 避免 DOF 問題
                advantage = (advantage - advantage.mean()) / std
            advantage = advantage * adv_scale                          # [B]
            
            # Policy gradient（對 batch 取平均）
            L_pg = -(advantage * logp_sample).mean()                   # scalar
            
            # 熵正則（用帶護欄的 logits）
            P_full = torch.softmax(((G / max(1e-3, tau)).clamp(-50,50)), dim=1)
            entropy = -(P_full * (P_full.clamp_min(1e-8).log())).sum(dim=1).mean()
            
            # ---- 總損失：像素監督讓位、RL 權重大、熵逐步衰減 ----
            loss_total = w_heat * L_heat + lambda_pg * L_pg - entropy_beta * entropy

            opt.zero_grad(set_to_none=True)
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step(); sch.step(); global_step += 1

            if (global_step % args.log_every)==0:
                Rs = R_sample.detach()
                Rg = R_greedy.detach()
                
                Rs_mean = Rs.mean().item()
                Rg_mean = Rg.mean().item()
                print(f"epoch {epoch} it {it} total {loss_total.item():.4f} "
                      f"L_heat {L_heat.item():.4f} L_pg {L_pg.item():.4f} "
                      f"R_s_list_mean {Rs_mean:.4f} R_g_list_mean {Rg_mean:.4f} "
                      f"lr {sch.get_last_lr()[0]:.3e}")
                print(f"(sched) w_heat {w_heat:.2f}  lambda_pg {lambda_pg:.2f}  "
                    f"entropy_beta {entropy_beta:.2e}  tau {tau:.2f}  "
                    f"adv_scale {adv_scale:.2f}  sample_n {sample_n}")
                print('\n')

        if (epoch+1) % args.eval_every == 0 and args.manifest_val:
            iou = evaluate(args, model, args.manifest_val, eval_loader)
            if best_iou < iou:
                best_iou = iou
                print(f"\033[94mBest IoU : {best_iou}\033[0m\n")
                os.makedirs(args.out_dir, exist_ok=True)
                save_checkpoint(os.path.join(args.out_dir, "ppnet_best.pt"),
                                model, opt, sch, epoch, best_iou, lc)
        if (epoch+1) % args.save_every == 0:
            os.makedirs(args.out_dir, exist_ok=True)
            save_checkpoint(os.path.join(args.out_dir, f"ppnet_epoch{epoch}.pt"),
                            model, opt, sch, epoch, best_iou, lc)
            print('Save checkpt.')

        print(f"[epoch {epoch}] time {time.time()-t0:.1f}s\n")

# ------------------------------ Evaluation ----------------------------
@torch.inference_mode()
def evaluate(args, model, manifest_val_path=None, loader=None):
    model.eval()
    device = next(model.parameters()).device

    if loader is None:
        idx_val = build_coco20i_index(manifest_val_path, fold=args.coco20i_fold, role=args.role)
        base_val = OneShotCOCO20iRoundRobin(index=idx_val, seed=2025, shuffle_classes=True)
        ds_val = COCO20iEpisodeAdapter(base_val, args.cache_dir, cache_long=args.cache_long, cache_multiple=args.cache_multiple)
        loader = DataLoader(ds_val, batch_size=1, shuffle=False, collate_fn=collate_keep_rgb_mask_as_list)

    ious = []
    predictor = sam2_build_image_predictor(args.sam2_cfg, args.sam2_ckpt) if args.val_with_sam2 else None
    skip = 0
    gridH = gridW = 16

    for batch in loader:
        x_in  = batch['x_in'].to(device)
        sam_t = batch['sam_t'].to(device).squeeze(1)
        dn_t  = batch['dn_t'].to(device)
        proto_fg_sam = batch['proto_fg_sam'].to(device).squeeze(1)
        proto_bg_sam = batch['proto_bg_sam'].to(device).squeeze(1)
        proto_fg_dn  = batch['proto_fg_dn'].to(device).squeeze(1)
        proto_bg_dn  = batch['proto_bg_dn'].to(device).squeeze(1)
        rgb_in = batch.get('rgb_in', None)
        if rgb_in is not None:
            rgb_in = rgb_in.to(device)


        H_fg, H_bg, G = model(
                        x_in, sam_t, dn_t,
                        proto_fg_sam, proto_bg_sam, proto_fg_dn, proto_bg_dn,
                        rgb_in=rgb_in   # ★ 傳進去
                    )
        
        # 選點（整批）
        P = torch.softmax(G, dim=1)   # [B,3,16,16]
        Ppos, Pneg = P[:,0], P[:,2]

        def topk_rc_batch(P2d_batch, K):
            outs = []
            B = P2d_batch.size(0)
            for b in range(B):
                P2d = P2d_batch[b]
                flat = P2d.view(-1)
                if K<=0 or flat.numel()==0:
                    outs.append([])
                    continue
                k = min(K, flat.numel())
                _, idxs = torch.topk(flat, k=k)
                rc = [(int(i//gridW), int(i%gridW)) for i in idxs]
                outs.append(rc)
            return outs

        pts_fg_list = topk_rc_batch(Ppos, args.kmax_f)
        pts_bg_list = topk_rc_batch(Pneg, args.kmax_b)

        # 若整批都沒點，則略過
        if all((len(pts_fg_list[b]) + len(pts_bg_list[b]) == 0) for b in range(x_in.size(0))):
            skip += 1
            continue

        if predictor is not None:
            tgt_rgb_list  = batch['tgt_rgb']
            tgt_mask_list = batch['tgt_mask']

            # 用 batched 版本一次算出所有 IoU（與 GT 同為 cache_long 尺寸）
            ious_batch, _ = sam2_reward_from_points_batched(
                pts_fg_list, pts_bg_list, gridH, gridW,
                tgt_rgb_list, tgt_mask_list, predictor, cache_long=args.cache_long
            )
            ious.extend(ious_batch)

            if args.val_samples > 0 and len(ious) >= args.val_samples:
                break

    print(f"skipping times: {skip}")
    if ious:
        miou = float(np.mean(ious))
        print(f"Val mIoU (SAM2) over {len(ious)}: {miou:.4f}\n")
        return miou
    else:
        return 0.0

# ------------------------------ CSV utils -----------------------------
def read_csv(path: str) -> List[Episode]:
    out=[]
    with open(path, 'r', newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            out.append(Episode(target=row['target'], target_mask=row['target_mask'],
                               ref=row['ref'], ref_mask=row['ref_mask'], split=row.get('split','train')))
    return out

def build_cache_from_manifests(cache_dir, dinov3_id, sam2_cfg, sam2_ckpt,
                               manifest_paths, cache_long=1024, cache_multiple=16):
    ds = EpisodeDataset([], cache_dir, dinov3_id, sam2_cfg, sam2_ckpt,
                        build_cache=False, cache_long=cache_long, cache_multiple=cache_multiple)
    paths = set()
    for mp in manifest_paths:
        with open(mp, 'r', newline='') as f:
            r = csv.DictReader(f)
            for row in r:
                if 'image_path' in row:
                    paths.add(row['image_path'])
    paths = sorted(paths)
    for p in tqdm(paths, desc=f"Caching images from manifests", unit="img"):
        ds._build_one(p)
    print(f"Cache built for {len(paths)} images → {cache_dir}")

# ------------------------------ Inference demo ------------------------
@torch.inference_mode()
def run_infer(args):
    tgt_rgb_raw = load_rgb(args.target)
    tgt_mask_raw = load_gray(args.target_mask) if args.target_mask else None
    ref_rgb_raw = load_rgb(args.ref); ref_mask_raw = load_gray(args.ref_mask)

    predictor = sam2_build_image_predictor(args.sam2_cfg, args.sam2_ckpt)

    # 優先從快取取特徵，缺檔再 fallback
    try:
        feats = compute_proto_and_sims_from_cache(
            args.ref, ref_mask_raw, args.target, args.cache_dir, use_bg_proto=True
        )
    except Exception as e:
        # 後備：即時編碼（僅 demo 使用）
        tgt_rgb, _ = resize_letterbox_rgb(tgt_rgb_raw, args.cache_long)
        ref_rgb, _ = resize_letterbox_rgb(ref_rgb_raw, args.cache_long)
        with AutocastCtx(True):
            predictor.set_image(ref_rgb)
        sam_r_dict = _get_predictor_features_dict(predictor)
        sam_r = _sam2_embed_from_cached(sam_r_dict)
        with AutocastCtx(True):
            predictor.set_image(tgt_rgb)
        sam_t_dict = _get_predictor_features_dict(predictor)
        sam_t = _sam2_embed_from_cached(sam_t_dict)
        dn_r = get_grid_feats_dinov3_hf(ref_rgb, args.dinov3_model_id).permute(2,0,1)
        dn_t = get_grid_feats_dinov3_hf(tgt_rgb, args.dinov3_model_id).permute(2,0,1)
        # 簡化成與上面一致的計算（略）

        raise RuntimeError("Inference fallback path not fully implemented for brevity; please build cache first.") from e

    sam_t = feats['sam_feat_tgt'].unsqueeze(0)                  # [1,C,Hf,Wf]
    dn_t  = feats['dino_feat_tgt'].permute(2,0,1)               # [Cd,Gh,Gw]
    Hf,Wf = sam_t.shape[-2:]
    sim_dn_up = F.interpolate(feats['sim_dino'].unsqueeze(0).unsqueeze(0), size=(Hf,Wf), mode='bicubic', align_corners=False).squeeze(0).squeeze(0)
    # 生成 edge 需要 letterbox 後的 RGB
    tgt_rgb_lb, _ = resize_letterbox_rgb(tgt_rgb_raw, args.cache_long)
    edge = sobel_edge_hint(tgt_rgb_lb, (Hf,Wf))
    x_in = torch.stack([feats['sim_sam'], sim_dn_up, edge], dim=0).unsqueeze(0).to(device)

    # 模型
    proj_dim=48; c_in = proj_dim*2 + 4
    model = PointPromptNetB(c_in=c_in, kmax_f=args.kmax_f, kmax_b=args.kmax_b).to(device)
    ckpt = torch.load(args.ckpt, map_location=device) if args.ckpt else None
    if ckpt: model.load_state_dict(ckpt['model'], strict=False)
    H_fg, H_bg, G = model(x_in.to(device), sam_t.to(device), dn_t.to(device),
                          feats['proto_fg_sam'].unsqueeze(0), feats.get('proto_bg_sam', None).unsqueeze(0),
                          feats['proto_fg_dn'].unsqueeze(0), feats.get('proto_bg_dn', None).unsqueeze(0))

    # 用 grid 取點 → SAM2（用快取回填 predictor）
    P = torch.softmax(G, dim=1)
    Ppos, Pneg = P[:,0], P[:,2]
    gridH=gridW=16
    def topk_rc(P2d, K):
        flat = P2d.view(-1)
        k = min(K, flat.numel())
        vals, idxs = torch.topk(flat, k=k)
        rc = [(int(i//gridW), int(i%gridW)) for i in idxs]
        return rc
    pts_fg = topk_rc(Ppos[0], args.kmax_f)
    pts_bg = topk_rc(Pneg[0], args.kmax_b)

    tgt_rgb_lb, _ = resize_letterbox_rgb(tgt_rgb_raw, args.cache_long)
    H_img, W_img = tgt_rgb_lb.shape[:2]
    pts_xy = np.array([grid_to_xy(rc, gridH, gridW, H_img, W_img) for rc in (pts_fg+pts_bg)], dtype=np.float32)
    labels = np.array([1]*len(pts_fg) + [0]*len(pts_bg), dtype=np.int32)

    # 用 cache 回填 predictor
    key = cache_key(args.target)
    pt = os.path.join(args.cache_dir, f"{key}.pt")
    if os.path.isfile(pt):
        data = torch.load(pt, map_location="cpu")
        predictor.set_image_from_cache(data["sam2"])
    else:
        with AutocastCtx(True):
            predictor.set_image(tgt_rgb_lb)
    
    _patch_predictor_transforms(predictor)
    
    masks, scores, _ = predictor.predict(point_coords=pts_xy.astype(np.float32), point_labels=labels, multimask_output=True)
    j = int(np.argmax(scores)); m = masks[j].astype(np.uint8)
    overlay = tgt_rgb_lb.copy(); overlay[m>0] = (overlay[m>0]*0.5 + np.array([0,255,0])*0.5).astype(np.uint8)
    for (x,y),lb in zip(pts_xy.astype(np.int32), labels):
        color = (0,255,0) if lb==1 else (255,0,0)
        cv2.circle(overlay, (int(x),int(y)), 5, color, -1)
    os.makedirs(args.out_prefix, exist_ok=True)
    cv2.imwrite(os.path.join(args.out_prefix, 'mask.png'), (m*255))
    cv2.imwrite(os.path.join(args.out_prefix, 'overlay.png'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print('Saved →', os.path.join(args.out_prefix, 'overlay.png'))

# ------------------------------ CLI -----------------------------------
def parse_args():
    p = argparse.ArgumentParser('PointPromptNet-B trainer (SAM2 as environment, 32x32 grid RL, predictor cache)')
    p.add_argument('--csv', type=str, default='episodes.csv')
    p.add_argument('--cache-dir', type=str, default='cache')
    p.add_argument('--cache-long', type=int, default=1024, help='long side during cache/eval/infer letterbox')
    p.add_argument('--cache-multiple', type=int, default=16, help='pad to multiple during cache/eval/infer letterbox')
    p.add_argument('--cache-scan-dirs', type=str, nargs='*', default=None)
    p.add_argument('--cache-exts', type=str, default='.jpg,.jpeg,.png,.webp')
    p.add_argument('--cache-recursive', action='store_true')
    p.add_argument('--sam2-cfg', type=str, required=False, default='sam2.1_hiera_s.yaml')
    p.add_argument('--sam2-ckpt', type=str, required=False, default='checkpoints/sam2.1_hiera_small.pt')
    p.add_argument('--dinov3-model-id', type=str, default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--all-epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=24)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-2)
    p.add_argument('--kmax_f', type=int, default=8)
    p.add_argument('--kmax_b', type=int, default=8)
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--eval-every', type=int, default=1)
    p.add_argument('--save-every', type=int, default=1)
    p.add_argument('--out-dir', type=str, default='outputs/ckpts')
    p.add_argument('--build-cache', action='store_true')
    p.add_argument('--train', action='store_true')
    p.add_argument('--val-with-sam2', action='store_true')
    p.add_argument('--val-samples', type=int, default=128)
    p.add_argument('--torch-compile', action='store_true')
    # --- H_bg_star 帶狀 band 與權重 ---
    p.add_argument('--hbg-alpha', type=float, default=0.5, help='weight of ring in H_bg_star')
    p.add_argument('--hbg-beta',  type=float, default=0.3, help='weight of outside in H_bg_star')
    p.add_argument('--hbg-ring-r', type=int, default=2, help='outer ring dilation radius (px at letterbox size)')
    p.add_argument('--hbg-band-dmin', type=int, default=32, help='>=0 to enable band gating; only distances in [dmin,dmax] keep bg targets')
    p.add_argument('--hbg-band-dmax', type=int, default=128, help='upper bound (px) for band gating')
    # RL 相關
    p.add_argument('--lambda-pg-start', type=float, default=3.0)
    p.add_argument('--lambda-pg-end',   type=float, default=1.0)

    p.add_argument('--entropy-beta-start', type=float, default=3e-3)
    p.add_argument('--entropy-beta-end',   type=float, default=5e-4)

    p.add_argument('--tau-start', type=float, default=1.3)     # 溫度取樣：大→小
    p.add_argument('--tau-end',   type=float, default=1.0)

    p.add_argument('--w-heat-start', type=float, default=0.8)  # 像素監督：小幅讓位給 RL
    p.add_argument('--w-heat-end',   type=float, default=0.5)

    p.add_argument('--iou-gamma', type=float, default=1.5)     # 獎勵非線性放大：R = IoU**gamma + bonus
    p.add_argument('--adv-norm', action='store_true', default=True)
    p.add_argument('--adv-scale-start', type=float, default=5.0)
    p.add_argument('--adv-scale-end',   type=float, default=1.0)

    p.add_argument('--sample-n-start', type=int, default=1)    # 前期多抽樣平均 → 訊號更穩
    p.add_argument('--sample-n-end',   type=int, default=1)
    
    p.add_argument('--neg-min-dist-px', type=int, default=32, help='penalize negative points whose distance-to-boundary < this (px at letterbox size); 0 disables')
    p.add_argument('--neg-close-penalty', type=float, default=-0.5, help='penalty added per too-close negative point')
    
    # 負點落在前景的懲罰
    p.add_argument('--neg-on-fg-penalty', type=float, default=-3,
                   help='penalty if a negative point lands on GT foreground (per-batch normalized)')
    p.add_argument('--neg-on-fg-thresh', type=float, default=0.75,
                   help='grid cell is considered FG if area fraction >= this threshold')
    
    # inference demo
    p.add_argument('--infer', action='store_true')
    p.add_argument('--target', type=str, default=None)
    p.add_argument('--target-mask', type=str, default=None)
    p.add_argument('--ref', type=str, default=None)
    p.add_argument('--ref-mask', type=str, default=None)
    p.add_argument('--ckpt', type=str, default=None)
    p.add_argument('--out-prefix', type=str, default='outputs/demo')
    # manifest
    p.add_argument('--manifest-train', type=str, default=None)
    p.add_argument('--manifest-val', type=str, default=None)
    p.add_argument('--coco20i-fold', type=int, default=0, choices=[0,1,2,3])
    p.add_argument('--episodes', type=int, default=10000)
    p.add_argument('--role', type=str, default='novel', choices=['novel','base'])
    p.add_argument('--shard-count', type=int, default=1)
    p.add_argument('--shard-idx', type=int, default=0)
    p.add_argument('--resume', type=str, default=None)
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.infer:
        run_infer(args)
    elif args.train or args.build_cache:
        if args.build_cache and args.cache_scan_dirs:
            ds = EpisodeDataset([], args.cache_dir, args.dinov3_model_id,
                                args.sam2_cfg, args.sam2_ckpt,
                                build_cache=False,
                                cache_long=args.cache_long,
                                cache_multiple=args.cache_multiple)
            exts = [e for e in args.cache_exts.split(',') if e.strip()]
            paths = scan_images(args.cache_scan_dirs, exts, recursive=args.cache_recursive)
            if args.shard_count < 1: raise ValueError("--shard-count 必須 >= 1")
            if not (0 <= args.shard_idx < args.shard_count):
                raise ValueError("--shard-idx 必須介於 [0, shard-count)")
            paths = sorted(paths)
            if args.shard_count > 1:
                paths = [p for i, p in enumerate(paths) if (i % args.shard_count) == args.shard_idx]
            for pth in tqdm(paths, desc=f"Caching images (shard {args.shard_idx}/{args.shard_count})", unit="img"):
                ds._build_one(pth)
            print(f"Cache built for {len(paths)} images in this shard → {args.cache_dir}")
        elif args.build_cache and (args.manifest_train or args.manifest_val):
            mlist = [p for p in [args.manifest_train, args.manifest_val] if p]
            build_cache_from_manifests(args.cache_dir, args.dinov3_model_id, args.sam2_cfg, args.sam2_ckpt,
                                        mlist, cache_long=args.cache_long, cache_multiple=args.cache_multiple)
        elif args.build_cache and not args.train:
            episodes = read_csv(args.csv)
            _ = EpisodeDataset(episodes, args.cache_dir, args.dinov3_model_id, args.sam2_ckpt,
                                args.sam2_ckpt, build_cache=True,
                                cache_long=args.cache_long, cache_multiple=args.cache_multiple)
        else:
            train_loop(args)
    else:
        print('Nothing to do. Use --build-cache, --train or --infer.')
