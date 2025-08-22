# runv3.py — Batch + Cache + COCO20i Test Interface + (保留 box / box+point 單圖推論)
# ---------------------------------------------------------------------------------
# 這版在你原本 runv3.py 的基礎上，做了三件事：
# 1) **保留** 原本單圖推論的 box / box+point 流程與參數（fused_points_sam2_dinov3）。
# 2) **新增** 以快取為核心的批次測試介面（--test + 兩份 CSV + --coco20i-fold --role）。
# 3) **新增** 影像快取建置（--build-cache），將 SAM2 encoder 輸出與 DINOv3 grid 寫入 .pt（每張一檔）。
#    測試時**不跑 encoder**，直接從快取回填（set_image_from_cache）。
#
# 用法（Examples）
# A) 建快取（一次）
#   python runv3.py \
#     --build-cache \
#     --manifest-test data/test_a.csv --manifest-test2 data/test_b.csv \
#     --cache-dir cache \
#     --sam2-cfg configs/sam2.1/sam2.1_hiera_s.yaml \
#     --sam2-ckpt checkpoints/sam2.1_hiera_small.pt \
#     --dinov3-model-id facebook/dinov3-vitb16-pretrain-lvd1689m \
#     --cache-long 1024
#
# B) COCO20i 測試（novel split, 兩份 CSV 會各自回報 mIoU 並印平均）
#   python runv3.py \
#     --test \
#     --manifest-test data/test_a.csv --manifest-test2 data/test_b.csv \
#     --coco20i-fold 0 --role novel \
#     --cache-dir cache --cache-long 1024 \
#     --alpha 0.5 --k-pos 4 --k-neg 4 --suppress 2 --tau 0.2
#
# C) 單圖推論（保留原本 box / box+point 功能）
#   python runv3.py \
#     --single \
#     --target path/to/target.jpg --ref path/to/ref.jpg --ref-mask path/to/ref_mask.png \
#     --sam2-cfg configs/sam2.1/sam2.1_hiera_s.yaml --sam2-ckpt checkpoints/sam2.1_hiera_small.pt \
#     --dinov3-model-id facebook/dinov3-vitb16-pretrain-lvd1689m \
#     --alpha 0.5 --k-pos 4 --k-neg 4 --suppress 2 \
#     --use-box-prompt --box-method largest --prompt-mode both \
#     --out-prefix outputs/sam2_dinov3_run

import os, csv, math, argparse, functools
from contextlib import nullcontext
from typing import List, Tuple, Optional

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn.functional as F

from transformers import AutoImageProcessor, AutoModel  # HF DINOv3

# ------------------------------ Globals ------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------ I/O utils ------------------------------

def load_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path)
    assert bgr is not None, f"Image not found: {path}"
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def load_gray(path: str) -> np.ndarray:
    g = cv2.imread(path, 0)
    assert g is not None, f"Mask not found: {path}"
    return g

def norm01(x: torch.Tensor) -> torch.Tensor:
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-6)

class AutocastCtx:
    def __init__(self, enabled: bool): self.enabled = enabled
    def __enter__(self):
        if self.enabled and DEVICE.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()
    def __exit__(self, exc_type, exc, tb): return False

def autocast_ctx(enabled: bool):
    # 與舊版 API 相容的簡易包裝
    return AutocastCtx(enabled)

# ------------------------------ Letterbox resize ------------------------------

def resize_letterbox_rgb(rgb: np.ndarray, fixed_size: int = 1024) -> tuple[np.ndarray, dict]:
    H, W = rgb.shape[:2]
    interp = cv2.INTER_AREA if (H >= fixed_size and W >= fixed_size) else cv2.INTER_CUBIC
    out = cv2.resize(rgb, (fixed_size, fixed_size), interpolation=interp)
    meta = dict(top=0, left=0, out_h=fixed_size, out_w=fixed_size, scale=1.0, orig_h=H, orig_w=W)
    return out, meta

def resize_letterbox_mask(mask: np.ndarray, fixed_size: int = 1024) -> np.ndarray:
    return cv2.resize(mask, (fixed_size, fixed_size), interpolation=cv2.INTER_NEAREST)

# ---------------- Monkey-patch SAM2 predictor cache I/O ----------------

def _monkey_patch_sam2_cache_io():
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except Exception:
        from sam2_image_predictor import SAM2ImagePredictor  # fallback alias

    if hasattr(SAM2ImagePredictor, "set_image_from_cache") and hasattr(SAM2ImagePredictor, "export_cache"):
        return

    def _as_hw_tuple(val):
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
            if 'numpy' in str(type(val)):
                v = np.array(val).flatten().tolist()
                if len(v) >= 2: return (int(v[-2]), int(v[-1]))
                if len(v) == 1:  return (int(v[0]), int(v[0]))
        except Exception:
            pass
        return None

    @torch.no_grad()
    def set_image_from_cache(self, cache: dict) -> None:
        assert isinstance(cache, dict), "cache must be a dict"
        img_key = None
        for k in ("image_embed", "image_embeddings", "image_features", "vision_feats"):
            if k in cache: img_key = k; break
        if img_key is None:
            raise KeyError("cache missing image embedding (image_embed / image_embeddings / ...)")

        image_embed = cache[img_key]
        high_res = cache.get("high_res_feats", None)
        if high_res is None:
            raise KeyError("cache missing 'high_res_feats'")

        dev = self.device if hasattr(self, 'device') else next(self.model.parameters()).device
        to_dev_fp32 = lambda x: (x if isinstance(x, torch.Tensor) else torch.as_tensor(x)).to(dev).to(torch.float32)

        if isinstance(image_embed, list): image_embed = image_embed[0]
        if isinstance(image_embed, torch.Tensor) and image_embed.dim() == 5 and image_embed.size(0) == 1:
            image_embed = image_embed[0]
        image_embed = to_dev_fp32(image_embed)
        high_res = [to_dev_fp32(t) for t in high_res]

        orig_hw = (_as_hw_tuple(cache.get("original_size"))
                   or _as_hw_tuple(cache.get("orig_size"))
                   or _as_hw_tuple(cache.get("original_size_hw")))
        if orig_hw is None:
            s = int(getattr(self.model, "image_size", 1024)); orig_hw = (s, s)

        self._features = {"image_embed": image_embed, "high_res_feats": high_res}
        self._orig_hw = [tuple(orig_hw)]
        self._is_image_set = True
        self._is_batch = False

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
        if not getattr(self, "_is_image_set", False) or getattr(self, "_features", None) is None:
            raise RuntimeError("Call set_image(...) before export_cache().")
        def to_cpu_fp16(x: torch.Tensor): return x.detach().to("cpu").contiguous().to(torch.float16)
        pack = {"image_embed": to_cpu_fp16(self._features["image_embed"]) }
        hrs = self._features.get("high_res_feats", None)
        assert isinstance(hrs, list) and len(hrs) > 0, "high_res_feats missing"
        pack["high_res_feats"] = [to_cpu_fp16(t) for t in hrs]
        if isinstance(getattr(self, "_orig_hw", None), list) and len(self._orig_hw) > 0:
            ohw = self._orig_hw[0]
        else:
            s = int(getattr(self.model, "image_size", 1024)); ohw = (s, s)
        pack["original_size"] = tuple(ohw)
        pack["orig_size"]     = tuple(ohw)
        pinp = getattr(getattr(self, "_transforms", None), "padded_input_image_size", None)
        pinp = _as_hw_tuple(pinp) or ohw
        pack["input_size"] = tuple(pinp)
        pack["padded_input_image_size"] = tuple(pinp)
        pack["meta"] = {
            "sizes_saved_as_tuple": True,
            "original_size_hw": tuple(ohw),
            "input_size_hw": tuple(pinp),
        }
        return pack

    SAM2ImagePredictor.set_image_from_cache = set_image_from_cache
    SAM2ImagePredictor.export_cache = export_cache

_monkey_patch_sam2_cache_io()

# ------------------------------ SAM2 builder ------------------------------

def sam2_build_image_predictor(cfg_yaml: str, ckpt_path: str):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model = build_sam2(cfg_yaml, ckpt_path).to(DEVICE)
    return SAM2ImagePredictor(model)

# ------------------------------ DINOv3 (HF) ------------------------------
_DINOV3_CACHE = {}

def _get_dinov3_hf(model_id: str):
    if model_id not in _DINOV3_CACHE:
        proc = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(DEVICE)
        model.eval()
        _DINOV3_CACHE[model_id] = (proc, model)
    return _DINOV3_CACHE[model_id]

@torch.inference_mode()
def get_grid_feats_dinov3_hf(image_rgb: np.ndarray, model_id: str) -> torch.Tensor:
    proc, model = _get_dinov3_hf(model_id)
    pil = Image.fromarray(image_rgb)

    inputs = proc(images=pil, return_tensors="pt", do_center_crop=False, do_resize=False)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    out = model(**inputs)
    x = out.last_hidden_state
    if x.dim() == 4:  # CNN-like
        grid = x[0].permute(1,2,0).contiguous()
        return F.normalize(grid, dim=-1).to(torch.float32)
    Hp, Wp = inputs["pixel_values"].shape[-2:]
    psize = int(getattr(getattr(model, "config", None), "patch_size", 16))
    Gh, Gw = Hp//psize, Wp//psize
    M = Gh*Gw
    toks = x[0, -M:, :]
    grid = toks.view(Gh, Gw, -1).contiguous()
    return F.normalize(grid, dim=-1).to(torch.float32)  # [Gh,Gw,C]

# ------------------------------ Cache builder ------------------------------

def cache_key(img_path: str) -> str:
    import hashlib
    p = os.path.abspath(img_path)
    h = hashlib.sha1(p.encode('utf-8')).hexdigest()[:16]
    stem = os.path.splitext(os.path.basename(img_path))[0]
    return f"{stem}__{h}"

def build_cache_from_manifests(cache_dir: str, dinov3_id: str, sam2_cfg: str, sam2_ckpt: str,
                               manifest_paths: List[str], cache_long: int = 1024):
    os.makedirs(cache_dir, exist_ok=True)
    predictor = sam2_build_image_predictor(sam2_cfg, sam2_ckpt)

    # collect unique image paths from csv(s)
    paths = set()
    for mp in manifest_paths:
        if not mp: continue
        with open(mp, 'r', newline='') as f:
            r = csv.DictReader(f)
            for row in r:
                if 'image_path' in row and row['image_path']:
                    paths.add(row['image_path'])
                for k in ('target','ref'):
                    if k in row and row[k]:
                        paths.add(row[k])
    paths = sorted(paths)
    print(f"[cache] Will build {len(paths)} images → {cache_dir}")

    for pth in paths:
        key = cache_key(pth)
        out = os.path.join(cache_dir, f"{key}.pt")
        if os.path.exists(out):
            continue
        rgb_raw = load_rgb(pth)
        rgb_lb, meta = resize_letterbox_rgb(rgb_raw, cache_long)
        # SAM2 cache
        with AutocastCtx(True):
            predictor.set_image(rgb_lb)
        sam2_feats = predictor.export_cache()
        # DINO grid
        dng = get_grid_feats_dinov3_hf(rgb_lb, dinov3_id).permute(2,0,1)  # [Cd,Gh,Gw]
        meta_out = {
            "out_h": int(meta['out_h']), "out_w": int(meta['out_w']),
            "top": 0, "left": 0, "scale": 1.0,
            "orig_h": int(meta['orig_h']), "orig_w": int(meta['orig_w']),
        }
        data = {"sam2": sam2_feats, "dng": dng.half().cpu(), "meta": meta_out}
        torch.save(data, out)

    print(f"[cache] Done. Built {len(paths)} images")

# ------------------------------ Cached feature helpers ------------------------------

def _sam2_embed_from_cached(feats: dict) -> torch.Tensor:
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
    return F.normalize(x.to(torch.float32), dim=0)  # [C,Hf,Wf]

@torch.inference_mode()
def compute_proto_and_sims_from_cache(ref_pt: str, ref_mask: np.ndarray,
                                      tgt_pt: str, cache_dir: str,
                                      use_bg_proto: bool=True, tau: float=0.2):
    def _load_pt(pth):
        key = cache_key(pth)
        data = torch.load(os.path.join(cache_dir, f"{key}.pt"), map_location="cpu")
        sam2 = data["sam2"]; dng = data["dng"].to(torch.float32)
        sam = _sam2_embed_from_cached(sam2)
        return sam, dng, data.get("meta", {}), sam2

    sam_r, dn_r, meta_r, _ = _load_pt(ref_pt)
    sam_t, dn_t, meta_t, _ = _load_pt(tgt_pt)

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
        proto_bg_sam = (sam_r * bg.unsqueeze(0)).sum((1,2))/(bg.sum()+1e-8)
        proto_bg_sam = F.normalize(proto_bg_sam, dim=0)
        sim_sam = torch.sigmoid((sim_fg_sam - (sam_t * proto_bg_sam.view(-1,1,1)).sum(0))/tau)
    else:
        sim_sam = (sim_fg_sam+1.0)/2.0

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
        "sam_feat_tgt": sam_t,                 # [Cs,Hf,Wf]
        "dino_feat_tgt": dn_t.permute(1,2,0),  # [Gh,Gw,Cd]
        "sim_sam": sim_sam,                    # [Hf,Wf]
        "sim_dino": sim_dn,                    # [Gh,Gw]
        "meta_t": meta_t
    }

# ------------------------------ 相似度 → 點/框 提示 ------------------------------

def pick_points_from_sim(sim_map: torch.Tensor, orig_size, k_pos=4, k_neg=4, suppress=2):
    Hf, Wf = sim_map.shape
    sim = sim_map.clone()
    pos = []
    for _ in range(int(k_pos)):
        idx = torch.argmax(sim)
        r = int(idx // Wf); c = int(idx % Wf)
        pos.append((r, c))
        r0, r1 = max(0, r - suppress), min(Hf, r + suppress + 1)
        c0, c1 = max(0, c - suppress), min(Wf, c + suppress + 1)
        sim[r0:r1, c0:c1] = -1.0

    low = sim_map.clone()
    for (r, c) in pos:
        r0, r1 = max(0, r - suppress), min(Hf, r + suppress + 1)
        c0, c1 = max(0, c - suppress), min(Wf, c + suppress + 1)
        low[r0:r1, c0:c1] = 1.0
    neg = []
    flat = low.view(-1)
    for _ in range(int(k_neg)):
        idx = torch.argmin(flat)
        r = int(idx // Wf); c = int(idx % Wf)
        neg.append((r, c))
        r0, r1 = max(0, r - suppress), min(Hf, r + suppress + 1)
        c0, c1 = max(0, c - suppress), min(Wf, c + suppress + 1)
        low[r0:r1, c0:c1] = 1.0
        flat = low.view(-1)

    H, W = orig_size
    def grid2xy(rc):
        r, c = rc
        x = int((c + 0.5) * W / Wf)
        y = int((r + 0.5) * H / Hf)
        return [x, y]

    pos_xy = np.array([grid2xy(rc) for rc in pos], dtype=np.int32)
    neg_xy = np.array([grid2xy(rc) for rc in neg], dtype=np.int32)
    pts = np.vstack([pos_xy, neg_xy]) if k_neg > 0 else pos_xy
    labels = np.array([1] * len(pos_xy) + [0] * len(neg_xy), dtype=np.int32)
    return pts, labels, pos_xy, neg_xy


def boxes_from_sim(sim_map: torch.Tensor, orig_size,
                   method="largest", k=1,
                   thresh: float = None, percentile: float = 0.9,
                   min_area: int = 256, expand: float = 0.02) -> np.ndarray:
    H, W = orig_size
    sim_up = torch.nn.functional.interpolate(
        sim_map.unsqueeze(0).unsqueeze(0).to(dtype=torch.float32),
        size=(H, W), mode="bicubic", align_corners=False
    ).squeeze(0).squeeze(0).cpu().numpy().astype("float32")

    t = float(np.quantile(sim_up, percentile)) if (thresh is None) else float(thresh)
    mask = (sim_up >= t).astype("uint8") * 255

    # 開關小區域
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    cand = []
    for i in range(1, n):
        x, y, w, h, a = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3], stats[i, 4]
        if a < int(min_area):
            continue
        cand.append((a, x, y, w, h))

    cand.sort(key=lambda z: z[0], reverse=True)
    if len(cand) == 0:
        return np.zeros((0,4), dtype=np.float32)

    if method == "largest":
        cand = cand[:1]
    elif method == "topk":
        cand = cand[:max(1, int(k))]
    else:
        raise ValueError(f"Unknown method: {method}")

    boxes = []
    dx = int(round(W * float(expand))); dy = int(round(H * float(expand)))
    for _, x, y, w, h in cand:
        x1 = max(0, x - dx); y1 = max(0, y - dy)
        x2 = min(W - 1, x + w + dx); y2 = min(H - 1, y + h + dy)
        boxes.append([x1, y1, x2, y2])
    return np.array(boxes, dtype=np.float32)

# ------------------------------ SAM2 單圖推論（points / boxes / both） ------------------------------

def sam2_predict_with_points(predictor, image_rgb, points_xy, labels, out_prefix, use_autocast=False):
    pts = points_xy.astype(np.float32); lbs = labels.astype(np.int32)
    with autocast_ctx(use_autocast):
        masks, scores, _ = predictor.predict(point_coords=pts, point_labels=lbs, multimask_output=True)

    best = int(np.argmax(scores)); mask = masks[best].astype(np.uint8)
    overlay = image_rgb.copy()
    overlay[mask > 0] = (overlay[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    for (x, y), lb in zip(pts.astype(int), lbs):
        color = (0, 255, 0) if lb == 1 else (255, 0, 0)
        cv2.circle(overlay, (int(x), int(y)), 6, color, -1)

    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    mpath = f"{out_prefix}_mask.png"; opath = f"{out_prefix}_overlay.png"
    cv2.imwrite(mpath, (mask * 255)); cv2.imwrite(opath, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return mpath, opath, float(scores[best])


def sam2_predict_with_points_and_boxes(predictor, image_rgb, points_xy, labels, boxes_xyxy,
                                       out_prefix, use_autocast=False, draw=True):
    pts = None if points_xy is None else points_xy.astype(np.float32)
    lbs = None if labels is None else labels.astype(np.int32)

    best = {"score": -1e9, "mask": None}
    for box in boxes_xyxy.astype(np.float32):
        with autocast_ctx(use_autocast):
            masks, scores, _ = predictor.predict(point_coords=pts, point_labels=lbs, box=box, multimask_output=True)
        j = int(np.argmax(scores))
        if float(scores[j]) > best["score"]:
            best["score"] = float(scores[j])
            best["mask"] = masks[j].astype(np.uint8)

    mask = best["mask"]; overlay = image_rgb.copy()
    overlay[mask > 0] = (overlay[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    if pts is not None and lbs is not None and draw:
        for (x, y), lb in zip(pts.astype(int), lbs):
            color = (0, 255, 0) if lb == 1 else (255, 0, 0)
            cv2.circle(overlay, (int(x), int(y)), 6, color, -1)
    if draw:
        for (x1, y1, x2, y2) in boxes_xyxy.astype(int):
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)

    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    mpath = f"{out_prefix}_mask.png"; opath = f"{out_prefix}_overlay.png"
    cv2.imwrite(mpath, (mask * 255)); cv2.imwrite(opath, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return mpath, opath, best["score"]

# ------------------------------ 後處理（可選） ------------------------------

def apply_densecrf(image_rgb: np.ndarray, prob_map: torch.Tensor,
                   n_iters: int = 5,
                   sxy_gaussian: int = 3, compat_gaussian: int = 3,
                   sxy_bilateral: int = 80, srgb: int = 13, compat_bilateral: int = 5) -> torch.Tensor:
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except Exception as e:
        print("[CRF] pydensecrf not installed, skip.")
        return prob_map

    img = image_rgb
    H, W = img.shape[:2]
    p = prob_map.detach().cpu().float().clamp(0,1).numpy()
    U = unary_from_softmax(np.stack([1.0-p, p], axis=0))
    d = dcrf.DenseCRF2D(W, H, 2)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=(sxy_gaussian, sxy_gaussian), compat=int(compat_gaussian))
    d.addPairwiseBilateral(sxy=(sxy_bilateral, sxy_bilateral), srgb=(int(srgb), int(srgb), int(srgb)), rgbim=img, compat=int(compat_bilateral))
    Q = d.inference(int(n_iters))
    q_fg = np.asarray(Q)[1].reshape(H, W).astype(np.float32)
    q_fg = np.clip(q_fg, 0.0, 1.0)
    return torch.from_numpy(q_fg)


def apply_stability_filter(prob_map: torch.Tensor,
                           thresholds=(0.4, 0.5, 0.6),
                           open_area: int = 256,
                           close_area: int = 256,
                           sharpen_hi: float = 0.9,
                           sharpen_lo: float = 0.1) -> torch.Tensor:
    p = prob_map.detach().cpu().float().clamp(0, 1).numpy()
    H, W = p.shape
    stable_fg = np.ones_like(p, dtype=bool); stable_bg = np.ones_like(p, dtype=bool)
    for t in thresholds:
        stable_fg &= (p >= float(t)); stable_bg &= (p <= float(1.0 - t))

    def remove_small_components(mask, min_area):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        keep = np.zeros_like(mask, dtype=bool)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area: keep |= (labels == i)
        return keep

    def fill_small_holes(mask, max_area):
        inv = ~mask
        n, labels, stats, _ = cv2.connectedComponentsWithStats(inv.astype(np.uint8), 8)
        hole = np.zeros_like(mask, dtype=bool)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] <= max_area: hole |= (labels == i)
        return mask | hole

    bin05 = (p >= 0.5)
    bin05 = remove_small_components(bin05, open_area)
    bin05 = fill_small_holes(bin05, close_area)

    p_new = p.copy()
    p_new[stable_fg] = np.maximum(p_new[stable_fg], sharpen_hi)
    p_new[stable_bg] = np.minimum(p_new[stable_bg], sharpen_lo)
    p_new[bin05] = np.maximum(p_new[bin05], 0.75)
    p_new[~bin05] = np.minimum(p_new[~bin05], 0.25)
    return torch.from_numpy(p_new.astype(np.float32))

# ------------------------------ SAM2 encoder 相似度（單圖路線） ------------------------------

@torch.inference_mode()
def _extract_sam2_image_embed(predictor) -> torch.Tensor:
    # 從已 set_image 的 predictor 中抓 image embedding + high_res pyramid，合成 [C,Hf,Wf]
    feats = getattr(predictor, "_features", None)
    assert feats is not None, "Call predictor.set_image(...) first."
    if "image_embed" in feats:
        x = feats["image_embed"]
    elif "image_embeddings" in feats:
        x = feats["image_embeddings"]
    else:
        raise KeyError("predictor._features lacks image embedding")
    if isinstance(x, list): x = x[0]
    if x.dim() == 4 and x.size(0) == 1: x = x[0]
    return x.to(torch.float32)

@torch.inference_mode()
def imgenc_sim_map_sam2_encoder(target_rgb: np.ndarray,
                                ref_rgb: np.ndarray,
                                ref_mask_path: str,
                                predictor,
                                use_bg_proto: bool = False,
                                tau: float = 0.2,
                                use_autocast: bool = False) -> torch.Tensor:
    """用 SAM2 encoder 特徵 + ref_mask 建原型相似圖，回傳 [Hf,Wf]∈[0,1]"""
    with autocast_ctx(use_autocast):
        predictor.set_image(ref_rgb)
    ref_embed = _extract_sam2_image_embed(predictor)        # [C,Hf,Wf]

    with autocast_ctx(use_autocast):
        predictor.set_image(target_rgb)
    tgt_embed = _extract_sam2_image_embed(predictor)        # [C,Hf,Wf]

    C, Hf, Wf = ref_embed.shape
    mask = load_gray(ref_mask_path)
    m_small = cv2.resize(mask, (Wf, Hf), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    m_small = torch.from_numpy(m_small).to(device=tgt_embed.device, dtype=tgt_embed.dtype)
    fg = (m_small > 0.5).float()
    bg = (m_small <= 0.5).float()

    ref_n = F.normalize(ref_embed, dim=0)
    proto_fg = (ref_n * fg.unsqueeze(0)).sum((1, 2)) / (fg.sum() + 1e-8)
    proto_fg = F.normalize(proto_fg, dim=0)

    tgt_n = F.normalize(tgt_embed, dim=0)
    sim_fg = (tgt_n * proto_fg.view(-1, 1, 1)).sum(0)  # [-1,1]
    if not use_bg_proto:
        return (sim_fg + 1.0) / 2.0

    if bg.sum() < 1:
        inv = (1.0 - m_small).flatten()
        topk = torch.topk(inv, k=min(10, inv.numel()))[1]
        bg = torch.zeros_like(inv); bg[topk] = 1.0; bg = bg.view(Hf, Wf)

    proto_bg = (ref_n * bg.unsqueeze(0)).sum((1, 2)) / (bg.sum() + 1e-8)
    proto_bg = F.normalize(proto_bg, dim=0)
    sim_bg = (tgt_n * proto_bg.view(-1, 1, 1)).sum(0)
    return torch.sigmoid((sim_fg - sim_bg) / float(tau))

# ------------------------------ 單圖主流程（保留 box/box+point） ------------------------------

def fused_points_sam2_dinov3(
    target_path, ref_path, ref_mask_path,
    # DINOv3
    dinov3_model_id="facebook/dinov3-vitb16-pretrain-lvd1689m",
    use_bg_proto=False, tau=0.2,
    # SAM2
    sam2_cfg="configs/sam2.1/sam2.1_hiera_s.yaml", sam2_ckpt="checkpoints/sam2.1_hiera_small.pt",
    # 融合
    alpha=0.5,
    # prompts
    k_pos=4, k_neg=4, suppress=2,
    use_box_prompt=False,
    box_method="largest", box_k=1, box_thresh=None, box_percentile=0.9,
    box_min_area=256, box_expand=0.02,
    prompt_mode="both",
    # 後處理（可選）
    post_crf=False, crf_from="sim", crf_iters=5,
    crf_sxy_gaussian=3, crf_compat_gaussian=3,
    crf_sxy_bilateral=80, crf_srgb=13, crf_compat_bilateral=5,
    stability_filter=False, stable_thresholds=(0.4,0.5,0.6),
    stable_open_area=256, stable_close_area=256,
    # 其他
    use_autocast=False,
    # 輸出
    out_prefix="outputs/sam2_dinov3_run"
):
    tgt_rgb = load_rgb(target_path)
    ref_rgb = load_rgb(ref_path)

    # 0) SAM2 predictor
    predictor = sam2_build_image_predictor(sam2_cfg, sam2_ckpt)

    # 1) SAM2 encoder sim
    sim_sam2 = imgenc_sim_map_sam2_encoder(
        tgt_rgb, ref_rgb, ref_mask_path,
        predictor=predictor, use_bg_proto=use_bg_proto, tau=tau, use_autocast=use_autocast
    )  # [Hf,Wf]

    # 2) DINOv3 grid sim（HF）
    sim_dino = None
    try:
        tgt_grid = get_grid_feats_dinov3_hf(tgt_rgb, dinov3_model_id)   # [Gh,Gw,C]
        ref_grid = get_grid_feats_dinov3_hf(ref_rgb, dinov3_model_id)
        mask = load_gray(ref_mask_path)
        Gh, Gw = ref_grid.shape[:2]
        m_small = cv2.resize(mask, (Gw, Gh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        m_small = torch.from_numpy(m_small).to(dtype=tgt_grid.dtype, device=tgt_grid.device)
        fg = (m_small > 0.5).to(dtype=tgt_grid.dtype)
        bg = (m_small <= 0.5).to(dtype=tgt_grid.dtype)
        proto_fg = (ref_grid * fg.unsqueeze(-1)).sum((0,1)) / (fg.sum() + 1e-8)
        proto_fg = F.normalize(proto_fg, dim=0)
        sim_fg = (tgt_grid * proto_fg.view(1,1,-1)).sum(-1)
        if not use_bg_proto:
            sim_dino = (sim_fg + 1.0) / 2.0
        else:
            proto_bg = (ref_grid * bg.unsqueeze(-1)).sum((0,1)) / (bg.sum() + 1e-8)
            proto_bg = F.normalize(proto_bg, dim=0)
            sim_bg = (tgt_grid * proto_bg.view(1,1,-1)).sum(-1)
            sim_dino = torch.sigmoid((sim_fg - sim_bg) / float(tau))
    except Exception as e:
        print("[DINOv3] HF feature failed, fallback to SAM2-only fusion (alpha→0)")
        sim_dino = torch.zeros_like(sim_sam2)
        alpha = 0.0

    # 3) 融合與可視化
    H, W = tgt_rgb.shape[:2]
    heat_sam2 = cv2.applyColorMap((cv2.resize(sim_sam2.detach().cpu().float().numpy(), (W, H)) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat_dino = cv2.applyColorMap((cv2.resize(sim_dino.detach().cpu().float().numpy(), (W, H)) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    cv2.imwrite(f"{out_prefix}_sim_sam2_heat.png", heat_sam2)
    cv2.imwrite(f"{out_prefix}_sim_dino_heat.png", heat_dino)

    Hf, Wf = sim_sam2.shape
    sim_dino_up = F.interpolate(sim_dino[None, None, ...], size=(Hf, Wf), mode="bicubic", align_corners=False).squeeze(0).squeeze(0)
    sim_fused = norm01(float(alpha) * norm01(sim_dino_up) + (1.0 - float(alpha)) * norm01(sim_sam2))

    heat_fused = cv2.applyColorMap((cv2.resize(sim_fused.detach().cpu().float().numpy(), (W, H)) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(f"{out_prefix}_sim_fused_heat.png", heat_fused)

    # 4) 取 prompts（點/盒）
    pts = labels = None
    if prompt_mode in ("points", "both"):
        pts, labels, pos_xy, neg_xy = pick_points_from_sim(sim_fused, (H, W), k_pos, k_neg, suppress)
        print("pos:", pos_xy.tolist(), "| neg:", neg_xy.tolist())

    boxes_xyxy = None
    if use_box_prompt and (prompt_mode in ("boxes", "both")):
        boxes_xyxy = boxes_from_sim(
            sim_fused, (H, W), method=box_method, k=box_k,
            thresh=box_thresh, percentile=box_percentile,
            min_area=box_min_area, expand=box_expand
        )
        print("boxes:", boxes_xyxy.tolist())

    # 5) SAM2 分割（再保險 set_image target 一次）
    with autocast_ctx(use_autocast):
        predictor.set_image(tgt_rgb)

    if (boxes_xyxy is not None) and (prompt_mode in ("boxes", "both")):
        mpath, opath, score = sam2_predict_with_points_and_boxes(
            predictor, tgt_rgb, pts, labels, boxes_xyxy, out_prefix, use_autocast=use_autocast
        )
    else:
        if pts is None:  # fallback：一定至少用點
            pts, labels, _, _ = pick_points_from_sim(sim_fused, (H, W), k_pos, k_neg, suppress)
        mpath, opath, score = sam2_predict_with_points(
            predictor, tgt_rgb, pts, labels, out_prefix, use_autocast=use_autocast
        )

    print("Saved:", mpath, opath, "| best score:", score)

    # 6) （選配）CRF / 穩定性加權
    if post_crf:
        if crf_from == "sim":
            p = cv2.resize(sim_fused.detach().cpu().float().numpy(), (W, H), interpolation=cv2.INTER_CUBIC)
            p = torch.from_numpy(p)
        else:
            p = torch.from_numpy((cv2.imread(f"{out_prefix}_mask.png", 0) / 255.0).astype(np.float32))
        p_crf = apply_densecrf(tgt_rgb, p,
                               n_iters=crf_iters,
                               sxy_gaussian=crf_sxy_gaussian, compat_gaussian=crf_compat_gaussian,
                               sxy_bilateral=crf_sxy_bilateral, srgb=crf_srgb, compat_bilateral=crf_compat_bilateral)
        overlay = tgt_rgb.copy(); mask_crf = (p_crf.numpy() >= 0.5).astype(np.uint8)
        overlay[mask_crf > 0] = (overlay[mask_crf > 0] * 0.5 + np.array([0,255,0]) * 0.5).astype(np.uint8)
        cv2.imwrite(f"{out_prefix}_mask_crf.png", (mask_crf*255))
        cv2.imwrite(f"{out_prefix}_overlay_crf.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print("CRF refined mask saved:", f"{out_prefix}_mask_crf.png", f"{out_prefix}_overlay_crf.png")

    if stability_filter:
        p = cv2.resize(sim_fused.detach().cpu().float().numpy(), (W, H), interpolation=cv2.INTER_CUBIC)
        p = torch.from_numpy(p)
        p = apply_stability_filter(p, thresholds=stable_thresholds,
                                   open_area=stable_open_area, close_area=stable_close_area)
        overlay = tgt_rgb.copy(); mask_s = (p.numpy() >= 0.5).astype(np.uint8)
        overlay[mask_s > 0] = (overlay[mask_s > 0] * 0.5 + np.array([0,255,0]) * 0.5).astype(np.uint8)
        cv2.imwrite(f"{out_prefix}_mask_stable.png", (mask_s*255))
        cv2.imwrite(f"{out_prefix}_overlay_stable.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print("Stability filter saved:", f"{out_prefix}_mask_stable.png")

    return mpath, opath, score

# ------------------------------ COCO20i evaluation (from cache) ------------------------------

# 期待你環境已有 dataloader_ps（或你專案內相對應模組）
try:
    from dataloader_ps import build_psv2_index as build_coco20i_index
    from dataloader_ps import OneShotPSV2RoundRobin as OneShotCOCO20iRoundRobin
except Exception:
    build_coco20i_index = None
    OneShotCOCO20iRoundRobin = None

@torch.inference_mode()
def evaluate_manifest(args, manifest_path: str) -> float:
    if build_coco20i_index is None or OneShotCOCO20iRoundRobin is None:
        raise ImportError("dataloader_ps not available. Please ensure build_psv2_index and OneShotPSV2RoundRobin can be imported.")

    idx = build_coco20i_index(manifest_path, fold=args.coco20i_fold, role=args.role)
    ds = OneShotCOCO20iRoundRobin(index=idx, seed=2025, shuffle_classes=True)

    predictor = sam2_build_image_predictor(args.sam2_cfg, args.sam2_ckpt)
    ious = []

    max_eps = args.test_episodes if args.test_episodes > 0 else len(ds)
    print(f"Evaluating {min(len(ds), max_eps)} episodes from {manifest_path} ...")
    
    for epi_idx in range(min(len(ds), max_eps)):
        ep = ds[epi_idx]
        sup = ep['support']; qry = ep['query']
        sup_img = (sup['image'].numpy().transpose(1,2,0) * 255.0).astype(np.uint8)
        qry_img = (qry['image'].numpy().transpose(1,2,0) * 255.0).astype(np.uint8)
        sup_mask = (sup['mask'].numpy().astype(np.uint8) * 255)
        qry_mask = (qry['mask'].numpy().astype(np.uint8) * 255)
        sup_path = sup['meta']['image_path']; qry_path = qry['meta']['image_path']

        # ---- sims from cache (no encoder) ----
        feats = compute_proto_and_sims_from_cache(sup_path, sup_mask, qry_path, args.cache_dir, use_bg_proto=True, tau=args.tau)

        # ---- fuse sims (DINO upsampled to SAM2 map size) ----
        Hf, Wf = feats['sim_sam'].shape
        sim_dn_up = F.interpolate(feats['sim_dino'].unsqueeze(0).unsqueeze(0), size=(Hf,Wf), mode='bicubic', align_corners=False).squeeze(0).squeeze(0)
        sim_fused = norm01(float(args.alpha) * norm01(sim_dn_up) + (1.0 - float(args.alpha)) * norm01(feats['sim_sam']))

        # ---- prompts (points / boxes / both) for TEST ----
        H_img = W_img = int(args.cache_long)
        pts_xy = labels = None
        if args.prompt_mode in ('points','both'):
            pts_xy, labels, _, _ = pick_points_from_sim(sim_fused, (H_img, W_img), k_pos=args.k_pos, k_neg=args.k_neg, suppress=args.suppress)

        boxes_xyxy = None
        if args.use_box_prompt and (args.prompt_mode in ('boxes','both')):
            boxes_xyxy = boxes_from_sim(
                sim_fused, (H_img, W_img), method=args.box_method, k=args.box_k,
                thresh=args.box_thresh, percentile=args.box_percentile,
                min_area=args.box_min_area, expand=args.box_expand
            )
            if boxes_xyxy.shape[0] == 0:
                boxes_xyxy = None

        # ---- decode with predictor from cache (no encoder) ----
        key = cache_key(qry_path)
        pt_path = os.path.join(args.cache_dir, f"{key}.pt")
        if not os.path.isfile(pt_path):
            ious.append(0.0); continue
        data = torch.load(pt_path, map_location="cpu")
        predictor.set_image_from_cache(data["sam2"])  # <- no encoder

        # ensure transforms accept our sizes
        _force_size_attrs(predictor, (H_img, W_img))
        _patch_predictor_transforms(predictor)

        # predict according to chosen prompt mode
        def _predict_mask_with_optional_box_and_points(pts_xy_in, labels_in):
            pts_local = pts_xy_in
            labels_local = labels_in
        
            if (boxes_xyxy is not None) and (args.prompt_mode in ('boxes','both')):
                # 逐 box（可帶 points）取分數最高的遮罩
                best_score = -1e9
                best_mask = None
                for box in boxes_xyxy.astype(np.float32):
                    masks, scores, _ = predictor.predict(
                        point_coords=None if pts_local is None else pts_local.astype(np.float32),
                        point_labels=None if labels_local is None else labels_local.astype(np.int32),
                        box=box, multimask_output=True
                    )
                    j = int(np.argmax(scores))
                    sc = float(scores[j])
                    if sc > best_score:
                        best_score = sc
                        best_mask = masks[j].astype(np.uint8)
                return best_mask
            else:
                # points-only 或 fallback 產生至少 1 個正點
                if pts_local is None:
                    pts_local, labels_local, _, _ = pick_points_from_sim(
                        sim_fused, (H_img, W_img),
                        k_pos=max(1, args.k_pos), k_neg=0, suppress=args.suppress
                    )
                masks, scores, _ = predictor.predict(
                    point_coords=pts_local.astype(np.float32),
                    point_labels=labels_local.astype(np.int32),
                    multimask_output=True
                )
                j = int(np.argmax(scores))
                return masks[j].astype(np.uint8)
        

        m = _predict_mask_with_optional_box_and_points(pts_xy, labels)

        gt_lb = resize_letterbox_mask(qry_mask, args.cache_long)
        gt = (gt_lb > 127).astype(np.uint8)
        inter = (m > 0) & (gt > 0)
        union = (m > 0) | (gt > 0)
        iou = float(inter.sum()) / float(union.sum() + 1e-6)
        ious.append(iou)

        if args.save_preds:
            os.makedirs(args.save_dir, exist_ok=True)
            base = os.path.join(args.save_dir, f"{epi_idx:06d}")
            overlay = cv2.cvtColor((qry_img if qry_img.shape[:2]==(H_img,W_img) else cv2.resize(qry_img,(H_img,W_img))), cv2.COLOR_RGB2BGR)
            overlay[m>0] = (overlay[m>0] * 0.5 + np.array([0,255,0]) * 0.5).astype(np.uint8)
            cv2.imwrite(base + "_mask.png", (m*255))
            cv2.imwrite(base + "_overlay.png", overlay)
            if boxes_xyxy is not None:
                ov2 = overlay.copy()
                for (x1,y1,x2,y2) in (boxes_xyxy.astype(int) if boxes_xyxy is not None else []):
                    cv2.rectangle(ov2, (x1,y1), (x2,y2), (0,255,255), 2)
                cv2.imwrite(base + "_overlay_boxes.png", ov2)

    miou = float(np.mean(ious)) if ious else 0.0
    print(f"[Test] {os.path.basename(manifest_path)} → episodes={len(ious)}  mIoU={miou:.4f}")
    return miou

# ------------------------------ Predictor helpers ------------------------------

def _force_size_attrs(predictor, hw_tuple):
    H, W = int(hw_tuple[0]), int(hw_tuple[1])
    hw = (H, W)
    try:
        predictor._orig_hw = [hw]
    except Exception:
        pass
    for name in ("original_size", "orig_size", "_original_size"):
        try: setattr(predictor, name, hw)
        except Exception: pass
    for name in ("input_size", "_input_size", "padded_input_image_size"):
        try: setattr(predictor, name, hw)
        except Exception: pass

def _patch_predictor_transforms(predictor):
    tr = getattr(predictor, "_transforms", None)
    if tr is None or not hasattr(tr, "transform_coords"):
        return
    if getattr(tr, "_patched_accepts_kwargs", False):
        return
    old = tr.transform_coords
    def _as_hw_tuple(val):
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
            if 'numpy' in str(type(val)):
                v = np.array(val).flatten().tolist()
                if len(v) >= 2: return (int(v[-2]), int(v[-1]))
                if len(v) == 1:  return (int(v[0]), int(v[0]))
        except Exception:
            pass
        return None
    @functools.wraps(old)
    def wrapped(coords, *args, **kwargs):
        if 'orig_hw' in kwargs:
            kwargs['orig_hw'] = _as_hw_tuple(kwargs['orig_hw']) or kwargs['orig_hw']
            return old(coords, *args, **kwargs)
        if len(args) >= 1:
            ohw = _as_hw_tuple(args[0]) or args[0]
            args = (ohw,) + args[1:]
        return old(coords, *args, **kwargs)
    tr.transform_coords = wrapped
    tr._patched_accepts_kwargs = True

# ------------------------------ CLI ------------------------------

def parse_args():
    p = argparse.ArgumentParser('runv3: Batch+Cache testing for COCO20i manifests + single-image (points/boxes/both)')

    # Cache & models
    p.add_argument('--cache-dir', type=str, default='cache')
    p.add_argument('--cache-long', type=int, default=1024)
    p.add_argument('--sam2-cfg', type=str, default='configs/sam2.1/sam2.1_hiera_s.yaml')
    p.add_argument('--sam2-ckpt', type=str, default='checkpoints/sam2.1_hiera_small.pt')
    p.add_argument('--dinov3-model-id', type=str, default='facebook/dinov3-vitb16-pretrain-lvd1689m')

    # Test manifests (COCO20i)
    p.add_argument('--manifest-test', type=str, default=None, help='CSV for test set A (COCO20i-style)')
    p.add_argument('--manifest-test2', type=str, default=None, help='CSV for test set B (optional)')
    p.add_argument('--coco20i-fold', type=int, default=0, choices=[0,1,2,3])
    p.add_argument('--role', type=str, default='novel', choices=['novel','base'])
    p.add_argument('--test-episodes', type=int, default=500, help='max episodes to evaluate per manifest (<= dataset size)')

    # Single image mode
    p.add_argument('--single', action='store_true', help='Run single-image pipeline (keeps original box/box+point)')
    p.add_argument('--target', type=str, default=None)
    p.add_argument('--ref', type=str, default=None)
    p.add_argument('--ref-mask', type=str, default=None)

    # Prompting / fusion
    p.add_argument('--alpha', type=float, default=0.5, help='sim_fused = alpha * sim_dino + (1-alpha) * sim_sam2')
    p.add_argument('--k-pos', type=int, default=4)
    p.add_argument('--k-neg', type=int, default=4)
    p.add_argument('--suppress', type=int, default=2)
    p.add_argument('--tau', type=float, default=0.2)

    # Boxes (single-image)
    p.add_argument('--use-box-prompt', action='store_true')
    p.add_argument('--box-method', type=str, default='largest', choices=['largest','topk'])
    p.add_argument('--box-k', type=int, default=1)
    p.add_argument('--box-thresh', type=float, default=None)
    p.add_argument('--box-percentile', type=float, default=0.9)
    p.add_argument('--box-min-area', type=int, default=256)
    p.add_argument('--box-expand', type=float, default=0.02)
    p.add_argument('--prompt-mode', type=str, default='both', choices=['points','boxes','both'])

    # Optional post-processing (single-image)
    p.add_argument('--post-crf', action='store_true')
    p.add_argument('--crf-from', type=str, default='sim', choices=['sim','mask'])
    p.add_argument('--crf-iters', type=int, default=5)
    p.add_argument('--crf-sxy-gaussian', type=int, default=3)
    p.add_argument('--crf-compat-gaussian', type=int, default=3)
    p.add_argument('--crf-sxy-bilateral', type=int, default=80)
    p.add_argument('--crf-srgb', type=int, default=13)
    p.add_argument('--crf-compat-bilateral', type=int, default=5)
    p.add_argument('--stability-filter', action='store_true')
    p.add_argument('--stable-thresholds', type=float, nargs=3, default=(0.4,0.5,0.6))
    p.add_argument('--stable-open-area', type=int, default=256)
    p.add_argument('--stable-close-area', type=int, default=256)

    # Modes
    p.add_argument('--build-cache', action='store_true')
    p.add_argument('--test', action='store_true')

    # Outputs
    p.add_argument('--save-preds', action='store_true')
    p.add_argument('--save-dir', type=str, default='outputs/test_preds')
    p.add_argument('--out-prefix', type=str, default='outputs/sam2_dinov3_run')
    return p.parse_args()

# ------------------------------ Main ------------------------------
if __name__ == '__main__':
    args = parse_args()

    # A) Build cache
    if args.build_cache:
        mlist = [p for p in [args.manifest_test, args.manifest_test2] if p]
        if not mlist:
            raise ValueError('--build-cache specified but no --manifest-test / --manifest-test2 provided')
        build_cache_from_manifests(args.cache_dir, args.dinov3_model_id, args.sam2_cfg, args.sam2_ckpt, mlist, cache_long=args.cache_long)

    # B) Test (COCO20i manifests)
    if args.test:
        if not args.manifest_test and not args.manifest_test2:
            raise ValueError('Please provide at least one of --manifest-test or --manifest-test2')
        miou_list = []
        if args.manifest_test:
            miou_list.append(evaluate_manifest(args, args.manifest_test))
        if args.manifest_test2:
            miou_list.append(evaluate_manifest(args, args.manifest_test2))
        if len(miou_list) >= 2:
            print(f"[Test] Average of two manifests: mIoU={float(np.mean(miou_list)):.4f}")

    # C) Single-image pipeline (kept original box/box+point)
    if args.single:
        for name in ['target','ref','ref-mask']:
            if getattr(args, name.replace('-', '_')) is None:
                raise ValueError(f"--single 需要 --target/--ref/--ref-mask")
        fused_points_sam2_dinov3(
            target_path=args.target,
            ref_path=args.ref,
            ref_mask_path=args.ref_mask,
            dinov3_model_id=args.dinov3_model_id,
            use_bg_proto=False, tau=args.tau,
            sam2_cfg=args.sam2_cfg, sam2_ckpt=args.sam2_ckpt,
            alpha=args.alpha,
            k_pos=args.k_pos, k_neg=args.k_neg, suppress=args.suppress,
            use_box_prompt=args.use_box_prompt,
            box_method=args.box_method, box_k=args.box_k,
            box_thresh=args.box_thresh, box_percentile=args.box_percentile,
            box_min_area=args.box_min_area, box_expand=args.box_expand,
            prompt_mode=args.prompt_mode,
            post_crf=args.post_crf, crf_from=args.crf_from, crf_iters=args.crf_iters,
            crf_sxy_gaussian=args.crf_sxy_gaussian, crf_compat_gaussian=args.crf_compat_gaussian,
            crf_sxy_bilateral=args.crf_sxy_bilateral, crf_srgb=args.crf_srgb, crf_compat_bilateral=args.crf_compat_bilateral,
            stability_filter=args.stability_filter,
            stable_thresholds=tuple(args.stable_thresholds),
            stable_open_area=args.stable_open_area,
            stable_close_area=args.stable_close_area,
            use_autocast=True,
            out_prefix=args.out_prefix
        )

    if not (args.build_cache or args.test or args.single):
        print('Nothing to do. Use --build-cache and/or --test, or --single for box/box+point mode.')