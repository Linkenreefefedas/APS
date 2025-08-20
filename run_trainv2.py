# runv3.py — Train PointPromptNet‑B (~1.55M) to emit SAM2 point prompts
# ---------------------------------------------------------------------
# This script keeps SAM2 completely out of the training loss (no E2E with SAM2).
# Supervision is on (1) foreground / background point heatmaps and (2) point counts.
# It reuses encoder-level features from SAM2 and DINOv3; features are precomputed & cached
# for efficient training on a single A100. Validation can optionally call SAM2 to report mIoU.
#
# Quick start
# 1) Prepare a CSV with columns: target, target_mask, ref, ref_mask, split (train/val).
# 2) Build feature cache (fast offline step):
#    python runv3.py --csv episodes.csv --cache-dir cache --sam2-cfg configs/sam2.1/sam2.1_hiera_s.yaml \
#                    --sam2-ckpt checkpoints/sam2.1_hiera_small.pt --dinov3-model-id facebook/dinov3-vitb16-pretrain-lvd1689m \
#                    --build-cache
# 3) Train:
#    python runv3.py --csv episodes.csv --cache-dir cache --epochs 80 --batch-size 24 --lr 1e-3 --kmax 8 \
#                    --sam2-cfg configs/sam2.1/sam2.1_hiera_s.yaml --sam2-ckpt checkpoints/sam2.1_hiera_small.pt \
#                    --dinov3-model-id facebook/dinov3-vitb16-pretrain-lvd1689m --train
# 4) Validate with SAM2 for true mIoU (optional, slower): add --val-with-sam2 --val-samples 128
# 5) Inference on one episode (save overlay):
#    python runv3.py --infer --target path.jpg --target-mask path.png --ref path.jpg --ref-mask path.png \
#                    --sam2-cfg ... --sam2-ckpt ... --dinov3-model-id ...

import os, math, csv, json, time, random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoImageProcessor, AutoModel  # HF DINOv3
import hashlib
import dataloader as dl
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate


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

def _get_base_model(m):
    return m.module if hasattr(m, 'module') else m

def save_checkpoint(path, model, opt, sch, epoch, best_iou, lc):
    state = {
        'model': _get_base_model(model).state_dict(),
        'optimizer': opt.state_dict(),
        'scheduler': sch.state_dict(),
        'epoch': int(epoch),
        'best_iou': float(best_iou),
        'lc': float(lc)
    }
    torch.save(state, path)

def collate_keep_rgb_mask_as_list(batch):
    # 先把會用於訓練的鍵保留，由 default_collate 做堆疊（它會檢查 shape）
    keys_stack = [
        'sam_t','dn_t',
        'proto_fg_sam','proto_bg_sam','proto_fg_dn','proto_bg_dn',
        'x_in',
        'H_fg_star','H_bg_star',
        'K_fg_star','K_bg_star'
    ]
    batch_core = [{k:v for k,v in b.items() if k in keys_stack} for b in batch]
    out = default_collate(batch_core)  # 這裡會把同形狀的 tensor 堆成 batch

    # 其餘變動尺寸的欄位保留成 list（不讓 default_collate 過手）
    if 'tgt_rgb' in batch[0]:
        out['tgt_rgb']  = [b['tgt_rgb']  for b in batch]
    if 'tgt_mask' in batch[0]:
        out['tgt_mask'] = [b['tgt_mask'] for b in batch]
    return out

def _to_bchw(t, name="tensor"):
    """
    將 t 規整為 [B,C,H,W]。
    規則：
        - 3D: [C,H,W] -> [1,C,H,W]
        - 4D: [B,C,H,W] -> 原樣
        - 5D: 常見為 [B,Cg,K,H,W] 或 [1,B,C,H,W] 等
            若形狀像 [B, Cg, K, H, W]：把 Cg 和 K 併到通道：C = Cg*K
            若形狀像 [1, B, C, H, W]：先 squeeze 前導的 1 再判斷
        - 其他維度數：報錯
    """
    if t is None:
        raise ValueError(f"{name} is None")

    # 先消掉所有「前導的 size=1」維度（但保留末兩維 H,W）
    while t.dim() >= 5 and t.shape[0] == 1:
        t = t.squeeze(0)

    if t.dim() == 3:
        # [C,H,W]
        t = t.unsqueeze(0)
    elif t.dim() == 4:
        # [B,C,H,W]
        pass
    elif t.dim() == 5:
        # 嘗試辨識 [B,Cg,K,H,W] -> 併 Cg,K
        B, D1, D2, H, W = t.shape
        # 若其中一維是小批次誤疊（像 [1,B,C,H,W]）已在上面 squeeze 過
        # 合併第1與第2維到通道
        t = t.reshape(B, D1 * D2, H, W)
    else:
        raise ValueError(f"{name}: unexpected ndim={t.dim()}, shape={tuple(t.shape)}")

    return t

# ---- New: deterministic resize+letterbox to unify cache grid size ----
def resize_letterbox_rgb(rgb: np.ndarray,
                         fixed_size: int = 1024) -> tuple[np.ndarray, dict]:
    """
    直接把影像 resize 成 fixed_size×fixed_size，並回傳與舊版相容的 meta。
    """
    H, W = rgb.shape[:2]
    if max(H, W) == 0:
        raise ValueError("Invalid image size")

    # 放大用 CUBIC、縮小用 AREA
    interp = cv2.INTER_AREA if (H >= fixed_size and W >= fixed_size) else cv2.INTER_CUBIC
    out = cv2.resize(rgb, (fixed_size, fixed_size), interpolation=interp)

    meta = dict(
        # 舊欄位（相容用）：不做 letterbox，所以 top/left=0
        top=0, left=0,
        out_h=fixed_size, out_w=fixed_size,
        # 舊欄位 scale：給 1.0（或你要改成 scale_x/scale_y 也行，但目前沒用到）
        scale=1.0,
        # 原圖尺寸（有時會用到）
        orig_h=H, orig_w=W
    )
    return out, meta

def resize_letterbox_mask(mask: np.ndarray,
                          fixed_size: int = 1024,k =None) -> np.ndarray:
    """
    把 mask 直接 resize 成固定大小 (fixed_size × fixed_size)。
    """
    out = cv2.resize(mask, (fixed_size, fixed_size),
                     interpolation=cv2.INTER_NEAREST)
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
    """Return DINOv3 grid features [Gh,Gw,C], L2-normalized, float32."""
    proc, model = _get_dinov3_hf(model_id)
    pil = Image.fromarray(image_rgb)
    inputs = proc(images=pil, return_tensors="pt", do_center_crop=False, do_resize=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs)
    x = out.last_hidden_state
    # ViT branch
    assert x.dim() in (3,4)
    if x.dim()==4:
        grid = x[0].permute(1,2,0).contiguous()
        return F.normalize(grid, dim=-1, eps=1e-6).to(torch.float32)
    Hp, Wp = inputs["pixel_values"].shape[-2:]
    psize = int(getattr(getattr(model, "config", None), "patch_size", 16))
    Gh, Gw = Hp//psize, Wp//psize
    M = Gh*Gw
    toks = x[0, -M:, :]
    grid = toks.view(Gh, Gw, -1).contiguous()
    return F.normalize(grid, dim=-1, eps=1e-6).to(torch.float32)

# ------------------------------ SAM2 image encoder ---------------------
def sam2_build_image_predictor(cfg_yaml: str, ckpt_path: str):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model = build_sam2(cfg_yaml, ckpt_path).to(device)
    return SAM2ImagePredictor(model)

def _extract_sam2_image_embed(predictor) -> torch.Tensor:
    feats_dict = None
    for attr in ("_features", "features"):
        if hasattr(predictor, attr):
            d = getattr(predictor, attr)
            if isinstance(d, dict):
                feats_dict = d; break
    if feats_dict is None:
        raise RuntimeError("SAM2 predictor has no feature dict (_features/features)")
    for k in ("image_embed", "image_embeddings", "image_features", "vision_feats"):
        if k in feats_dict:
            t = feats_dict[k]
            if isinstance(t, (list, tuple)): t = t[0]
            if t.dim()==4: t = t[0]
            assert t.dim()==3
            return t.to(torch.float32)
    raise KeyError("Cannot locate SAM2 image embedding tensor in predictor features")

@torch.inference_mode()
def get_sam2_feat(image_rgb: np.ndarray, predictor, use_autocast: bool=False) -> torch.Tensor:
    with AutocastCtx(use_autocast):
        predictor.set_image(image_rgb)
    feat = _extract_sam2_image_embed(predictor)  # [C,Hf,Wf]
    return F.normalize(feat, dim=0, eps=1e-6).to(torch.float32)

# ------------------------------ Prototypes & sims ----------------------
def compute_proto_and_sims(ref_rgb: np.ndarray, ref_mask: np.ndarray,
                           tgt_rgb: np.ndarray,
                           predictor, dinov3_id: str,
                           use_bg_proto: bool=True,
                           tau: float=0.2,
                           use_autocast: bool=False):
    # SAM2
    ref_sam = get_sam2_feat(ref_rgb, predictor, use_autocast)
    C, Hf, Wf = ref_sam.shape
    tgt_sam = get_sam2_feat(tgt_rgb, predictor, use_autocast)
    m_small = cv2.resize(ref_mask, (Wf, Hf), interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
    m_small = torch.from_numpy(m_small).to(device=tgt_sam.device, dtype=torch.float32)
    fg = (m_small>0.5).float(); bg = (m_small<=0.5).float()
    proto_fg_sam = (ref_sam * fg.unsqueeze(0)).sum((1,2)) / (fg.sum()+1e-8)
    proto_fg_sam = F.normalize(proto_fg_sam, dim=0, eps=1e-6)
    sim_fg_sam = (tgt_sam * proto_fg_sam.view(-1,1,1)).sum(0)
    if use_bg_proto:
        if bg.sum()<1:
            inv = (1.0-m_small).flatten(); topk = torch.topk(inv, k=min(10, inv.numel()))[1]
            bg = torch.zeros_like(inv); bg[topk]=1.0; bg = bg.view(Hf,Wf)
        proto_bg_sam = (ref_sam * bg.unsqueeze(0)).sum((1,2)) / (bg.sum()+1e-8)
        proto_bg_sam = F.normalize(proto_bg_sam, dim=0, eps=1e-6)
        sim_sam = torch.sigmoid((sim_fg_sam - (tgt_sam * proto_bg_sam.view(-1,1,1)).sum(0))/tau)
    else:
        sim_sam = (sim_fg_sam+1.0)/2.0

    # DINOv3
    ref_grid = get_grid_feats_dinov3_hf(ref_rgb, dinov3_id)   # [Gr,Gr,Cd]
    tgt_grid = get_grid_feats_dinov3_hf(tgt_rgb, dinov3_id)   # [Gt,Gt,Cd]
    Gr = ref_grid.shape[0]
    m_small2 = cv2.resize(ref_mask, (Gr, Gr), interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
    m_small2 = torch.from_numpy(m_small2).to(device=tgt_grid.device, dtype=torch.float32)
    fg2 = (m_small2>0.5).float(); bg2 = (m_small2<=0.5).float()
    proto_fg_dn = (ref_grid * fg2.unsqueeze(-1)).sum((0,1)) / (fg2.sum()+1e-8)
    proto_fg_dn = F.normalize(proto_fg_dn, dim=0, eps=1e-6)
    sim_fg_dn = (tgt_grid * proto_fg_dn.view(1,1,-1)).sum(-1)
    if use_bg_proto:
        proto_bg_dn = (ref_grid * bg2.unsqueeze(-1)).sum((0,1)) / (bg2.sum()+1e-8)
        proto_bg_dn = F.normalize(proto_bg_dn, dim=0, eps=1e-6)
        sim_dn = torch.sigmoid((sim_fg_dn - (tgt_grid * proto_bg_dn.view(1,1,-1)).sum(-1))/tau)
    else:
        sim_dn = (sim_fg_dn+1.0)/2.0

    return {
        "sam_feat_ref": ref_sam, "sam_feat_tgt": tgt_sam,
        "dino_feat_ref": ref_grid, "dino_feat_tgt": tgt_grid,
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

# ------------------------------ PointPromptNet‑B ----------------------
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

class PointPromptNetB(nn.Module):
    def __init__(self, c_in=101, c0=192, c1=256, c2=384, kmax_f=8, kmax_b = 8 , 
                 sam_c=256, dino_c=768, proj_dim=48):
        super().__init__()
        self.proj_sam = nn.Conv2d(sam_c, proj_dim, 1, bias=False)
        self.proj_dn  = nn.Conv2d(dino_c, proj_dim, 1, bias=False)
        self.proto_lin = nn.Linear(proj_dim*2, c0)
        self.in_proj = nn.Sequential(nn.Conv2d(c_in, c0, 1, bias=False), nn.BatchNorm2d(c0), nn.GELU())
        self.enc0 = nn.Sequential(Block(c0), Block(c0))
        self.down1 = Down(c0, c1); self.enc1 = nn.Sequential(Block(c1), Block(c1))
        self.down2 = Down(c1, c2); self.enc2 = nn.Sequential(Block(c2), Block(c2))
        self.bot  = Block(c2)
        self.pxattn = ProtoCrossAttn(c=c0, heads=4, mlp_ratio=2.0)
        self.up1  = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                  nn.Conv2d(c2, c1, 1, bias=False), nn.BatchNorm2d(c1), nn.GELU(), Block(c1))
        self.up0  = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                  nn.Conv2d(c1, c0, 1, bias=False), nn.BatchNorm2d(c0), nn.GELU(), Block(c0))
        self.fg_head = nn.Conv2d(c0, 1, 1)
        self.bg_head = nn.Conv2d(c0, 1, 1)
        self.cnt_fg  = nn.Conv2d(c0, kmax_f+1, 1)
        self.cnt_bg  = nn.Conv2d(c0, kmax_b+1, 1)
        self.kmax_f = kmax_f
        self.kmax_b = kmax_b
        # initialize lightly
        for m in [self.proj_sam, self.proj_dn]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
        nn.init.xavier_uniform_(self.proto_lin.weight)

    def map_proto(self, proto_fg_sam, proto_bg_sam, proto_fg_dn, proto_bg_dn):
        # use conv1x1 weights to project prototypes (C -> proj_dim)
        Ws = self.proj_sam.weight.view(self.proj_sam.out_channels, -1)  # [ps, Cs]
        Wd = self.proj_dn.weight.view(self.proj_dn.out_channels, -1)    # [pd, Cd]
        fg_s = (Ws @ proto_fg_sam.unsqueeze(-1)).squeeze(-1)
        bg_s = (Ws @ proto_bg_sam.unsqueeze(-1)).squeeze(-1)
        fg_d = (Wd @ proto_fg_dn.unsqueeze(-1)).squeeze(-1)
        bg_d = (Wd @ proto_bg_dn.unsqueeze(-1)).squeeze(-1)
        fg = torch.cat([fg_s, fg_d], dim=-1)
        bg = torch.cat([bg_s, bg_d], dim=-1)
        fg = self.proto_lin(fg); bg = self.proto_lin(bg)
        return fg, bg  # [B,c0] each

    def forward(self, x_in, sam_feat_tgt, dino_feat_tgt,
                proto_fg_sam, proto_bg_sam, proto_fg_dn, proto_bg_dn):
    
        # --- 規整成 [B,C,H,W] ---
        x_in  = _to_bchw(x_in,  "x_in")
    
        # SAM/DINO 輸入規整（你先前已處理 3D/4D，沿用即可）
        # sam_in: [B,Cs,Hf,Wf]
        if sam_feat_tgt.dim() == 3:
            sam_in = sam_feat_tgt.unsqueeze(0)
        elif sam_feat_tgt.dim() == 4:
            sam_in = sam_feat_tgt
        else:
            raise ValueError(f"sam_feat_tgt ndim={sam_feat_tgt.dim()}")
    
        # dino: 支援 [Gh,Gw,C], [C,Gh,Gw], [B,C,Gh,Gw], [B,Gh,Gw,C]
        if dino_feat_tgt.dim() == 3:
            if dino_feat_tgt.shape[-1] in (256, 384, 768):   # [Gh,Gw,C]
                dn_in = dino_feat_tgt.permute(2,0,1).unsqueeze(0)
            elif dino_feat_tgt.shape[0] in (256, 384, 768):  # [C,Gh,Gw]
                dn_in = dino_feat_tgt.unsqueeze(0)
            else:
                raise ValueError(f"dino_feat_tgt shape not recognized: {tuple(dino_feat_tgt.shape)}")
        elif dino_feat_tgt.dim() == 4:
            if dino_feat_tgt.shape[1] in (256, 384, 768):    # [B,C,Gh,Gw]
                dn_in = dino_feat_tgt
            elif dino_feat_tgt.shape[-1] in (256, 384, 768): # [B,Gh,Gw,C]
                dn_in = dino_feat_tgt.permute(0,3,1,2)
            else:
                raise ValueError(f"dino_feat_tgt shape not recognized: {tuple(dino_feat_tgt.shape)}")
        else:
            raise ValueError(f"dino_feat_tgt ndim={dino_feat_tgt.dim()}")
    
        # --- 投影 ---
        sam_p = self.proj_sam(sam_in)  # [B, ps, Hf, Wf]
        dn_p  = self.proj_dn(dn_in)    # [B, pd, Gh, Gw]
    
        # --- 空間對齊到 x_in 的 H×W ---
        H, W = x_in.shape[-2], x_in.shape[-1]
        if sam_p.shape[-2:] != (H, W):
            sam_p = F.interpolate(sam_p, size=(H, W), mode='bilinear', align_corners=False)
        if dn_p.shape[-2:] != (H, W):
            dn_p  = F.interpolate(dn_p,  size=(H, W), mode='bilinear', align_corners=False)
    
        # --- 若 x_in 偶然是 5D（例如 [B,2,K,H,W]）再保險一次壓成 BCHW ---
        x_in = _to_bchw(x_in, "x_in(BCHW-check)")
    
        # --- 這裡就能安全 cat ---
        x = torch.cat([sam_p, dn_p, x_in], dim=1)
        x = self.in_proj(x)
        s0 = self.enc0(x)
        s1 = self.enc1(self.down1(s0))
        s2 = self.enc2(self.down2(s1))
        z  = self.bot(s2)
        u1 = self.up1(z) + s1
        u0 = self.up0(u1) + s0
        # proto cross-attn
        fg_tok, bg_tok = self.map_proto(proto_fg_sam, proto_bg_sam, proto_fg_dn, proto_bg_dn)
        protos = torch.stack([fg_tok, bg_tok], dim=1)  # [B,2,c0]
        u0, _ = self.pxattn(u0, protos)
        # H_fg = torch.sigmoid(self.fg_head(u0))
        # H_bg = torch.sigmoid(self.bg_head(u0))
        H_fg = self.fg_head(u0)
        H_bg = self.bg_head(u0)

        Pk_fg = self.cnt_fg(u0).mean(dim=(2,3))
        Pk_bg = self.cnt_bg(u0).mean(dim=(2,3))
        return H_fg, H_bg, Pk_fg, Pk_bg

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

def count_targets(mask: np.ndarray) -> int:
    m = (mask>127).astype(np.uint8)
    n, labels = cv2.connectedComponents(m, connectivity=8)
    return max(1, n-1)

def build_gt(target_mask: np.ndarray,
             sim_fg_outside: torch.Tensor,
             size_hw: Tuple[int,int], kmax_f:int=8, kmax_b:int =8,
             alpha:float=0.5, beta:float=0.5):
    # 以 sim_fg_outside 的 device 為準（cuda 或 cpu）
    device = sim_fg_outside.device
    Ht, Wt = size_hw

    # 這兩個 helper 會回傳 CPU tensor；這裡統一搬到 device
    H_fg_star = distance_transform_heatmap(target_mask, size_hw).to(device)  # [Ht,Wt]
    ring      = outer_ring(target_mask, size_hw, r=3).to(device)            # [Ht,Wt]

    # outside 已經在正確的 device（由呼叫端傳入）
    outside = torch.clamp(sim_fg_outside, 0, 1)

    # 背景熱度圖
    H_bg_star = torch.clamp(alpha*ring + beta*outside, 0, 1)

    # counts（純 CPU/Numpy 計算，回傳 python int 不涉 device）
    m_small = (cv2.resize(target_mask, (Wt, Ht), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
    n, _ = cv2.connectedComponents(m_small, connectivity=8)
    K_fg_star = min(max(1, n-1), kmax_f)

    cnts = cv2.findContours(m_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    perim = sum(cv2.arcLength(c, True) for c in cnts) if cnts else 0.0
    K_bg_star = int(np.clip(round(perim/200.0), 0, kmax_b))

    return H_fg_star, H_bg_star, K_fg_star, K_bg_star


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

def grid_to_xy(rc: Tuple[int,int], Hf:int, Wf:int, H:int, W:int):
    r,c = rc
    x = int((c+0.5)*W/Wf)
    y = int((r+0.5)*H/Hf)
    return [x,y]

# ------------------------------ Dataset & cache -----------------------
@dataclass
class Episode:
    target: str
    target_mask: str
    ref: str
    ref_mask: str
    split: str

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
        self.cache_long = int(cache_long)
        self.cache_multiple = int(cache_multiple)
        if build_cache:
            self._build_all()

    def _key(self, img_path: str) -> str:
        return cache_key(img_path)

    def _build_one(self, target: str):
        key = self._key(target)
        out = os.path.join(self.cache_dir, f"{key}.npz")
        if os.path.exists(out):
            return
        rgb_raw = load_rgb(target)
        rgb_lb, meta = resize_letterbox_rgb(rgb_raw, self.cache_long)
        sam = get_sam2_feat(rgb_lb, self.predictor, use_autocast=True)           # [Cs,Hf,Wf]
        dng = get_grid_feats_dinov3_hf(rgb_lb, self.dinov3_id).permute(2,0,1)    # [Cd,Gh,Gw]
        np.savez_compressed(out,
            sam= sam.cpu().numpy().astype(np.float16),
            dng= dng.cpu().numpy().astype(np.float16),
            out_h = np.int32(meta['out_h']), out_w = np.int32(meta['out_w']),
            top = np.int32(meta['top']), left = np.int32(meta['left']),
            scale = np.float32(meta['scale']), orig_h = np.int32(meta['orig_h']), orig_w = np.int32(meta['orig_w']))

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
        data = np.load(os.path.join(self.cache_dir, f"{key}.npz"))
        sam = torch.from_numpy(data['sam'].astype(np.float32))
        dng = torch.from_numpy(data['dng'].astype(np.float32))
        return sam, dng  # [Cs,Hf,Wf], [Cd,Gh,Gw]

    def __len__(self): return len(self.episodes)

    def __getitem__(self, i):
        e = self.episodes[i]
        tgt_rgb = load_rgb(e.target); ref_rgb = load_rgb(e.ref)
        tgt_mask = load_gray(e.target_mask); ref_mask = load_gray(e.ref_mask)
        # load cached feats
        sam_t, dn_t = self._load_cached(e.target)
        sam_r, dn_r = self._load_cached(e.ref)
        # sims & protos (cheap on cached feats)
        C,Hf,Wf = sam_r.shape
        m_small = cv2.resize(ref_mask, (Wf,Hf), interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
        m_small = torch.from_numpy(m_small)
        fg = (m_small>0.5).float(); bg = (m_small<=0.5).float()
        proto_fg_sam = (sam_r * fg.unsqueeze(0)).sum((1,2))/(fg.sum()+1e-8)
        proto_fg_sam = F.normalize(proto_fg_sam, dim=0, eps=1e-6)
        sim_fg_sam = (sam_t * proto_fg_sam.view(-1,1,1)).sum(0)
        if bg.sum()<1:
            inv = (1.0-m_small).flatten(); topk=torch.topk(inv, k=min(10, inv.numel()))[1]
            bg = torch.zeros_like(inv); bg[topk]=1.0; bg = bg.view(Hf,Wf)
        proto_bg_sam = (sam_r * bg.unsqueeze(0)).sum((1,2))/(bg.sum()+1e-8)
        proto_bg_sam = F.normalize(proto_bg_sam, dim=0, eps=1e-6)
        sim_bg_sam = (sam_t * proto_bg_sam.view(-1,1,1)).sum(0)
        sim_sam = torch.sigmoid((sim_fg_sam - sim_bg_sam)/0.2).to(torch.float32)
        # dino
        Cd, Gh, Gw = dn_r.shape
        m_small2 = cv2.resize(ref_mask, (Gw,Gh), interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
        m_small2 = torch.from_numpy(m_small2)
        fg2 = (m_small2>0.5).float(); bg2 = (m_small2<=0.5).float()
        proto_fg_dn = (dn_r * fg2.unsqueeze(0)).sum((1,2))/(fg2.sum()+1e-8)
        proto_fg_dn = F.normalize(proto_fg_dn, dim=0, eps=1e-6)
        sim_fg_dn = (dn_t * proto_fg_dn.view(-1,1,1)).sum(0)
        proto_bg_dn = (dn_r * bg2.unsqueeze(0)).sum((1,2))/(bg2.sum()+1e-8)
        proto_bg_dn = F.normalize(proto_bg_dn, dim=0, eps=1e-6)
        sim_bg_dn = (dn_t * proto_bg_dn.view(-1,1,1)).sum(0)
        sim_dn = torch.sigmoid((sim_fg_dn - sim_bg_dn)/0.2).to(torch.float32)
        # upsample sims to Hf,Wf
        sim_dn_up = F.interpolate(sim_dn.unsqueeze(0).unsqueeze(0), size=(Hf,Wf), mode='bicubic', align_corners=False).squeeze(0).squeeze(0)
        # input extra channels: sims + edge
        edge = sobel_edge_hint(tgt_rgb, (Hf,Wf))
        x_in = torch.stack([sim_sam, sim_dn_up,
                            (sim_sam+sim_dn_up)/2.0,  # fused sim
                            1.0 - (sim_sam+sim_dn_up)/2.0,  # inverse as bg prior
                            edge], dim=0).unsqueeze(0)  # [1,5,Hf,Wf]
        # GT
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
            'tgt_rgb': tgt_rgb, 'tgt_mask': tgt_mask
        }
        return sample

class COCO20iEpisodeAdapter(Dataset):
    """
    Wrap OneShotCOCO20iRandom/RoundRobin to produce the same training sample dict
    the original EpisodeDataset returned, but entirely from cached features.
    """
    def __init__(self, base_dataset, cache_dir: str, cache_long: int = 1024, cache_multiple: int = 16):
        self.base = base_dataset
        self.cache_dir = cache_dir
        self.cache_long = int(cache_long)
        self.cache_multiple = int(cache_multiple)

    def __len__(self): return len(self.base)

    def _load_cached(self, img_path: str):
        key = cache_key(img_path)
        data = np.load(os.path.join(self.cache_dir, f"{key}.npz"))
        sam = torch.from_numpy(data['sam'].astype(np.float32))  # CPU
        dng = torch.from_numpy(data['dng'].astype(np.float32))  # CPU
        out_h, out_w = int(data['out_h']), int(data['out_w'])
        return sam, dng, (out_h, out_w)

    def __getitem__(self, i):
        ep = self.base[i]
        sup = ep['support']; qry = ep['query']
    
        # 取得原圖與遮罩（numpy）
        sup_img = (sup['image'].numpy().transpose(1,2,0) * 255.0).astype(np.uint8)
        qry_img = (qry['image'].numpy().transpose(1,2,0) * 255.0).astype(np.uint8)
        sup_mask = (sup['mask'].numpy().astype(np.uint8) * 255)
        qry_mask = (qry['mask'].numpy().astype(np.uint8) * 255)
    
        sup_path = sup['meta']['image_path']
        qry_path = qry['meta']['image_path']
    
        # 載入快取特徵
        sam_r, dn_r, _ = self._load_cached(sup_path)
        sam_t, dn_t, _ = self._load_cached(qry_path)
    
        # === SAM2 原型 + 相似度 ===
        Cr, Hr, Wr = sam_r.shape
        Ct, Ht, Wt = sam_t.shape
    
        m_small = cv2.resize(sup_mask, (Wr, Hr), interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
        m_small = torch.from_numpy(m_small)
        fg = (m_small > 0.5).float()
        bg = (m_small <= 0.5).float()
    
        proto_fg_sam = (sam_r * fg.unsqueeze(0)).sum((1,2)) / (fg.sum()+1e-8)
        proto_fg_sam = F.normalize(proto_fg_sam, dim=0, eps=1e-6)
        sim_fg_sam = (sam_t * proto_fg_sam.view(-1,1,1)).sum(0)
    
        if bg.sum() < 1:
            inv = (1.0-m_small).flatten(); topk = torch.topk(inv, k=min(10, inv.numel()))[1]
            bg = torch.zeros_like(inv); bg[topk]=1.0; bg = bg.view(Hr,Wr)
        proto_bg_sam = (sam_r * bg.unsqueeze(0)).sum((1,2)) / (bg.sum()+1e-8)
        proto_bg_sam = F.normalize(proto_bg_sam, dim=0, eps=1e-6)
        sim_bg_sam = (sam_t * proto_bg_sam.view(-1,1,1)).sum(0)
    
        sim_sam = torch.sigmoid((sim_fg_sam - sim_bg_sam)/0.2).to(torch.float32)
    
        # === DINOv3 原型 + 相似度 ===
        Cd, Gh, Gw = dn_r.shape
        m_small2 = cv2.resize(sup_mask, (Gw,Gh), interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
        m_small2 = torch.from_numpy(m_small2)
        fg2 = (m_small2>0.5).float(); bg2 = (m_small2<=0.5).float()
    
        proto_fg_dn = (dn_r * fg2.unsqueeze(0)).sum((1,2))/(fg2.sum()+1e-8)
        proto_fg_dn = F.normalize(proto_fg_dn, dim=0, eps=1e-6)
        sim_fg_dn = (dn_t * proto_fg_dn.view(-1,1,1)).sum(0)
    
        proto_bg_dn = (dn_r * bg2.unsqueeze(0)).sum((1,2))/(bg2.sum()+1e-8)
        proto_bg_dn = F.normalize(proto_bg_dn, dim=0, eps=1e-6)
        sim_bg_dn = (dn_t * proto_bg_dn.view(-1,1,1)).sum(0)
    
        sim_dn = torch.sigmoid((sim_fg_dn - sim_bg_dn)/0.2).to(torch.float32)
        sim_dn_up = F.interpolate(sim_dn.unsqueeze(0).unsqueeze(0),
                                  size=(Ht,Wt), mode='bicubic', align_corners=False
                                 ).squeeze(0).squeeze(0)
    
        # === edge hint + 輸入堆疊 ===
        edge = sobel_edge_hint(qry_img, (Ht, Wt))
        x_in = torch.stack([
            sim_sam, sim_dn_up,
            (sim_sam+sim_dn_up)/2.0,
            1.0 - (sim_sam+sim_dn_up)/2.0,
            edge
        ], dim=0).unsqueeze(0)  # [1,5,Ht,Wt]
    
        # === GT (使用 query 的 mask) ===
        sim_fg_outside = torch.clamp(
            sim_sam - torch.from_numpy(
                (cv2.resize(qry_mask,(Wt,Ht), interpolation=cv2.INTER_NEAREST) > 127).astype(np.float32)
            ),
            0, 1
        )
        H_fg_star, H_bg_star, K_fg_star, K_bg_star = build_gt(qry_mask, sim_fg_outside, (Ht,Wt))
    
        return {
            'sam_t': sam_t.unsqueeze(0), 'dn_t': dn_t,
            'proto_fg_sam': proto_fg_sam.unsqueeze(0), 'proto_bg_sam': proto_bg_sam.unsqueeze(0),
            'proto_fg_dn':  proto_fg_dn.unsqueeze(0),  'proto_bg_dn':  proto_bg_dn.unsqueeze(0),
            'x_in': x_in,
            'H_fg_star': H_fg_star.unsqueeze(0), 'H_bg_star': H_bg_star.unsqueeze(0),
            'K_fg_star': torch.tensor(K_fg_star, dtype=torch.long),
            'K_bg_star': torch.tensor(K_bg_star, dtype=torch.long),
            'tgt_rgb': qry_img, 'tgt_mask': qry_mask
        }
    

# ------------------------------ Losses --------------------------------
# class FocalBCE(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
#         super().__init__(); self.a=alpha; self.g=gamma; self.red=reduction
#     def forward(self, pred, target):
#         pred = torch.clamp(pred, 1e-6, 1-1e-6)
#         pt = pred*target + (1-pred)*(1-target)
#         w = self.a*target + (1-self.a)*(1-target)
#         loss = -w*((1-pt)**self.g)*torch.log(pt)
#         if self.red=='mean': return loss.mean()
#         if self.red=='sum': return loss.sum()
#         return loss

class FocalBCEWithLogits(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', eps=1e-6):
        super().__init__()
        self.a = alpha
        self.g = gamma
        self.red = reduction
        self.eps = eps

    def forward(self, logits, target, mask=None):
        # 1) 基礎 BCE（logits 版），避免自己手刻 log/exp 的數值風險
        ce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        # 2) p_t 與 alpha_t
        p = torch.sigmoid(logits)
        p_t = p * target + (1 - p) * (1 - target)
        alpha_t = self.a * target + (1 - self.a) * (1 - target)
        # 3) 調制因子 (1 - p_t)^gamma
        mod = (1 - p_t).clamp_min(self.eps) ** self.g
        loss = alpha_t * mod * ce  # 對每一像素/元素

        if mask is not None:
            loss = loss * mask
            denom = mask.sum().clamp_min(1.0)
            return loss.sum() / denom

        if self.red == 'mean':
            return loss.mean()
        if self.red == 'sum':
            return loss.sum()
        return loss

def prep_target_like(logits, target):
    """
    將 target 調整為與 logits 相容的形狀與解析度：
    - 接受 [B,H,W]、[B,1,H,W]、[B,1,1,H,W]
    - 回傳 [B,1,Hlog,Wlog]
    """
    # 去掉多餘的 1 維
    if target.dim() == 5 and target.size(2) == 1:
        target = target.squeeze(2)   # [B,1,H,W]
    if target.dim() == 3:
        target = target.unsqueeze(1) # [B,1,H,W]
    assert target.dim() == 4, f"target ndim should be 4, got {target.dim()} with shape {tuple(target.shape)}"

    # 解析度對齊
    Ht, Wt = target.shape[-2:]
    Hl, Wl = logits.shape[-2:]
    if (Ht, Wt) != (Hl, Wl):
        target = F.interpolate(target.float(), size=(Hl, Wl), mode='bilinear', align_corners=False)
        # 二值/機率標籤：插值後夾一下範圍
        target = target.clamp_(0.0, 1.0)
    return target

# ------------------------------ Train / Eval --------------------------
def train_loop(args):
    start_epoch = 0
    global_step = 0
    best_iou = 0.0
    lc = args.lc

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        _get_base_model(model).load_state_dict(ckpt['model'], strict=False)
        if 'optimizer' in ckpt: opt.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt: sch.load_state_dict(ckpt['scheduler'])
        best_iou = float(ckpt.get('best_iou', 0.0))
        start_epoch = int(ckpt.get('epoch', -1)) + 1
        lc = float(ckpt.get('lc', args.lc))
    
        print(f"[resume] from {args.resume} → epoch {start_epoch}, step {global_step}, best_iou {best_iou:.4f}")


    # === 建 COCO-20i 索引（以 manifest_train.csv）===
    assert args.manifest_train is not None, "Please set --manifest-train (manifest_train.csv)"
    idx_train = dl.build_coco20i_index(args.manifest_train, fold=args.coco20i_fold, role=args.role)

    # === 用 COCO-20i Random episodes 當訓練資料 ===
    base_train = dl.OneShotCOCO20iRandom(index=idx_train, episodes=args.episodes, seed=2025)
    ds = COCO20iEpisodeAdapter(base_train, args.cache_dir, cache_long=args.cache_long, cache_multiple=args.cache_multiple)
    dl_torch = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=True, drop_last=True,
        collate_fn=collate_keep_rgb_mask_as_list,
    )
    
    # eval data
    idx_val = dl.build_coco20i_index(args.manifest_val, fold=args.coco20i_fold, role=args.role)
    base_val = dl.OneShotCOCO20iRandom(index=idx_val, episodes=args.val_samples, seed=2025)
    ds_val = COCO20iEpisodeAdapter(base_val, args.cache_dir, cache_long=args.cache_long, cache_multiple=args.cache_multiple)
    eval_loader = DataLoader(
        ds_val, batch_size=1, shuffle=False,
        collate_fn=collate_keep_rgb_mask_as_list,
    )

    # === 估計通道數，建立模型 ===
    # 從第一筆資料探測 Cs/Cd
    probe = ds[0]
    sam_C = probe['sam_t'].shape[1]
    dino_C = probe['dn_t'].shape[0]
    model = PointPromptNetB(c_in=101, kmax_f=args.kmax_f,kmax_b= args.kmax_b, sam_c=sam_C, dino_c=dino_C).to(device)
    if args.torch_compile:
        model = torch.compile(model, mode='reduce-overhead')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs*max(1,len(dl_torch)))
    bce = FocalBCEWithLogits(alpha=0.25, gamma=2.0)
    ce  = nn.CrossEntropyLoss()

    global_step = 0
    for epoch in range(start_epoch, args.epochs):
        if (epoch+1) % 10 == 0:
            base_train = dl.OneShotCOCO20iRandom(index=idx_train, episodes=args.episodes, seed=epoch)
            ds = COCO20iEpisodeAdapter(base_train, args.cache_dir, cache_long=args.cache_long, cache_multiple=args.cache_multiple)
            dl_torch = DataLoader(
                    ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=True,
                    persistent_workers=True, drop_last=True,
                    collate_fn=collate_keep_rgb_mask_as_list,
                )
            
            # eval data
            idx_val = dl.build_coco20i_index(args.manifest_val, fold=args.coco20i_fold, role=args.role)
            base_val = dl.OneShotCOCO20iRandom(index=idx_val, episodes=args.val_samples, seed=epoch)
            ds_val = COCO20iEpisodeAdapter(base_val, args.cache_dir, cache_long=args.cache_long, cache_multiple=args.cache_multiple)
            eval_loader = DataLoader(
                    ds_val, batch_size=1, shuffle=False,
                    collate_fn=collate_keep_rgb_mask_as_list,
                )

        model.train(); t0=time.time()
        for it, batch in enumerate(tqdm(dl_torch, desc=f"Epoch {epoch}", unit="batch")):
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

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type=="cuda")):
                H_fg, H_bg, Pk_fg, Pk_bg = model(x_in, sam_t, dn_t, proto_fg_sam, proto_bg_sam, proto_fg_dn, proto_bg_dn)
            
            # ↓↓↓ logits/targets 轉成 float32 再算 loss（更穩定）
            H_fg = H_fg.float(); H_bg = H_bg.float()

            # 超參數（可放 args；先給合理預設）
            tau = getattr(args, "bg_sup_tau", 1.0)     # 抑制強度，1.0 是個安全起點
            detach_sup = getattr(args, "bg_sup_detach", True)  # 是否在抑制項上做 detach
            
            # 準備 target 形狀（你已經有 prep_target_like 類似的函式就沿用）
            Hfg_s = prep_target_like(H_fg, Hfg_s.float())
            Hbg_s = prep_target_like(H_bg, Hbg_s.float())
            
            # 抑制後 logits：前景被背景壓、背景被前景壓
            if detach_sup:
                logits_fg = H_fg - tau * H_bg.detach()
                logits_bg = H_bg - tau * H_fg.detach()
            else:
                logits_fg = H_fg - tau * H_bg
                logits_bg = H_bg - tau * H_fg
            
            # 用 logits 版 focal/BCE 計算（你若用我之前給的 FocalBCEWithLogits 就用它）
            L_heat = bce(logits_fg, Hfg_s) + bce(logits_bg, Hbg_s)


            # Hfg_s = prep_target_like(H_fg, Hfg_s.float())
            # Hbg_s = prep_target_like(H_bg, Hbg_s.float())
            
            # # 目標維度：Hfg_s/Hbg_s 目前是 [B,H,W]；focal 要 [B,1,H,W]
            # L_heat = bce(H_fg, Hfg_s) + bce(H_bg, Hbg_s)
            L_cnt  = ce(Pk_fg, Kfg_s) + ce(Pk_bg, Kbg_s)
            loss   = L_heat + lc * L_cnt
            
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step(); sch.step(); global_step += 1
            if (global_step % args.log_every)==0:
                print(f"epoch {epoch} it {it} loss {loss.item():.4f} L_heat {L_heat.item():.4f} L_cnt {L_cnt.item():.4f} lr {sch.get_last_lr()[0]:.3e}")
        print(f"[epoch {epoch}] time {time.time()-t0:.1f}s\n")



        if (epoch+1) % args.eval_every == 0 and args.manifest_val:
            iou = evaluate(args, model, args.manifest_val, eval_loader)
            if best_iou < iou:
                best_iou = iou
                lc =  min(lc * 1.05, 0.5)
                print(f"Best IoU : {best_iou}")
                os.makedirs(args.out_dir, exist_ok=True)
                save_checkpoint(os.path.join(args.out_dir, "ppnet_best.pt"),
                                model, opt, sch, epoch, best_iou, lc)
        
        if (epoch+1) % args.save_every == 0:
            os.makedirs(args.out_dir, exist_ok=True)
            # 以 epoch 命名的留存點
            save_checkpoint(os.path.join(args.out_dir, f"ppnet_epoch{epoch}.pt"),
                            model, opt, sch, epoch, best_iou, lc)

# ------------------------------ Evaluation ----------------------------
@torch.inference_mode()
def evaluate(args, model, manifest_val_path=None, loader=None):
    model.eval()
    device = next(model.parameters()).device

    if loader is None:
        idx_val = dl.build_coco20i_index(manifest_val_path, fold=args.coco20i_fold, role=args.role)
        base_val = dl.OneShotCOCO20iRoundRobin(index=idx_val, seed=2025, shuffle_classes=True)
        ds_val = COCO20iEpisodeAdapter(base_val, args.cache_dir, cache_long=args.cache_long, cache_multiple=args.cache_multiple)
        loader = DataLoader(ds_val, batch_size=1, shuffle=False)

    ious = []
    predictor = sam2_build_image_predictor(args.sam2_cfg, args.sam2_ckpt) if args.val_with_sam2 else None
    skip = 0
    for batch in loader:
        # --- 移到裝置 ---
        x_in  = batch['x_in'].to(device)
        sam_t = batch['sam_t'].to(device).squeeze(1)
        dn_t  = batch['dn_t'].to(device)
        proto_fg_sam = batch['proto_fg_sam'].to(device).squeeze(1)
        proto_bg_sam = batch['proto_bg_sam'].to(device).squeeze(1)
        proto_fg_dn  = batch['proto_fg_dn'].to(device).squeeze(1)
        proto_bg_dn  = batch['proto_bg_dn'].to(device).squeeze(1)

        # --- 前向 + logits→prob（評估取點用 prob） ---
        with torch.no_grad():
            H_fg, H_bg, Pk_fg, Pk_bg = model(
                x_in, sam_t, dn_t, proto_fg_sam, proto_bg_sam, proto_fg_dn, proto_bg_dn
            )
            H_fg = torch.sigmoid(H_fg)
            H_bg = torch.sigmoid(H_bg)

        # --- 解析度 & K 值 ---
        Hf, Wf = H_fg.shape[-2], H_fg.shape[-1]
        # Pk_* 形狀通常是 [B, Kmax+1]；穩健些用 dim=1
        Kfg = int(torch.argmax(Pk_fg, dim=1).item())
        Kbg = int(torch.argmax(Pk_bg, dim=1).item())
        Kfg = max(0, min(Kfg, args.kmax_f))
        Kbg = max(0, min(Kbg, args.kmax_b))

        # --- 取峰值 + 後備策略 ---
        def pick_points(heat_4d, K, nms_radius=5, thresh=0.2):
            h2d = heat_4d[0, 0] if heat_4d.dim() == 4 else heat_4d
            pts = nms_peaks(h2d, K, radius=nms_radius, thresh=thresh)  # [(r,c), ...]
            if len(pts) == 0 and K > 0:
                # 後備：取 argmax（至少給一個點）
                rc = torch.nonzero(h2d == h2d.max(), as_tuple=False)[0]
                pts = [(int(rc[0].item()), int(rc[1].item()))]
            return pts[:K]

        radius = max(2, int(0.02 * max(Hf, Wf)))
        pts_fg = pick_points(H_fg, Kfg, nms_radius=radius, thresh=0.2)
        pts_bg = pick_points(H_bg, Kbg, nms_radius=radius, thresh=0.2)

        # 若前景與背景都不取點（Kfg=Kbg=0），本樣本略過 SAM2 評估
        if (Kfg + Kbg) == 0:
            continue

        # --- 組 (N,2) 座標與 (N,) labels（SAM2 需要 x,y 順序） ---
        def make_coords_labels(pts_fg, pts_bg):
            fg_xy = np.array([[c, r] for (r, c) in pts_fg], dtype=np.float32)  # (Nf,2)
            bg_xy = np.array([[c, r] for (r, c) in pts_bg], dtype=np.float32)  # (Nb,2)
            if fg_xy.size == 0: fg_xy = np.zeros((0, 2), dtype=np.float32)
            if bg_xy.size == 0: bg_xy = np.zeros((0, 2), dtype=np.float32)
            coords = np.concatenate([fg_xy, bg_xy], axis=0) if (len(fg_xy) + len(bg_xy)) > 0 \
                     else np.zeros((0, 2), dtype=np.float32)
            labels = np.concatenate([
                np.ones((len(fg_xy),), dtype=np.int32),
                np.zeros((len(bg_xy),), dtype=np.int32)
            ], axis=0) if coords.shape[0] > 0 else np.zeros((0,), dtype=np.int32)
            return coords, labels

        pts_xy, pts_label = make_coords_labels(pts_fg, pts_bg)

        # 若仍是空集合，跳過本樣本
        if pts_xy.shape[0] == 0:
            skip+=1
            continue

        if predictor is not None:
            # --- 映射到 letterbox 影像座標 ---
            tgt_rgb_raw = batch['tgt_rgb'][0]                   # HxWx3, uint8
            tgt_rgb_lb, _ = resize_letterbox_rgb(tgt_rgb_raw, args.cache_long)
            H_img, W_img = tgt_rgb_lb.shape[:2]

            # 將 feature grid (Hf,Wf) 座標映射到 letterbox 影像 (H_img,W_img)
            pts_xy = np.array(
                [grid_to_xy((int(y), int(x)), Hf, Wf, H_img, W_img) for (x, y) in pts_xy],
                dtype=np.int32
            ).astype(np.float32)

            # --- SAM2 預測 ---
            predictor.set_image(tgt_rgb_lb)
            masks, scores, _ = predictor.predict(
                point_coords=pts_xy,                  # (N,2) float32
                point_labels=pts_label.astype(np.int32),  # (N,)
                multimask_output=True
            )
            j = int(np.argmax(scores))
            m = masks[j].astype(np.uint8)

            # --- GT 準備（letterbox 一致） ---
            gt_raw = batch['tgt_mask'][0]
            gt_lb = resize_letterbox_mask(gt_raw, args.cache_long)
            gt = (gt_lb > 127).astype(np.uint8)

            # --- IoU ---
            inter = (m > 0) & (gt > 0)
            union = (m > 0) | (gt > 0)
            iou = float(inter.sum()) / float(union.sum() + 1e-6)
            ious.append(iou)

            if args.val_samples > 0 and len(ious) >= args.val_samples:
                break
    
    print(f"skipping times: {skip}")
    if ious:
        miou = float(np.mean(ious))
        print(f"Val mIoU (SAM2) over {len(ious)}: {miou:.4f}")
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

# --- add below read_csv() ---
def build_cache_from_manifests(cache_dir, dinov3_id, sam2_cfg, sam2_ckpt,
                               manifest_paths, cache_long=1024, cache_multiple=16):
    import csv
    ds = EpisodeDataset([], cache_dir, dinov3_id, sam2_cfg, sam2_ckpt,
                        build_cache=False, cache_long=cache_long, cache_multiple=cache_multiple)
    paths = set()
    for mp in manifest_paths:
        with open(mp, 'r', newline='') as f:
            r = csv.DictReader(f)
            for row in r:
                if 'image_path' in row:
                    paths.add(row['image_path'])
                # 若還有其他欄位放影像，也可加:
                # if 'ref_image_path' in row: paths.add(row['ref_image_path'])
    paths = sorted(paths)
    for p in tqdm(paths, desc=f"Caching images from manifests", unit="img"):
        ds._build_one(p)
    print(f"Cache built for {len(paths)} images → {cache_dir}")


# ------------------------------ Inference demo ------------------------
@torch.inference_mode()
def run_infer(args):
    # Minimal demo without cache
    tgt_rgb_raw = load_rgb(args.target)
    tgt_mask_raw = load_gray(args.target_mask) if args.target_mask else None
    ref_rgb_raw = load_rgb(args.ref); ref_mask_raw = load_gray(args.ref_mask)
    # Letterbox both ref and target for consistency
    tgt_rgb, _ = resize_letterbox_rgb(tgt_rgb_raw, args.cache_long)
    ref_rgb, _ = resize_letterbox_rgb(ref_rgb_raw, args.cache_long)
    ref_mask = resize_letterbox_mask(ref_mask_raw, args.cache_long, args.cache_multiple)
    tgt_mask = resize_letterbox_mask(tgt_mask_raw, args.cache_long, args.cache_multiple) if tgt_mask_raw is not None else None
    predictor = sam2_build_image_predictor(args.sam2_cfg, args.sam2_ckpt)
    feats = compute_proto_and_sims(ref_rgb, ref_mask, tgt_rgb, predictor, args.dinov3_model_id, use_bg_proto=True)
    sam_t = feats['sam_feat_tgt'].unsqueeze(0)  # [1,Cs,Hf,Wf]
    dn_t  = feats['dino_feat_tgt'].permute(2,0,1)  # [Cd,Gh,Gw]
    predictor = sam2_build_image_predictor(args.sam2_cfg, args.sam2_ckpt)
    
    Hf,Wf = sam_t.shape[-2:]
    sim_dn_up = F.interpolate(feats['sim_dino'].unsqueeze(0).unsqueeze(0), size=(Hf,Wf), mode='bicubic', align_corners=False).squeeze(0).squeeze(0)
    edge = sobel_edge_hint(tgt_rgb, (Hf,Wf))
    x_in = torch.stack([feats['sim_sam'], sim_dn_up, (feats['sim_sam']+sim_dn_up)/2.0, 1.0-(feats['sim_sam']+sim_dn_up)/2.0, edge], dim=0).unsqueeze(0).to(device)
    model = PointPromptNetB(c_in=x_in.shape[1], kmax_f=args.kmax_f, kmax_b= args.kmax_b).to(device)
    ckpt = torch.load(args.ckpt, map_location=device) if args.ckpt else None
    if ckpt: model.load_state_dict(ckpt['model'], strict=False)
    H_fg, H_bg, Pk_fg, Pk_bg = model(x_in.to(device), sam_t.to(device), dn_t.to(device),
                                     feats['proto_fg_sam'].unsqueeze(0), feats['proto_bg_sam'].unsqueeze(0),
                                     feats['proto_fg_dn'].unsqueeze(0), feats['proto_bg_dn'].unsqueeze(0))
    H_fg = torch.sigmoid(H_fg)  # ← 新增
    H_bg = torch.sigmoid(H_bg)  # ← 新增
    
    Kfg = int(torch.argmax(Pk_fg, dim=-1).item()); Kbg = int(torch.argmax(Pk_bg, dim=-1).item())
    pts_fg = nms_peaks(H_fg[0,0], Kfg, radius=max(2,int(0.02*max(Hf,Wf))), thresh=0.2)
    pts_bg = nms_peaks(H_bg[0,0], Kbg, radius=max(2,int(0.02*max(Hf,Wf))), thresh=0.2)
    pts = pts_fg + pts_bg; labels=[1]*len(pts_fg) + [0]*len(pts_bg)
    H,W = tgt_rgb.shape[:2]
    pts_xy = np.array([grid_to_xy(rc, Hf, Wf, H, W) for rc in pts], dtype=np.int32)
    predictor.set_image(tgt_rgb)
    masks, scores, _ = predictor.predict(point_coords=pts_xy.astype(np.float32), point_labels=np.array(labels, dtype=np.int32), multimask_output=True)
    j = int(np.argmax(scores)); m = masks[j].astype(np.uint8)
    overlay = tgt_rgb.copy(); overlay[m>0] = (overlay[m>0]*0.5 + np.array([0,255,0])*0.5).astype(np.uint8)
    for (x,y),lb in zip(pts_xy, labels):
        color = (0,255,0) if lb==1 else (255,0,0)
        cv2.circle(overlay, (int(x),int(y)), 5, color, -1)
    os.makedirs(args.out_prefix, exist_ok=True)
    cv2.imwrite(os.path.join(args.out_prefix, 'mask.png'), (m*255))
    cv2.imwrite(os.path.join(args.out_prefix, 'overlay.png'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print('Saved →', os.path.join(args.out_prefix, 'overlay.png'))

# ------------------------------ CLI -----------------------------------
def parse_args():
    p = argparse.ArgumentParser('PointPromptNet‑B trainer (SAM2+DINOv3 encoders, no SAM2 loss)')
    p.add_argument('--csv', type=str, default='episodes.csv')
    p.add_argument('--cache-dir', type=str, default='cache')
    p.add_argument('--cache-long', type=int, default=1024, help='long side during cache/eval/infer letterbox')
    p.add_argument('--cache-multiple', type=int, default=16, help='pad to multiple during cache/eval/infer letterbox')
    p.add_argument('--cache-scan-dirs', type=str, nargs='*', default=None,
               help='When --build-cache is set, scan these directory(ies) and cache ALL images inside.')
    p.add_argument('--cache-exts', type=str, default='.jpg,.jpeg,.png,.webp',
                help='Comma separated image extensions to include when scanning dirs.')
    p.add_argument('--cache-recursive', action='store_true',
                help='Recursively scan subdirs when building cache from folders.')
    p.add_argument('--sam2-cfg', type=str, required=False, default='sam2.1_hiera_s.yaml')
    p.add_argument('--sam2-ckpt', type=str, required=False, default='checkpoints/sam2.1_hiera_small.pt')
    p.add_argument('--dinov3-model-id', type=str, default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch-size', type=int, default=24)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-2)
    p.add_argument('--kmax_f', type=int, default=8)
    p.add_argument('--kmax_b', type=int, default=8)
    p.add_argument('--lc', type=float, default=0.2, help='weight for count loss')
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--eval-every', type=int, default=1)
    p.add_argument('--save-every', type=int, default=1)
    p.add_argument('--out-dir', type=str, default='outputs/ckpts')
    p.add_argument('--build-cache', action='store_true')
    p.add_argument('--train', action='store_true')
    p.add_argument('--bg-sup-tau')
    p.add_argument('--val-with-sam2', action='store_true')
    p.add_argument('--val-samples', type=int, default=128)
    p.add_argument('--torch-compile', action='store_true')
    # inference demo
    p.add_argument('--infer', action='store_true')
    p.add_argument('--target', type=str, default=None)
    p.add_argument('--target-mask', type=str, default=None)
    p.add_argument('--ref', type=str, default=None)
    p.add_argument('--ref-mask', type=str, default=None)
    p.add_argument('--ckpt', type=str, default=None)
    p.add_argument('--out-prefix', type=str, default='outputs/demo')
    # manifest
    p.add_argument('--manifest-train', type=str, default=None,
                   help='Path to manifest_train.csv produced by dataloader.py preprocess.')
    p.add_argument('--manifest-val', type=str, default=None,
                   help='Path to manifest_val.csv produced by dataloader.py preprocess.')
    p.add_argument('--coco20i-fold', type=int, default=0, choices=[0,1,2,3],
                   help='COCO-20i fold for novel/base split in dataloader.py')
    p.add_argument('--episodes', type=int, default=10000,
                   help='Number of training episodes sampled by OneShotCOCO20iRandom')
    p.add_argument('--role', type=str, default='novel', choices=['novel','base'],
                   help='Use novel or base classes for training')
    p.add_argument('--shard-count', type=int, default=1,
                   help='把要快取的影像切成 shard-count 份（依排序索引取模）')
    p.add_argument('--shard-idx', type=int, default=0,
                   help='本分頁要處理哪一份（0..shard-count-1）')
    p.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.infer:
        run_infer(args)
    elif args.train or args.build_cache:
        # --- 新增：從資料夾掃描並建快取 ---
        if args.build_cache and args.cache_scan_dirs:
            ds = EpisodeDataset([], args.cache_dir, args.dinov3_model_id,
                                args.sam2_cfg, args.sam2_ckpt,
                                build_cache=False,
                                cache_long=args.cache_long,
                                cache_multiple=args.cache_multiple)
            exts = [e for e in args.cache_exts.split(',') if e.strip()]
            paths = scan_images(args.cache_scan_dirs, exts, recursive=args.cache_recursive)
            # for pth in tqdm(paths, desc="Caching images", unit="img"):
            #     ds._build_one(pth)
            # print(f"Cache built for {len(paths)} images → {args.cache_dir}")
            # --- 新增：基本檢查與分片 ---
            if args.shard_count < 1:
                raise ValueError("--shard-count 必須 >= 1")
            if not (0 <= args.shard_idx < args.shard_count):
                raise ValueError("--shard-idx 必須介於 [0, shard-count)")
        
            # 固定排序後依索引取模，平均切片
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
        
        # --- 舊流程（沿用 CSV） ---
        elif args.build_cache and not args.train:
            episodes = read_csv(args.csv)
            _ = EpisodeDataset(episodes, args.cache_dir, args.dinov3_model_id, args.sam2_cfg, args.sam2_ckpt,
                                build_cache=True, cache_long=args.cache_long, cache_multiple=args.cache_multiple)
        else:
            train_loop(args)
    else:
        print('Nothing to do. Use --build-cache, --train or --infer.')