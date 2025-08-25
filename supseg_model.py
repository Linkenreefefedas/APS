# =============================================================
# File: supseg_model.py
# Description: Supervised few-shot segmentation (grid 16x16 + SAM2)
#   - Frozen encoders (SAM2 image encoder via cache, DINOv3 via cache)
#   - Prototypes from support mask (MAP)
#   - Augment target with proto + pseudo mask
#   - Cross-attn fusion (proto tokens -> target feature)
#   - 16x16 3-class head (pos/neg/neutral) with CE
#   - Optionally: get points -> SAM2 prompt encoder/decoder -> Dice (metric or loss)
# Notes:
#   * This file implements the MODEL ONLY. See supseg_train.py for the trainer.
#   * For SAM2 differentiable path, set sam2_mode='torch' and ensure `sam2` library is available.
# =============================================================
from __future__ import annotations
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
ggt=0

# -------------------------- small utils --------------------------
def mask_average_pool(feat: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """feat: [B,C,H,W], mask: [B,1,H,W] in {0,1} -> proto [B,C] (L2-normalized)"""
    w = mask.float()
    num = (feat * w).sum(dim=(2,3))
    den = w.sum(dim=(2,3)).clamp_min(eps)
    proto = num / den
    return F.normalize(proto, dim=1)


def cosine_sim_map(feat: torch.Tensor, proto: torch.Tensor) -> torch.Tensor:
    """feat: [B,C,H,W], proto: [B,C] -> [B,1,H,W] in [-1,1]"""
    f = F.normalize(feat, dim=1)
    p = F.normalize(proto, dim=1).unsqueeze(-1).unsqueeze(-1)
    return (f * p).sum(dim=1, keepdim=True)


def softargmax2d(prob_map: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """prob_map: [B,1,H,W] (>=0) -> [B,2] (x,y) continuous; temperature softmax"""
    B,_,H,W = prob_map.shape
    logits = torch.log(prob_map.clamp_min(1e-8)) / max(1e-6, tau)
    sm = F.softmax(logits.view(B,-1), dim=1).view(B,1,H,W)
    xs = torch.linspace(0, W-1, W, device=prob_map.device).view(1,1,1,W).expand(B,1,H,W)
    ys = torch.linspace(0, H-1, H, device=prob_map.device).view(1,1,H,1).expand(B,1,H,W)
    x = (sm * xs).sum(dim=(2,3))
    y = (sm * ys).sum(dim=(2,3))
    return torch.stack([x, y], dim=-1)


def make_grid_labels(gt_mask_512: torch.Tensor, pos_th: float = 0.7, neg_th: float = 0.3) -> torch.Tensor:
    """gt_mask_512: [B,1,512,512] float/bool -> [B,16,16] long in {0(pos),1(neg),2(neu)}"""
    ratio = F.avg_pool2d(gt_mask_512.float(), kernel_size=32, stride=32)  # [B,1,16,16]
    y = torch.full_like(ratio, 2)  # neutral
    y[ratio >= pos_th] = 0
    y[ratio <= neg_th] = 1
    return y.squeeze(1).long()


def dice_loss_from_logits(pred_logits: torch.Tensor, gt_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """pred_logits: [B,1,h,w]; gt_mask: [B,1,H,W] in {0,1}. Returns scalar dice loss."""
    pred = torch.sigmoid(pred_logits)
    g = F.interpolate(gt_mask.float(), size=pred.shape[-2:], mode='nearest')
    inter = (pred*g).sum(dim=(1,2,3))
    denom = pred.sum(dim=(1,2,3)) + g.sum(dim=(1,2,3)) + eps
    return (1 - (2*inter + eps)/denom).mean()


# -------------------------- blocks --------------------------
class SharedProjector(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.proj(x)

class SinCosPosEnc2D(nn.Module):
    """Fixed 2D sine-cosine absolute positional encoding: [1,4F,H,W]."""
    def __init__(self, num_freqs: int = 16):
        super().__init__()
        self.num_freqs = num_freqs

    @torch.no_grad()
    def forward(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, H, device=device)
        x = torch.linspace(-1.0, 1.0, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        freqs = 2.0 ** torch.arange(self.num_freqs, device=device).float()
        xx = xx.unsqueeze(-1) * freqs
        yy = yy.unsqueeze(-1) * freqs
        pe = torch.cat([xx.sin(), xx.cos(), yy.sin(), yy.cos()], dim=-1)  # [H,W,4F]
        return pe.permute(2, 0, 1).unsqueeze(0).contiguous()  # [1,4F,H,W]


class ConvPosBias(nn.Module):
    """Depthwise Conv positional bias: x + DWConv(x) (k=3)。"""
    def __init__(self, dim: int, k: int = 3):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=k, padding=k//2, groups=dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dw(x)
    
class HierProj(nn.Module):
    """
    階層式融合：
    (SAM, DINO) → mid1 → (+proto) → mid2 → (+pseudo, coord, extra) → proj_dim
    """
    def __init__(self, proj_dim: int, e_channels: int = 0, use_coord: bool = True):
        super().__init__()
        self.use_coord = use_coord
        self.e_channels = e_channels
        d_mid = proj_dim          # 中間節點通道
        d_small = max(32, proj_dim // 4)

        # stage 1: 影像 backbone 融合
        self.s1 = nn.Sequential(
            nn.Conv2d(256+768, d_mid, 1, bias=False),
            nn.BatchNorm2d(d_mid), nn.ReLU(inplace=True)
        )
        # stage 2: 加入 proto
        self.s2 = nn.Sequential(
            nn.Conv2d(d_mid+512, d_mid, 1, bias=False),  # ← 接收 FG(256)+BG(256)
            nn.BatchNorm2d(d_mid), nn.ReLU(inplace=True)
        )
        
        # 將小提示壓小再融合
        self.pseudo_proj = nn.Conv2d(1, d_small, 1, bias=False)
        self.coord_proj  = nn.Conv2d(2, d_small, 1, bias=False) if use_coord else None
        self.extra_proj  = nn.Conv2d(e_channels, d_small, 1, bias=False) if e_channels>0 else None

        # stage 3: 最終融合到 proj_dim
        in_s3 = d_mid + d_small  # pseudo
        if use_coord:   in_s3 += d_small
        if e_channels>0:in_s3 += d_small

        self.s3 = nn.Sequential(
            nn.Conv2d(in_s3, proj_dim, 1, bias=False),
            nn.BatchNorm2d(proj_dim), nn.ReLU(inplace=True)
        )

    def forward(self, sam32, dn32, proto_broad, pseudo, coord=None, extra=None):
        x = self.s1(torch.cat([sam32, dn32], dim=1))
        x = self.s2(torch.cat([x, proto_broad], dim=1))

        tips = [self.pseudo_proj(pseudo)]
        if self.use_coord and coord is not None:
            tips.append(self.coord_proj(coord))
        if self.e_channels>0 and extra is not None:
            tips.append(self.extra_proj(extra))

        x = self.s3(torch.cat([x] + tips, dim=1))
        return x
    
class CrossAttentionFuse(nn.Module):
    def __init__(self, dim: int, heads: int = 8, pe_freqs: int = 16):
        super().__init__()
        # 絕對 2D 位置編碼 → 1x1 投影到 dim，殘差相加
        self.abs_pe_gen = SinCosPosEnc2D(num_freqs=pe_freqs)
        self.abs_pe_proj = nn.Conv2d(4 * pe_freqs, dim, kernel_size=1, bias=False)
        # 局部位置偏置（depthwise conv）
        self.local_bias_q = ConvPosBias(dim)
        self.local_bias_kv = ConvPosBias(dim)
        # q/k/v 與 MHA
        self.q = nn.Conv2d(dim, dim, 1, bias=False)
        self.k = nn.Conv2d(dim, dim, 1, bias=False)
        self.v = nn.Conv2d(dim, dim, 1, bias=False)
        self.mha = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.1)
        # FFN
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1), nn.ReLU(inplace=True),
            nn.Dropout2d(0.10),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.Dropout2d(0.10),
        )
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(self, ref: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        B, C, H, W = tgt.shape
        pe = self.abs_pe_proj(self.abs_pe_gen(H, W, tgt.device))  # [1,dim,H,W]
        tgt_pe = self.local_bias_q(tgt + pe)
        ref_pe = self.local_bias_kv(ref + pe)

        q = self.q(tgt_pe).flatten(2).transpose(1, 2)  # [B,HW,C]
        k = self.k(ref_pe).flatten(2).transpose(1, 2)
        v = self.v(ref_pe).flatten(2).transpose(1, 2)
        out, _ = self.mha(self.norm_q(q), self.norm_kv(k), self.norm_kv(v))
        out = out.transpose(1, 2).view(B, C, H, W)
        out = out + tgt  # 殘差
        return out + self.ffn(out)

class GridHead(nn.Module):
    def __init__(self, in_ch: int, num_classes: int = 3):
        super().__init__()
        self.down = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1, bias=False)  # 32->16
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch//2, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.cls = nn.Conv2d(in_ch//2, num_classes, 1)

    def forward(self, x32: torch.Tensor) -> torch.Tensor:
        x = self.down(x32)     # 替代 avg_pool2d
        x = self.block(x)
        return self.cls(x)

# -------------------------- SAM2 wrappers --------------------------
class Sam2TorchWrapper(nn.Module):
    """
    Differentiable SAM2 prompt->mask path (best-effort). Requires the `sam2` library.
    Fallback to non-diff predictor will be handled outside the model when needed.
    """
    def __init__(self, cfg_yaml: str, ckpt_path: str, freeze: bool = True):
        super().__init__()
        try:
            from sam2.build_sam import build_sam2
        except Exception as e:
            raise ImportError("sam2 package not available for differentiable path") from e
        model = build_sam2(cfg_yaml, ckpt_path)
        self.image_encoder = model.image_encoder
        self.prompt_encoder = model.sam_prompt_encoder
        self.mask_decoder = model.sam_mask_decoder
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward_with_embeddings(self, image_embeddings: torch.Tensor,
                                point_coords: torch.Tensor,
                                point_labels: torch.Tensor,
                                orig_hw: Tuple[int,int]) -> torch.Tensor:
        """
        image_embeddings: [B,C,Hf,Wf] (no grad is fine)
        point_coords: [B,K,2] in image pixels (orig_hw scale)
        point_labels: [B,K] in {1,0}
        returns mask_logits: [B,1,h,w]
        """
        B = image_embeddings.size(0)
        # Encode sparse prompts
        pe = []
        for b in range(B):
            pts = point_coords[b].float()
            lbl = point_labels[b].float()
            # SAM2 prompt encoder expects (B=1) tensors
            sp = self.prompt_encoder(points=(pts.unsqueeze(0), lbl.unsqueeze(0)), boxes=None, masks=None)
            pe.append(sp)
        # Merge batch prompt embeddings (keeping API generic)
        # Many SAM2 impls return tuple(sparse, dense)
        sparse_list = [t[0] if isinstance(t, (tuple, list)) else t for t in pe]
        dense_list  = [t[1] if (isinstance(t, (tuple, list)) and len(t)>1) else None for t in pe]
        sparse = torch.cat(sparse_list, dim=0)
        dense  = dense_list[0] if all(d is None for d in dense_list) else torch.cat([d if d is not None else torch.zeros_like(dense_list[0]) for d in dense_list], dim=0)
        # Decode
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=None,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False
        )
        # Upsample to a fixed size (256x256 typical), caller can resize to GT
        return low_res_masks  # [B,1,h,w]


# -------------------------- Main model --------------------------
class SupSegGridSAM2(nn.Module):
    """
    Inputs (per batch):
      - sam_tgt: [B,256,64,64]
      - dino_tgt: [B,768,32,32]
      - proto_fg_sam: [B,256]; proto_bg_sam: [B,256]
      - proto_fg_dn:  [B,768]; proto_bg_dn:  [B,768]
      - (optional) extra_maps32: [B,E,32,32] (e.g., edge, support-cond upsampled)
    Behavior:
      - Downsample SAM 64->32, concat with DINO 32, + proto broadcasts, + pseudo mask from SAM proto
      - Shared 1x1 proj -> Cross-attn fuse (ref path omitted; we fuse via proto tokens)
      - Grid head -> [B,3,16,16]
      - If sam2_torch provided, can compute differentiable mask from points.
    """
    def __init__(self, proj_dim: int = 256, pos_th: float = 0.7, neg_th: float = 0.3,
                 k_pos: int = 1, k_neg: int = 1, tau: float = 1.0,
                 lambda_ce: float = 1.0, lambda_dice: float = 1.0,
                 sam2_torch: Optional[Sam2TorchWrapper] = None,sam2_pred=None):
        super().__init__()
        self.pos_th, self.neg_th = float(pos_th), float(neg_th)
        self.k_pos, self.k_neg = int(k_pos), int(k_neg)
        self.tau = float(tau)
        self.lambda_ce = float(lambda_ce)
        self.lambda_dice = float(lambda_dice)
        self.sam2_torch = sam2_torch
        self.sam2_pred = sam2_pred
        self.register_buffer("pseudo_temp", torch.tensor(0.20))
        self.register_buffer("pseudo_bias", torch.tensor(0.00))
        

        # SAM 64->32
        self.sam_down = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.ReLU(inplace=True)
        )
        # Input channels: SAM(256) + DINO(768) + proto(256) + pseudo(1) + optional extra E
        # We'll dynamically add E at forward by concatenation then projecting with 1x1.
        self.share_proj = SharedProjector(256 + 768 + 256 + 1, proj_dim)
        self.fuse = CrossAttentionFuse(proj_dim, heads=8)
        self.grid_head = GridHead(proj_dim, num_classes=3)

        # Map prototypes (sam+dn) into proto tokens at proj_dim for cross-attn (2 tokens: fg,bg)
        self.map_sam = nn.Linear(256, proj_dim)
        self.map_dn  = nn.Linear(768, proj_dim)
        self.proto_merge = nn.Linear(2*proj_dim, proj_dim)

    def _make_pseudo(self, sam32, proto_fg, proto_bg=None):
        sim_fg = cosine_sim_map(sam32, proto_fg)
        if proto_bg is not None:
            sim_bg = cosine_sim_map(sam32, proto_bg)
            t = self.pseudo_temp.clamp_min(1e-3)
            b = self.pseudo_bias
            s = torch.sigmoid((sim_fg - sim_bg - b) / t)
        else:
            s = (sim_fg + 1.0) / 2.0
        return s

    def _proto_tokens(self, p_fg_sam: torch.Tensor, p_bg_sam: torch.Tensor,
                      p_fg_dn: torch.Tensor,  p_bg_dn: torch.Tensor) -> torch.Tensor:
        fg = torch.cat([self.map_sam(p_fg_sam), self.map_dn(p_fg_dn)], dim=-1)
        bg = torch.cat([self.map_sam(p_bg_sam), self.map_dn(p_bg_dn)], dim=-1)
        fg = self.proto_merge(fg)
        bg = self.proto_merge(bg)
        # shape: [B,2,proj_dim]
        return torch.stack([fg, bg], dim=1)

    def forward_grid(self,
                     sam_tgt_64: torch.Tensor,
                     dino_tgt_32: torch.Tensor,
                     proto_fg_sam: torch.Tensor,
                     proto_bg_sam: torch.Tensor,
                     proto_fg_dn:  torch.Tensor,
                     proto_bg_dn:  torch.Tensor,
                     extra_maps32: Optional[torch.Tensor] = None
                     ) -> torch.Tensor:
        """Returns grid logits [B,3,16,16]."""
        B = sam_tgt_64.size(0)
        sam32 = self.sam_down(sam_tgt_64)               # [B,256,32,32]
        dn32  = dino_tgt_32                             # [B,768,32,32]
        pseudo = self._make_pseudo(sam32, proto_fg_sam, proto_bg_sam)  # [B,1,32,32]
        p_broad = proto_fg_sam.view(B, -1, 1, 1).expand(-1, proto_fg_sam.size(1), 32, 32)
        x = torch.cat([sam32, dn32, p_broad, pseudo], dim=1)
        if extra_maps32 is not None:
            x = torch.cat([x, extra_maps32], dim=1)
        x = self.share_proj(x)  # [B,proj,32,32]
        # proto tokens -> [B,2,proj]
        ptoks = self._proto_tokens(proto_fg_sam, proto_bg_sam, proto_fg_dn, proto_bg_dn)
        # fold tokens into spatial fusion by treating ref as two tokens tiled
        ref = ptoks.mean(dim=1).unsqueeze(-1).unsqueeze(-1).expand(-1, x.size(1), 32, 32)
        x = self.fuse(ref, x)
        return self.grid_head(x)  # [B,3,16,16]
    
class SupSegGridSAM2Spatial(SupSegGridSAM2):
    """
    與 SupSegGridSAM2 介面相同；多了：
        - use_coord: 是否加 (x,y) CoordConv 通道
        - e_channels: 若你會傳 extra_maps32，請在建構子告知通道數，避免 in_ch 不匹配
    """
    def __init__(self, proj_dim: int = 256, pos_th: float = 0.7, neg_th: float = 0.3,
                    k_pos: int = 1, k_neg: int = 1, tau: float = 1.0,
                    lambda_ce: float = 1.0, lambda_dice: float = 1.0,
                    sam2_torch: Optional[Sam2TorchWrapper] = None, sam2_pred=None,
                    use_coord: bool = True, e_channels: int = 2, pe_freqs: int = 16, lambda_aux: float = 0.3):
        super().__init__(proj_dim, pos_th, neg_th, k_pos, k_neg, tau,
                            lambda_ce, lambda_dice, sam2_torch, sam2_pred)
        self.use_coord = bool(use_coord)
        self.e_channels = int(e_channels)
        self.lambda_aux = float(lambda_aux)  # auxiliary mask loss weight


        # 重新設定 share_proj 的輸入通道（原: 256 + 768 + 256 + 1）
        base_in = 256 + 768 + 256 + 1
        if self.use_coord:
            base_in += 2  # (x,y)
        base_in += self.e_channels  # 若會傳 extra_maps32

        self.share_proj = HierProj(proj_dim=proj_dim, e_channels=self.e_channels, use_coord=self.use_coord)

        # 替換 fuse 為帶 PE 的版本
        self.fuse = CrossAttentionFuse(proj_dim, heads=8, pe_freqs=pe_freqs)

        # Local conv path（保留空間局部性）
        self.local_path = nn.Sequential(
            nn.Conv2d(proj_dim, proj_dim, 3, padding=1, groups=proj_dim, bias=False),  # DW
            nn.BatchNorm2d(proj_dim),
            nn.GELU(),
            nn.Conv2d(proj_dim, proj_dim, 1, bias=False),  # PW
            nn.BatchNorm2d(proj_dim),
        )

        # 門控融合（SE風格）：根據空間全域訊息調整 attn/local 權重
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(proj_dim * 2, proj_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(proj_dim // 4, 2, 1),
            nn.Sigmoid()  # 兩路權重 ∈ (0,1)
        )

        # 輕量 mask head（預設輸出 1 通道 logits）
        self.aux_mask_head = nn.Sequential(
            nn.Conv2d(proj_dim, proj_dim//2, 3, padding=1, bias=False),
            nn.GroupNorm(32, proj_dim//2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 32→64
            nn.Conv2d(proj_dim//2, 1, 1)  # logits at 64×64
        )
        
        # grid head 可沿用；若想更在地，可換成擴張卷積/多層
        # self.grid_head = GridHead(proj_dim, num_classes=3)

    @staticmethod
    def _coord_maps(B: int, H: int, W: int, device) -> torch.Tensor:
        """產生 [B,2,H,W]，x,y ∈ [-1,1]。"""
        y = torch.linspace(-1.0, 1.0, H, device=device)
        x = torch.linspace(-1.0, 1.0, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coord = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)  # [B,2,H,W]
        return coord.contiguous()

    def forward_grid(self,
                        sam_tgt_64: torch.Tensor,
                        dino_tgt_32: torch.Tensor,
                        proto_fg_sam: torch.Tensor,
                        proto_bg_sam: torch.Tensor,
                        proto_fg_dn:  torch.Tensor,
                        proto_bg_dn:  torch.Tensor,
                        extra_maps32: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        回傳 [B,3,16,16]；與父類別相同，但在 32x32 特徵融合時加入：
            CoordConv + PE CrossAttn + Local Conv + 門控融合。
        """
        B = sam_tgt_64.size(0)
        sam32 = self.sam_down(sam_tgt_64)  # [B,256,32,32]
        dn32  = dino_tgt_32               # [B,768,32,32]
        pseudo = self._make_pseudo(sam32, proto_fg_sam, proto_bg_sam)  # [B,1,32,32]

        # 先算兩張相似度圖
        sim_fg = cosine_sim_map(sam32, proto_fg_sam)   # [B,1,32,32]
        sim_bg = cosine_sim_map(sam32, proto_bg_sam)   # [B,1,32,32]
        
        # 新增 BG 原型 broadcast
        p_broad_fg = proto_fg_sam.view(B, -1, 1, 1).expand(-1, proto_fg_sam.size(1), 32, 32)
        p_broad_bg = proto_bg_sam.view(B, -1, 1, 1).expand(-1, proto_bg_sam.size(1), 32, 32)
        
        
        # CoordConv
        coord = self._coord_maps(B, 32, 32, sam32.device) if self.use_coord else None

        # 額外圖層
        extra_list = []
        if extra_maps32 is not None:
            extra_list.append(extra_maps32)         # 既有的 extra
        extra_list += [sim_fg, sim_bg]              # 新增兩張相似度圖
        extra = torch.cat(extra_list, dim=1) if len(extra_list) > 0 else None
        
        x = self.share_proj(
            sam32, dn32,
            torch.cat([p_broad_fg, p_broad_bg], dim=1),
            pseudo,
            coord=coord,
            extra=extra
        )

        # Proto tokens → 產出 ref（保持你原本的作法）
        ptoks = self._proto_tokens(proto_fg_sam, proto_bg_sam, proto_fg_dn, proto_bg_dn)
        ref = ptoks.mean(dim=1).unsqueeze(-1).unsqueeze(-1).expand(-1, x.size(1), 32, 32)

        # 兩路：Attn 與 Local
        x_attn  = self.fuse(ref, x)          # [B,proj,32,32]
        x_local = self.local_path(x) + x     # 在地殘差

        # 門控融合
        g = self.gate(torch.cat([x_attn, x_local], dim=1))   # [B,2,1,1]；兩路權重
        wa = g[:, 0:1]; wl = g[:, 1:2]
        x_fused = wa * x_attn + wl * x_local

        # 進 grid head（16x16 三分類）+ 輕量 aux mask（32x32 前景）
        grid_logits     = self.grid_head(x_fused)            # [B,3,16,16]
        aux_mask_logits = self.aux_mask_head(x_fused)        # [B,1,32,32]
        return grid_logits, aux_mask_logits

    # def points_from_grid(self, grid_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     grid_logits: [B,3,16,16]
    #     回傳:
    #       points_xy: [B,K,2]  (K = k_pos + k_neg)，影像像素座標(對 512x512)
    #       labels:    [B,K]    (1=pos, 0=neg)
    #     作法:
    #       1) 在 16x16 上對 pos/neg 各自做 NMS 取 top-k
    #       2) 對每個峰值開一個小窗 (例如 3x3/5x5)，做局部 soft-argmax 精修成連續座標
    #       3) 映射到 512px（cell 中心 + 次像素偏移）
    #     """
    #     B,_,H,W = grid_logits.shape  # H=W=16
    #     P = F.softmax(grid_logits, dim=1)
    #     pos_map = P[:, 0:1]  # [B,1,16,16]
    #     neg_map = P[:, 1:2]  # 注意：你設計是 0=pos,1=neg,2=neutral；若不同請對應調整
    
    #     def _topk_peaks(heat: torch.Tensor, k: int, th: float = None):
    #         if k <= 0:
    #             return heat.new_zeros(B,0,2), heat.new_zeros(B,0)
    #         pool = F.max_pool2d(heat, 6, 1, 1)
    #         keep = (heat == pool) * heat  # 峰值
    #         if th is not None:
    #             keep = torch.where(keep > th, keep, keep.new_zeros(()))
    #         scores = keep.view(B, -1)
    #         # 若全都 <= th，topk 仍會回傳（但都是 0），可搭配「保底」機制
    #         topv, topi = torch.topk(scores, k=min(k, scores.size(1)), dim=1)
    #         ys, xs = (topi // W).float(), (topi % W).float()
    #         return torch.stack([xs, ys], dim=-1), topv
    
    #     pos_xy_g, _ = _topk_peaks(pos_map, self.k_pos,0.5)  # [B,k_pos,2]
    #     neg_xy_g, _ = _topk_peaks(neg_map, self.k_neg,0.7)  # [B,k_neg,2]

    #     if self.k_pos > 0 and (pos_xy_g.numel() == 0 or pos_xy_g.size(1) == 0):
    #         flat = pos_map.view(B, -1)              # [B, H*W]
    #         topi = flat.argmax(dim=1)               # [B]
    #         ys = (topi // W).float().unsqueeze(1)   # [B,1]
    #         xs = (topi %  W).float().unsqueeze(1)   # [B,1]
    #         pos_xy_g = torch.stack([xs, ys], dim=-1)  # [B,1,2]
    
    #     # 局部精修：對每個點在 16x16 上開一個小窗 (r=1 → 3x3)，做局部 soft-argmax
    #     r = 1  # 可調 1/2 (3x3 或 5x5)
    #     def _refine_local(heat: torch.Tensor, xy_g: torch.Tensor):
    #         # heat: [B,1,H,W]; xy_g: [B,k,2] (grid coords, 整數或浮點)
    #         if xy_g.numel() == 0:
    #             return xy_g
    #         B,k,_ = xy_g.shape
    #         # 取整數中心
    #         cx = xy_g[...,0].round().clamp_(0, W-1).long()  # [B,k]
    #         cy = xy_g[...,1].round().clamp_(0, H-1).long()
    #         # 擷取 local patch
    #         patches = []
    #         for b in range(B):
    #             ph = []
    #             for i in range(k):
    #                 x0 = max(0, cx[b,i]-r); x1 = min(W, cx[b,i]+r+1)
    #                 y0 = max(0, cy[b,i]-r); y1 = min(H, cy[b,i]+r+1)
    #                 patch = heat[b:b+1, :, y0:y1, x0:x1]  # [1,1,hh,ww]
    #                 # local soft-argmax
    #                 hh, ww = patch.shape[-2:]
    #                 logits = torch.log(patch.clamp_min(1e-8)).view(1, -1)
    #                 sm = F.softmax(logits / max(1e-6, self.tau), dim=1).view(1,1,hh,ww)
    #                 xs = torch.linspace(0, ww-1, ww, device=heat.device).view(1,1,1,ww)
    #                 ys = torch.linspace(0, hh-1, hh, device=heat.device).view(1,1,hh,1)
    #                 dx = (sm*xs).sum(dim=(2,3))  # [1,1]
    #                 dy = (sm*ys).sum(dim=(2,3))
    #                 gx = x0 + dx.squeeze()  # 連續 grid 座標
    #                 gy = y0 + dy.squeeze()
    #                 ph.append(torch.stack([gx, gy], dim=-1))
    #             patches.append(torch.stack(ph, dim=0))  # [k,2]
    #         return torch.stack(patches, dim=0)  # [B,k,2]
        
    #     pos_xy_g = _refine_local(pos_map, pos_xy_g)  # [B,k_pos,2]
    #     neg_xy_g = _refine_local(neg_map, neg_xy_g)  # [B,k_neg,2]
    
    #     # 併在一起（用實際數量 p, n）
    #     p = pos_xy_g.size(1)                 # 實際正點數
    #     n = neg_xy_g.size(1)                 # 實際負點數
    #     pts_g = torch.cat([pos_xy_g, neg_xy_g], dim=1)  # [B, p+n, 2]
    #     K = p + n
        
    #     # grid → pixel（維持你的 512 映射；也可寫成用 H/W 自動換算）
    #     pts_px = (pts_g + 0.5) * (512.0 / 16.0)         # [B, K, 2]
        
    #     # labels：前 p 個為 1(正)，其餘為 0(負)
    #     labels = torch.zeros(B, K, device=grid_logits.device, dtype=torch.long)
    #     if p > 0:
    #         labels[:, :p] = 1
        
    #     return pts_px, labels

    def points_from_grid(self, grid_logits: torch.Tensor):
      """
      輸入:  grid_logits [B,3,16,16]
      輸出:  pts_px_list: List[Tensor (K_i,2)]  (像素座標，對 cache_long；K_i ∈ [1, kp+kn])
            lbl_list:    List[Tensor (K_i,)]    (1=pos, 0=neg)
      規則:
        - 先信心門檻(th_pos/th_neg)再 NMS；只收合格者，kp/kn 為上限，不硬湊。
        - 最少 1 點：若兩類皆為 0，取 pos_map 全局 argmax 作為唯一正點。
      """
      import torch
      import torch.nn.functional as F

      B, C, H, W = grid_logits.shape         # 期望 H=W=16
      tau = float(getattr(self, "tau", 1.0))
      cache_long = int(getattr(self, "cache_long", 512))

      # --- softmax 前溫度縮放（僅 pos/neg），neutral 不動 ---
      logits = grid_logits.clone()
      logits[:, 0:2] = logits[:, 0:2] / max(1e-6, tau)
      P = F.softmax(logits, dim=1)
      pos_map = P[:, 0:1]                     # [B,1,H,W]
      neg_map = P[:, 1:2]

      # 參數（若未在 model 上設定則給預設）
      kp, kn = int(getattr(self, "k_pos", 0)), int(getattr(self, "k_neg", 0))  # 上限
      cand_factor = int(getattr(self, "cand_factor", 5))
      min_dist    = float(getattr(self, "nms_min_dist", 2.0))   # 以 grid 為單位
      th_pos      = float(getattr(self, "peak_th_pos", 0.5))    # 信心門檻
      th_neg      = float(getattr(self, "peak_th_neg", 0.8))
      r           = int(getattr(self, "refine_r", 1))           # 1→3x3, 2→5x5

      # --- 候選：先 3×3 極大值，再套信心門檻，最後取較多候選 ---
      def _candidates(heat: torch.Tensor, kmax: int, th: float):
          if kmax <= 0:
              z = heat.new_zeros((heat.size(0), 0, 2)); v = heat.new_zeros((heat.size(0), 0))
              return z, v
          pool = F.max_pool2d(heat, 3, 1, 1)
          keep = (heat == pool) * heat
          keep = torch.where(keep >= th, keep, keep.new_zeros(keep.shape))
          scores = keep.view(heat.size(0), -1)                 # [B,H*W]，門檻以下已是 0
          topk = min(max(kmax * cand_factor, kmax), scores.size(1))
          topv, topi = torch.topk(scores, k=topk, dim=1)
          ys, xs = (topi // W).float(), (topi % W).float()
          coords = torch.stack([xs, ys], dim=-1)               # [B,topk,2]
          return coords, topv                                  # topv 會含 0（未達門檻）

      pos_cand, pos_v = _candidates(pos_map, kp, th_pos)
      neg_cand, neg_v = _candidates(neg_map, kn, th_neg)

      # --- 類內 greedy NMS（只保留「分數 > 0」的合格候選；不硬湊到定長） ---
      def _nms_single(coords_b, scores_b, k_keep, min_d):
          # 先過濾未達門檻（score=0）
          if scores_b.numel() == 0:
              return coords_b.new_zeros((0, 2)), scores_b.new_zeros((0,))
          valid = scores_b > 0
          coords_b = coords_b[valid]
          scores_b = scores_b[valid]
          if k_keep <= 0 or coords_b.numel() == 0:
              return coords_b.new_zeros((0, 2)), scores_b.new_zeros((0,))
          order = scores_b.argsort(descending=True)
          coords_b = coords_b[order]; scores_b = scores_b[order]
          keep_idx = []
          for i in range(coords_b.size(0)):
              if not keep_idx:
                  keep_idx.append(i)
              else:
                  d = torch.cdist(coords_b[i:i+1], coords_b[keep_idx]).squeeze(0)
                  if d.min() >= min_d:
                      keep_idx.append(i)
              if len(keep_idx) >= k_keep:  # 上限，不硬湊
                  break
          keep_idx = torch.tensor(keep_idx, device=coords_b.device, dtype=torch.long)
          return coords_b[keep_idx], scores_b[keep_idx]

      pos_list, neg_list = [], []
      for b in range(B):
          pc, _ = _nms_single(pos_cand[b], pos_v[b], kp, min_dist)
          nc, _ = _nms_single(neg_cand[b], neg_v[b], kn, min_dist)
          pos_list.append(pc); neg_list.append(nc)

      #  最少 1「正」點：只要正點為 0，就取 pos_map 全域 argmax 補 1 個正點
      flat = pos_map.view(B, -1)
      topi = flat.argmax(dim=1)
      ys0 = (topi // W).float(); xs0 = (topi % W).float()
      gmax = torch.stack([xs0, ys0], dim=-1)  # [B,2]
      for b in range(B):
        if pos_list[b].size(0) == 0:
            pos_list[b] = gmax[b:b+1]
              
      # --- 局部 refine（各自用本類 heat；soft-argmax with τ） ---
      def _refine_local(heat_b: torch.Tensor, xy_g_b: torch.Tensor):
          if xy_g_b.numel() == 0: return xy_g_b
          k = xy_g_b.size(0)
          cx = xy_g_b[:, 0].round().clamp_(0, W-1).long()
          cy = xy_g_b[:, 1].round().clamp_(0, H-1).long()
          out = []
          for i in range(k):
              x0 = max(0, cx[i].item()-r); x1 = min(W, cx[i].item()+r+1)
              y0 = max(0, cy[i].item()-r); y1 = min(H, cy[i].item()+r+1)
              patch = heat_b[:, :, y0:y1, x0:x1]
              hh, ww = patch.shape[-2:]
              logits = torch.log(patch.clamp_min(1e-8)).view(1, -1)
              sm = F.softmax(logits / max(1e-6, tau), dim=1).view(1,1,hh,ww)
              xs = torch.linspace(0, ww-1, ww, device=heat_b.device).view(1,1,1,ww)
              ys = torch.linspace(0, hh-1, hh, device=heat_b.device).view(1,1,hh,1)
              dx = (sm*xs).sum(dim=(2,3)); dy = (sm*ys).sum(dim=(2,3))
              gx = x0 + dx.squeeze(); gy = y0 + dy.squeeze()
              out.append(torch.stack([gx, gy], dim=-1))
          return torch.stack(out, dim=0)  # [k,2]

      pos_refined, neg_refined = [], []
      for b in range(B):
          pos_refined.append(_refine_local(pos_map[b:b+1], pos_list[b]))
          neg_refined.append(_refine_local(neg_map[b:b+1], neg_list[b]))

      # --- grid→像素（對 cache_long） ---
      sx = float(cache_long) / W; sy = float(cache_long) / H
      def _grid2px(xy):
          if xy.numel() == 0: return xy
          return torch.stack([(xy[...,0]+0.5)*sx, (xy[...,1]+0.5)*sy], dim=-1)

      pts_px_list, lbl_list = [], []
      for b in range(B):
          p_px = _grid2px(pos_refined[b]).to(grid_logits.dtype)
          n_px = _grid2px(neg_refined[b]).to(grid_logits.dtype)
          # 合併（先正後負；不硬湊到 kp/kn）
          if p_px.numel() == 0 and n_px.numel() == 0:
              # 正常不會發生（上面已保底），保險再給一個 (0,0) 的正點
              pts_px = grid_logits.new_zeros((1,2))
              lbl    = grid_logits.new_zeros((1,), dtype=torch.long) + 1
          else:
              pts_px = torch.cat([p_px, n_px], dim=0)
              lbl    = torch.cat([
                          torch.ones(p_px.size(0), dtype=torch.long, device=grid_logits.device),
                          torch.zeros(n_px.size(0), dtype=torch.long, device=grid_logits.device)
                      ], dim=0)
          # 保證每張圖至少 1 點
          if lbl.numel() == 0:
              pts_px = grid_logits.new_zeros((1,2))
              lbl    = grid_logits.new_zeros((1,), dtype=torch.long) + 1
          pts_px_list.append(pts_px)
          lbl_list.append(lbl)

      return pts_px_list, lbl_list
      
    def focal_loss_multiclass_ignore_neutral(
        self,
        logits: torch.Tensor,         # [B, 3, H, W]，你的 grid_logits
        targets: torch.Tensor,        # [B, H, W]，0=pos, 1=neg, 2=neutral
        gamma: float = 2.0,           # 焦點參數，越大越聚焦困難樣本
        alpha_pos: float = 3.0,       # 正類權重（> neg）
        alpha_neg: float = 1.0,       # 負類權重
        alpha_neu: float = 0.25, normalize_by_weight: bool = True
    ):
        assert logits.dim() == 4 and logits.size(1) == 3, f"logits 應為 [B,3,H,W]，got {tuple(logits.shape)}"
        if targets.dim() == 4 and targets.size(1) == 1:
            targets = targets.squeeze(1)  # [B,1,H,W] -> [B,H,W]
        assert targets.dim() == 3, f"targets 應為 [B,H,W]，got {tuple(targets.shape)}"
    
        B, _, H, W = logits.shape
        assert targets.shape[0] == B and targets.shape[-2:] == (H, W), \
            f"batch/HW 不對齊：targets={tuple(targets.shape)}, logits={tuple(logits.shape)}"
    
        # --- 基本量 ---
        y = targets.to(logits.device).long()               # 確保在同一裝置
        logp = F.log_softmax(logits, dim=1)               # [B,3,H,W]
        logp_t = logp.gather(1, y.unsqueeze(1)).squeeze(1)  # [B,H,W]
        pt = logp_t.exp()
    
        # --- 類別 α 權重（broadcast 到 [B,H,W]）---
        alpha = logits.new_tensor([alpha_pos, alpha_neg, alpha_neu])  # [3] same device/dtype
        alpha_t = alpha[y]                                      # [B,H,W]  ← 這行取代 gather
    
        # --- Focal ---
        loss_pix = -(alpha_t * (1.0 - pt).pow(gamma) * logp_t)  # [B,H,W]

        # 不再忽略 neutral，用加權樣本數做歸一化
        if normalize_by_weight:
            denom = (alpha_t).sum().clamp_min(1.0)
        else:
            denom = torch.tensor(float(y.numel()), device=logits.device)
        
        return loss_pix.sum() / denom
    
    def forward(self,
                sam_tgt_64: torch.Tensor,
                dino_tgt_32: torch.Tensor,
                proto_fg_sam: torch.Tensor,
                proto_bg_sam: torch.Tensor,
                proto_fg_dn:  torch.Tensor,
                proto_bg_dn:  torch.Tensor,
                tgt_gt_mask_512: Optional[torch.Tensor] = None,
                extra_maps32: Optional[torch.Tensor] = None,
                image_embeddings_for_sam2: Optional[torch.Tensor] = None,
                ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with: grid_logits, ce_loss, dice_loss (0 if unavailable), total_loss, (optional) pred_mask_logits
        """
        grid_logits, aux_mask_logits = self.forward_grid(
            sam_tgt_64, dino_tgt_32, proto_fg_sam, proto_bg_sam, proto_fg_dn, proto_bg_dn, extra_maps32
        )
        out = {
            "grid_logits": grid_logits,
            "loss_ce": grid_logits.new_tensor(0.0),
            "loss_dice": grid_logits.new_tensor(0.0),
            "loss_aux_dice": grid_logits.new_tensor(0.0)
        }
        if tgt_gt_mask_512 is not None:
            y = make_grid_labels(tgt_gt_mask_512, self.pos_th, self.neg_th)
            #ce = F.cross_entropy(grid_logits, y)
            ce = self.focal_loss_multiclass_ignore_neutral(
                grid_logits, y,
                gamma=2.0,      # 常用 1.5~2.5
                alpha_pos=3.0,  # 正類較重
                alpha_neg=1.0,  # 負類較輕
                alpha_neu=0.2,  # neutral 類輕（但非 0）
            )
            
            out["loss_ce"] = ce
        else:
            ce = grid_logits.new_tensor(0.0)
            out["loss_ce"] = ce

        dice = grid_logits.new_tensor(0.0)
        out["loss_dice"] = dice

        pred_mask_logits = None
        B = grid_logits.size(0)
        if (tgt_gt_mask_512 is not None):
            # 取得點
            pts_list, lbl_list = self.points_from_grid(grid_logits)
            
            if self.sam2_pred is not None:
                # sam2_image_predictor 已經在外部 set_image_batch_from_cache(...) 好這一個 batch
                # 準備 list[tensor]，每張一個 (保持 torch，不 numpy)
                point_coords_batch = [p.to(dtype=torch.float32) for p in pts_list]
                point_labels_batch = [l.to(dtype=torch.long)    for l in lbl_list]

                global ggt
                if ggt==0:
                    print(point_coords_batch[0], point_labels_batch[0])
                    ggt=1

                masks_list, _, lowres_list = self.sam2_pred.predict_batch_logits_torch(
                    point_coords_batch=point_coords_batch,
                    point_labels_batch=point_labels_batch,
                    box_batch=None,
                    mask_input_batch=None,
                    multimask_output=False,
                    return_logits=True,
                    normalize_coords=True,
                )
                # lowres_list: List[Tensor [1,1,h,w] 或 [1,h,w]]
                # 統一成 [B,1,h,w]
                lowres = []
                for t in lowres_list:
                    if t.dim() == 3:  # [1,h,w]
                        t = t.unsqueeze(0)
                    lowres.append(t)
                pred_mask_logits = torch.cat(lowres, dim=0)  # [B,1,h,w]
                dice = dice_loss_from_logits(pred_mask_logits, tgt_gt_mask_512)
        
        if pred_mask_logits is not None:
            out["pred_mask_logits"] = pred_mask_logits

        out["loss_dice"] = dice

        # === Aux mask dice ===
        out["loss_aux_dice"] = grid_logits.new_tensor(0.0)   # 先初始化，確保不是 None
        if tgt_gt_mask_512 is not None:
            aux_dice = dice_loss_from_logits(
                F.interpolate(aux_mask_logits, size=(256,256), mode="bilinear", align_corners=False),
                F.interpolate(tgt_gt_mask_512.float(), size=(256,256), mode="nearest")
            )
            out["loss_aux_dice"] = aux_dice
        
        # === Total ===
        total = self.lambda_ce * ce + self.lambda_dice * dice + self.lambda_aux * out["loss_aux_dice"]
        out["loss"] = total
        return out