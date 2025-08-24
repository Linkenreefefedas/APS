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
    y[ratio > pos_th] = 0
    y[ratio < neg_th] = 1
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

# ---- Add these helpers above CrossAttentionFuse (e.g., near other blocks) ----
class SinCosPosEnc2D(nn.Module):
    """
    Fixed 2D sine-cosine absolute positional encoding:
      returns [1, Cpe, H, W] with Cpe = 2 * num_freqs * 2 (x/y with sin+cos).
    Then a 1x1 conv will project to model dim for residual add.
    """
    def __init__(self, num_freqs: int = 16):
        super().__init__()
        self.num_freqs = num_freqs

    @torch.no_grad()
    def forward(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        # coords in [-1,1]
        y = torch.linspace(-1.0, 1.0, H, device=device)
        x = torch.linspace(-1.0, 1.0, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")  # [H,W]
        freqs = 2.0 ** torch.arange(self.num_freqs, device=device).float()  # [F]
        # [H,W,F]
        xx = xx.unsqueeze(-1) * freqs
        yy = yy.unsqueeze(-1) * freqs
        # sin-cos for x and y -> [H,W, 4F]
        pe = torch.cat([xx.sin(), xx.cos(), yy.sin(), yy.cos()], dim=-1)
        pe = pe.permute(2, 0, 1).unsqueeze(0)  # [1,4F,H,W]
        return pe.contiguous()


class ConvPosBias(nn.Module):
    """
    Depthwise Conv positional bias: x + DWConv(x) (3x3)。保留局部空間偏差。
    """
    def __init__(self, dim: int, k: int = 3):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=k, padding=k//2, groups=dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dw(x)


# ---- Replace original CrossAttentionFuse with this version ----
class CrossAttentionFuse(nn.Module):
    def __init__(self, dim: int, heads: int = 8, pe_freqs: int = 16):
        super().__init__()
        # q/k/v 前加上絕對位置編碼 (用 1x1 投影到 dim 後 residual 相加)
        self.abs_pe_gen = SinCosPosEnc2D(num_freqs=pe_freqs)
        self.abs_pe_proj = nn.Conv2d(4 * pe_freqs, dim, kernel_size=1, bias=False)

        # 局部位置偏置：避免 attention 只做全域平均
        self.local_bias_q = ConvPosBias(dim)
        self.local_bias_kv = ConvPosBias(dim)

        # q/k/v 與 MHA
        self.q = nn.Conv2d(dim, dim, 1, bias=False)
        self.k = nn.Conv2d(dim, dim, 1, bias=False)
        self.v = nn.Conv2d(dim, dim, 1, bias=False)
        self.mha = nn.MultiheadAttention(dim, heads, batch_first=True)

        # 前饋 + 殘差
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1)
        )
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(self, ref: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        ref: [B,C,H,W]  — 你目前把原型 token 平均後鋪成 32x32 進來
        tgt: [B,C,H,W]  — 目標特徵 (32x32)
        這裡會對兩者都加同一組 2D 絕對位置編碼（在各自座標網格），
        並經 depthwise conv 形成局部偏置，再進入 q/k/v + MHA。
        """
        B, C, H, W = tgt.shape

        # 2D 絕對位置編碼（固定 sin-cos），投影到 dim 後 residual 相加
        pe = self.abs_pe_gen(H, W, tgt.device)                # [1,4F,H,W]
        pe_proj = self.abs_pe_proj(pe)                         # [1,dim,H,W]
        tgt_pe = tgt + pe_proj                                 # 保留空間座標感
        ref_pe = ref + pe_proj                                 # ref 也加同樣網格座標

        # 局部位置偏置（depthwise conv）
        tgt_pe = self.local_bias_q(tgt_pe)
        ref_pe = self.local_bias_kv(ref_pe)

        # q/k/v
        q = self.q(tgt_pe).flatten(2).transpose(1, 2)  # [B,HW,C]
        k = self.k(ref_pe).flatten(2).transpose(1, 2)
        v = self.v(ref_pe).flatten(2).transpose(1, 2)

        # MHA（保留座標的同時做 token 對齊）
        out, _ = self.mha(self.norm_q(q), self.norm_kv(k), self.norm_kv(v))
        out = out.transpose(1, 2).view(B, C, H, W)

        # 殘差 + FFN
        out = out + tgt
        return out + self.ffn(out)

class GridHead(nn.Module):
    def __init__(self, in_ch: int, num_classes: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch//2, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.cls = nn.Conv2d(in_ch//2, num_classes, 1)
    def forward(self, x32: torch.Tensor) -> torch.Tensor:
        x = F.avg_pool2d(x32, kernel_size=2, stride=2)   # 32 -> 16
        x = self.block(x)
        return self.cls(x)                                # [B,3,16,16]


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

    def _make_pseudo(self, sam32: torch.Tensor, proto_fg: torch.Tensor, proto_bg: Optional[torch.Tensor] = None) -> torch.Tensor:
        sim_fg = cosine_sim_map(sam32, proto_fg)  # [-1,1]
        if proto_bg is not None:
            sim_bg = cosine_sim_map(sam32, proto_bg)
            s = torch.sigmoid((sim_fg - sim_bg)/0.2)
        else:
            s = (sim_fg + 1.0)/2.0
        return s  # [B,1,32,32] in [0,1]

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
        grid_logits: [B,3,16,16]
        回傳:
          points_xy: [B,K,2]  (K = self.k_pos + self.k_neg)，影像像素座標(對 512x512)
          labels:    [B,K]    (1=pos, 0=neg)
        """
        import torch
        import torch.nn.functional as F
    
        B, C, H, W = grid_logits.shape  # H=W=16
        P = F.softmax(grid_logits, dim=1)
        pos_map = P[:, 0:1]  # [B,1,H,W]
        neg_map = P[:, 1:2]
    
        # 可調參數（不在屬性時給預設）
        cand_factor = getattr(self, "cand_factor", 5)      # 候選放大倍數
        min_dist    = float(getattr(self, "nms_min_dist", 2.0))  # 格為單位的最小距離
        th_pos      = getattr(self, "peak_th_pos", None)   # 0~1；None=不設門檻
        th_neg      = getattr(self, "peak_th_neg", None)
        r           = getattr(self, "refine_r", 1)         # 精修半徑: 1→3x3, 2→5x5
        tau         = float(getattr(self, "tau", 1.0))
    
        def _topk_candidates(heat: torch.Tensor, k: int, th: float | None):
            # 先做局部極大值（3x3）再取較多候選
            pool = F.max_pool2d(heat, 3, 1, 1)
            keep = (heat == pool) * heat
            if th is not None:
                keep = torch.where(keep > th, keep, keep.new_zeros(keep.shape))
            scores = keep.view(B, -1)                                         # [B,H*W]
            topk = min(max(k * cand_factor, k), scores.size(1))               # 至少 k
            topv, topi = torch.topk(scores, k=topk, dim=1)
            ys, xs = (topi // W).float(), (topi % W).float()
            coords = torch.stack([xs, ys], dim=-1)                            # [B,topk,2]
            return coords, topv                                               # 均為 tensor
    
        def _greedy_nms_single(coords_b: torch.Tensor, scores_b: torch.Tensor,
                               k_keep: int, min_d: float):
            # coords_b: [M,2], scores_b: [M]
            if k_keep <= 0 or coords_b.numel() == 0:
                return coords_b.new_zeros((0, 2)), scores_b.new_zeros((0,))
            order = scores_b.argsort(descending=True)
            coords_b = coords_b[order]
            scores_b = scores_b[order]
            keep_idx = []
            for i in range(coords_b.size(0)):
                if len(keep_idx) == 0:
                    keep_idx.append(i)
                else:
                    # 與已保留點的最小距離（在 grid 座標上）
                    d = torch.cdist(coords_b[i:i+1], coords_b[keep_idx]).squeeze(0)
                    if d.min() >= min_d:
                        keep_idx.append(i)
                if len(keep_idx) >= k_keep:
                    break
            keep_idx = torch.tensor(keep_idx, device=coords_b.device, dtype=torch.long)
            kept_c = coords_b[keep_idx]
            kept_s = scores_b[keep_idx]
            # 若不夠 k_keep，從剩餘（被距離抑制掉的）裡，依分數補齊至定長
            if kept_c.size(0) < k_keep and coords_b.size(0) > kept_c.size(0):
                mask = torch.ones(coords_b.size(0), device=coords_b.device, dtype=torch.bool)
                mask[keep_idx] = False
                rest_c = coords_b[mask]
                rest_s = scores_b[mask]
                need = k_keep - kept_c.size(0)
                add_n = min(need, rest_c.size(0))
                if add_n > 0:
                    kept_c = torch.cat([kept_c, rest_c[:add_n]], dim=0)
                    kept_s = torch.cat([kept_s, rest_s[:add_n]], dim=0)
            # 若候選本來就少於 k_keep，這裡可能仍小於 k_keep；下面會再補保底正點
            return kept_c, kept_s
    
        def _refine_local_single(heat_b: torch.Tensor, xy_g_b: torch.Tensor):
            # heat_b: [1,1,H,W]；xy_g_b: [k,2]（grid coords, float）
            if xy_g_b.numel() == 0:
                return xy_g_b
            k = xy_g_b.size(0)
            cx = xy_g_b[:, 0].round().clamp_(0, W - 1).long()
            cy = xy_g_b[:, 1].round().clamp_(0, H - 1).long()
            out = []
            for i in range(k):
                x0 = max(0, cx[i].item() - r); x1 = min(W, cx[i].item() + r + 1)
                y0 = max(0, cy[i].item() - r); y1 = min(H, cy[i].item() + r + 1)
                patch = heat_b[:, :, y0:y1, x0:x1]  # [1,1,hh,ww]
                hh, ww = patch.shape[-2:]
                logits = torch.log(patch.clamp_min(1e-8)).view(1, -1)
                sm = F.softmax(logits / max(1e-6, tau), dim=1).view(1, 1, hh, ww)
                xs = torch.linspace(0, ww - 1, ww, device=heat_b.device).view(1, 1, 1, ww)
                ys = torch.linspace(0, hh - 1, hh, device=heat_b.device).view(1, 1, hh, 1)
                dx = (sm * xs).sum(dim=(2, 3))
                dy = (sm * ys).sum(dim=(2, 3))
                gx = x0 + dx.squeeze()
                gy = y0 + dy.squeeze()
                out.append(torch.stack([gx, gy], dim=-1))
            return torch.stack(out, dim=0)  # [k,2]
    
        # 1) 候選（帶 3×3 局部極大值 NMS + 門檻）
        pos_cand, pos_v = _topk_candidates(pos_map, self.k_pos, th_pos)
        neg_cand, neg_v = _topk_candidates(neg_map, self.k_neg, th_neg)
    
        # 2) 類內貪婪 NMS +（必要時）補齊到定長
        pos_list, neg_list = [], []
        for b in range(B):
            pc, _ = _greedy_nms_single(pos_cand[b], pos_v[b], self.k_pos, min_dist)
            nc, _ = _greedy_nms_single(neg_cand[b], neg_v[b], self.k_neg, min_dist)
            pos_list.append(pc)
            neg_list.append(nc)
    
        # 3) 正點保底：若 k_pos>0 但該圖沒有正點，補 global argmax
        if self.k_pos > 0:
            flat = pos_map.view(B, -1)
            topi = flat.argmax(dim=1)
            ys0 = (topi // W).float(); xs0 = (topi % W).float()
            gmax = torch.stack([xs0, ys0], dim=-1)  # [B,2]
            for b in range(B):
                if pos_list[b].size(0) == 0:
                    pos_list[b] = gmax[b:b+1]  # [1,2]
                # 若仍不足 k_pos（候選太少），重複最後一個保底填滿到定長
                if pos_list[b].size(0) < self.k_pos:
                    need = self.k_pos - pos_list[b].size(0)
                    pos_list[b] = torch.cat([pos_list[b], pos_list[b][-1:].repeat(need, 1)], dim=0)
    
        # 4) 若負點不足，同樣重複最後一個填滿到定長（維持輸出形狀）
        for b in range(B):
            if neg_list[b].size(0) < self.k_neg and self.k_neg > 0:
                if neg_list[b].size(0) == 0:
                    # 用 neg_map 的 global argmax 當起點
                    flatn = neg_map.view(B, -1)
                    topin = flatn.argmax(dim=1)
                    ys1 = (topin // W).float(); xs1 = (topin % W).float()
                    gmaxn = torch.stack([xs1, ys1], dim=-1)  # [B,2]
                    neg_list[b] = gmaxn[b:b+1]
                need = self.k_neg - neg_list[b].size(0)
                neg_list[b] = torch.cat([neg_list[b], neg_list[b][-1:].repeat(need, 1)], dim=0)
    
        # 5) 局部精修（建議：先 NMS 再 refine；這裡逐圖處理）
        pos_refined, neg_refined = [], []
        for b in range(B):
            pos_refined.append(_refine_local_single(pos_map[b:b+1], pos_list[b]))  # [kp,2]
            neg_refined.append(_refine_local_single(neg_map[b:b+1], neg_list[b]))  # [kn,2]
    
        # 6) 拼接 + 映射到像素（512）；輸出固定長度 K = k_pos + k_neg
        kp, kn = self.k_pos, self.k_neg
        pts_g = torch.stack([torch.cat([pos_refined[b], neg_refined[b]], dim=0)
                             for b in range(B)], dim=0)                           # [B,K,2]
        scale_x = 512.0 / W
        scale_y = 512.0 / H
        pts_px = torch.stack([(pts_g[..., 0] + 0.5) * scale_x,
                              (pts_g[..., 1] + 0.5) * scale_y], dim=-1)           # [B,K,2]
    
        labels = torch.zeros(B, kp + kn, device=grid_logits.device, dtype=torch.long)
        if kp > 0:
            labels[:, :kp] = 1  # 前 kp 個為正，其餘為負
    
        return pts_px, labels
    
    
    def focal_loss_multiclass_ignore_neutral(
        self,
        logits: torch.Tensor,         # [B, 3, H, W]，你的 grid_logits
        targets: torch.Tensor,        # [B, H, W]，0=pos, 1=neg, 2=neutral
        gamma: float = 2.0,           # 焦點參數，越大越聚焦困難樣本
        alpha_pos: float = 2.0,       # 正類權重（> neg）
        alpha_neg: float = 1.0,       # 負類權重
        ignore_index: int = 2,        # 忽略 neutral
        normalize_by_weight: bool = True,  # 用加權樣本數作分母，穩定尺度
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
        alpha = logits.new_tensor([alpha_pos, alpha_neg, 0.0])  # [3] same device/dtype
        alpha_t = alpha[y]                                      # [B,H,W]  ← 這行取代 gather
    
        # --- Focal ---
        loss_pix = -(alpha_t * (1.0 - pt).pow(gamma) * logp_t)  # [B,H,W]
    
        # --- 忽略 neutral 並平均 ---
        valid = (y != ignore_index).float()
        loss_pix = loss_pix * valid
        if normalize_by_weight:
            denom = (alpha_t * valid).sum().clamp_min(1.0)      # 加權有效樣本數
        else:
            denom = valid.sum().clamp_min(1.0)
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
        grid_logits = self.forward_grid(
            sam_tgt_64, dino_tgt_32, proto_fg_sam, proto_bg_sam, proto_fg_dn, proto_bg_dn, extra_maps32
        )
        out = {"grid_logits": grid_logits}
        if tgt_gt_mask_512 is not None:
            y = make_grid_labels(tgt_gt_mask_512, self.pos_th, self.neg_th)
            #ce = F.cross_entropy(grid_logits, y)
            ce = self.focal_loss_multiclass_ignore_neutral(
                grid_logits, y,
                gamma=2.0,      # 常用 1.5~2.5
                alpha_pos=2.0,  # 正類較重
                alpha_neg=1.0,  # 負類較輕
                ignore_index=2,
            )
            
            out["loss_ce"] = ce
        else:
            ce = grid_logits.new_tensor(0.0)
            out["loss_ce"] = ce

        dice = grid_logits.new_tensor(0.0)
        pred_mask_logits = None
        B = grid_logits.size(0)
        if (tgt_gt_mask_512 is not None):
            # 取得點
            pts, lbl = self.points_from_grid(grid_logits)  # [B,K,2], [B,K]
        
            if self.sam2_pred is not None:
                # sam2_image_predictor 已經在外部 set_image_batch_from_cache(...) 好這一個 batch
                # 準備 list[tensor]，每張一個 (保持 torch，不 numpy)
                point_coords_batch = [pts[b] for b in range(B)]
                point_labels_batch = [lbl[b] for b in range(B)]
        
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
        
            elif self.sam2_torch is not None and (image_embeddings_for_sam2 is not None):
                # 備援：可微 wrapper（逐張或整批）—如果你想保留
                pred_mask_logits = self.sam2_torch.forward_with_embeddings(
                    image_embeddings=image_embeddings_for_sam2,
                    point_coords=pts,
                    point_labels=lbl,
                    orig_hw=(512, 512)
                )
                dice = dice_loss_from_logits(pred_mask_logits, tgt_gt_mask_512)
        
        if pred_mask_logits is not None:
            out["pred_mask_logits"] = pred_mask_logits
        out["loss_dice"] = dice

        total = self.lambda_ce*ce + self.lambda_dice*dice
        out["loss"] = total
        return out