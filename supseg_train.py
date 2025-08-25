import os, time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import math

# import base episodic builders from your project
from dataloader_ps import build_psv2_index as build_coco20i_index
from dataloader_ps import OneShotPSV2Random as OneShotCOCO20iRandom

# from our modules above
from supseg_model import Sam2TorchWrapper,SupSegGridSAM2Spatial#,CrossAttentionFuseWin2
from supseg_dataset import SupEpisodeAdapter
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR


# reuse predictor (non-diff) for speed when sam2_torch is not available
from train import sam2_build_image_predictor, resize_mask, cache_key, _patch_predictor_transforms

def save_checkpoint(path, model, opt, sch, epoch, best_iou):
    state = {
        'model': model.state_dict(),
        'optimizer': opt.state_dict(),
        'scheduler': sch.state_dict(),
        'epoch': int(epoch),
        'best_iou': float(best_iou),
    }
    torch.save(state, path)

def collate_sup(batch):
    out = {}
    keys_tensor = ['sam_t','dn_t','proto_fg_sam','proto_bg_sam','proto_fg_dn','proto_bg_dn','extra_maps']
    for k in keys_tensor:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    out['tgt_rgb']  = [b['tgt_rgb'] for b in batch]
    out['tgt_mask'] = [b['tgt_mask'] for b in batch]
    out['tgt_path'] = [b['tgt_path'] for b in batch]
    return out

t=0
def infer_sam2_masks_cached(points_xy, labels, tgt_rgb_list, tgt_mask_list, tgt_path_list, predictor, cache_dir, cache_long=1024):
    """Use your cached predictor path for non-differentiable mask decoding (fast). Returns list of masks and numpy IoUs."""
    ious = []
    masks_out = []
    B = len(tgt_rgb_list)
    for b in range(B):
        pts = points_xy[b]; lbl = labels[b]
        if pts is None:
            ious.append(0.0); masks_out.append(None); continue
        # map grid-based px (already 512 scale) -> letterbox cache_long
        rgb_lb = cv2.resize(tgt_rgb_list[b], (cache_long, cache_long), interpolation=cv2.INTER_AREA)
        H_img, W_img = rgb_lb.shape[:2]
        xy = pts.astype(np.float32)
        key = cache_key(tgt_path_list[b])
        pt_path = os.path.join(cache_dir, f"{key}.pt")
        if not os.path.isfile(pt_path):
            ious.append(0.0); masks_out.append(None); continue
        data = torch.load(pt_path, map_location="cpu")
        predictor.set_image_from_cache(data["sam2"])  # no encoder
        _patch_predictor_transforms(predictor)
        global t
        if t==0:
            print(xy, lbl)
            t=1
        masks, scores, _ = predictor.predict(point_coords=xy, point_labels=lbl.astype(np.int32), multimask_output=True, normalize_coords=True)
        j = int(np.argmax(scores)); m = masks[j].astype(np.uint8)
        gt_lb = resize_mask(tgt_mask_list[b], cache_long)
        gt = (gt_lb>127).astype(np.uint8)
        inter = (m>0) & (gt>0); union = (m>0) | (gt>0)
        iou = float(inter.sum())/float(union.sum()+1e-6)
        ious.append(iou); masks_out.append(m)
    return ious, masks_out


def parse_args():
    p = argparse.ArgumentParser('Supervised Grid+SAM2 Trainer (no RL)')
    p.add_argument('--manifest-train', type=str, required=True)
    p.add_argument('--manifest-val', type=str, default=None)
    p.add_argument('--fold', type=int, default=0, choices=[0,1,2,3])
    p.add_argument('--episodes', type=int, default=10000)
    p.add_argument('--role', type=str, default='novel', choices=['novel','base'])
    p.add_argument('--cache-dir', type=str, required=True)
    p.add_argument('--cache-long', type=int, default=512)
    p.add_argument('--sam2-cfg', type=str, default='sam2.1_hiera_s.yaml')
    p.add_argument('--sam2-ckpt', type=str, default='checkpoints/sam2.1_hiera_small.pt')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--lr', type=float, default=5e-3)
    p.add_argument('--pos-th', type=float, default=0.75)
    p.add_argument('--neg-th', type=float, default=0.25)
    p.add_argument('--lambda-ce', type=float, default=0.5)
    p.add_argument('--lambda-dice', type=float, default=1.0)
    p.add_argument('--lambda-aux', type=float, default=0.3)
    p.add_argument('--sam2-diff', action='store_true', help='use differentiable SAM2 path (prompt->mask)')
    p.add_argument('--eval-every', type=int, default=1)
    p.add_argument('--val-samples', type=int, default=200)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument('--out-dir', type=str, default='outputs_sup')
    p.add_argument('--k_pos', type=int, default=10)
    p.add_argument('--k_neg', type=int, default=8)
    p.add_argument('--tau-start', type=float, default=1.2)
    p.add_argument('--tau-end',   type=float, default=0.8)
    p.add_argument('--pseudo-temp-start', type=float, default=0.6)
    p.add_argument('--pseudo-temp-end',   type=float, default=0.3)
    p.add_argument('--resume', type=str, default=None)
    return p.parse_args()


def build_loaders(args):
    idx_tr = build_coco20i_index(args.manifest_train, fold=args.fold, role=args.role)
    base_tr = OneShotCOCO20iRandom(index=idx_tr, episodes=args.episodes, seed=2025)
    ds_tr = SupEpisodeAdapter(base_tr, cache_dir=args.cache_dir, cache_long=args.cache_long)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                       pin_memory=True, persistent_workers=(args.workers>0), collate_fn=collate_sup, drop_last=True)
    dl_val = None
    if args.manifest_val:
        idx_va = build_coco20i_index(args.manifest_val, fold=args.fold, role=args.role)
        base_va = OneShotCOCO20iRandom(index=idx_va, episodes=args.val_samples, seed=2025)
        ds_va = SupEpisodeAdapter(base_va, cache_dir=args.cache_dir, cache_long=args.cache_long)
        dl_val = DataLoader(ds_va, batch_size=1, shuffle=False, collate_fn=collate_sup)
    return dl_tr, dl_val


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dl_tr, dl_val = build_loaders(args)
    start_epoch = 0; best_iou = 0.0

    # model
    sam2_torch = None
    if args.sam2_diff:
        try:
            sam2_torch = Sam2TorchWrapper(args.sam2_cfg, args.sam2_ckpt, freeze=True)
            sam2_torch = sam2_torch.to(device)
            print('[sam2] differentiable prompt->mask enabled')
        except Exception as e:
            print(f'[sam2] differentiable path unavailable: {e}. Continue without gradient through SAM2.')
            sam2_torch = None

    
    predictor = sam2_build_image_predictor(args.sam2_cfg, args.sam2_ckpt)


    # model = SupSegGridSAM2(proj_dim=256, pos_th=args.pos_th, neg_th=args.neg_th,
    #                        lambda_ce=args.lambda_ce, lambda_dice=args.lambda_dice,
    #                        sam2_torch=sam2_torch,k_pos=args.k_pos, k_neg=args.k_neg, sam2_pred=predictor).to(device)

    # model = SupSegGridSAM2Spatial(proj_dim=256, pos_th=args.pos_th, neg_th=args.neg_th,
    #                            lambda_ce=args.lambda_ce, lambda_dice=args.lambda_dice,
    #                            sam2_torch=sam2_torch,k_pos=args.k_pos, k_neg=args.k_neg, sam2_pred=predictor,
    #                            use_coord=True, e_channels=0,  pe_freqs=16).to(device)

    model = SupSegGridSAM2Spatial(
        proj_dim=256,
        pos_th=args.pos_th, neg_th=args.neg_th,
        lambda_ce=args.lambda_ce, lambda_dice=args.lambda_dice,
        sam2_torch=sam2_torch, k_pos=args.k_pos, k_neg=args.k_neg, sam2_pred=predictor,
        use_coord=True, e_channels=0, pe_freqs=16,lambda_aux=args.lambda_aux
    ).to(device)

    model.tau = float(args.tau_start)
    with torch.no_grad():
        model.pseudo_temp.copy_(torch.tensor(args.pseudo_temp_start, device=device))
    
    # # 直接把單層 fuse 換成「雙層 Window Fuse」
    # model.fuse = CrossAttentionFuseWin2(
    #     dim=256,           # 要和 proj_dim 一致
    #     heads=8,           # 和原本一致
    #     window_size=8,     # 32×32 特徵 → 8 最剛好（切 4×4 個窗）
    #     use_abspe=False,   # 先關；若想再加絕對PE再開 True
    #     pe_freqs=16
    # ).to(device)
    

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * max(1, len(dl_tr))
    warmup_steps = int(0.1 * total_steps)
    sch = SequentialLR(
        opt,
        schedulers=[
            LinearLR(opt, start_factor=1e-6/args.lr, end_factor=1.0, total_iters=warmup_steps),
            CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=1e-6)
        ],
        milestones=[warmup_steps]
    )

    if args.resume and os.path.isfile(args.resume):
            ckpt = torch.load(args.resume, map_location=device)
            model.load_state_dict(ckpt['model'], strict=False)
            if 'optimizer' in ckpt: opt.load_state_dict(ckpt['optimizer'])
            if 'scheduler' in ckpt: sch.load_state_dict(ckpt['scheduler'])
            best_iou = float(ckpt.get('best_iou', 0.0))
            start_epoch = int(ckpt.get('epoch', -1)) + 1
            print(f"[resume] from {args.resume} → epoch {start_epoch},  best_iou {best_iou:.4f}")

    for epoch in range(start_epoch, args.epochs):
        model.train(); t0=time.time(); losses=[];dice_loss=[];ce_loss=[];aux_loss=[]

        # ---- 退火排程（放在每個 epoch 的一開始；確保跑完最後一個 epoch 時到達 end 值）----
        T = max(1, args.epochs - start_epoch)
        # 用 (epoch + 1) 讓最後一輪到 1.0
        prog = min(1.0, (epoch + 1 - start_epoch) / T)
        
        # τ 線性：tau_start -> tau_end
        model.tau = args.tau_start + (args.tau_end - args.tau_start) * prog
        
        # pseudo_temp 餘弦：start -> end
        # 建議：start=0.6, end=0.35（先穩，再慢慢銳化）
        cos_w = 0.5 * (1.0 + math.cos(math.pi * (1.0 - prog)))  # 0->1 時，cos_w: 0->1（方便閱讀）
        pseudo_t = args.pseudo_temp_start * (1 - cos_w) + args.pseudo_temp_end * cos_w
        
        with torch.no_grad():
            # 若 pseudo_temp 是 buffer（推薦做法），這行照樣可用
            model.pseudo_temp.copy_(torch.tensor(float(pseudo_t), device=device))
        
        # 輕量 head/主 Dice 的暖啟排程
        # 前 3 個 epoch 降低 Dice / aux 的權重，減少和 CE 的衝突
        if epoch < start_epoch + 10:
            model.lambda_dice = getattr(args, "lambda_dice_warm", 0.2)
            model.lambda_aux = getattr(args, "aux_mask_weight_warm", 0.2)
        else:
            model.lambda_dice = args.lambda_dice
            model.lambda_aux = args.lambda_aux
        

        for batch in tqdm(dl_tr, desc=f'Epoch {epoch}', unit='batch'):
            # === 批量把這個 batch 的 SAM2 特徵灌進 predictor ===
            cache_pack_list = []
            for pth in batch['tgt_path']:
                key = cache_key(pth)
                pt_path = os.path.join(args.cache_dir, f"{key}.pt")
                data = torch.load(pt_path, map_location='cpu')
                cache_pack_list.append(data['sam2'])
            predictor.set_image_batch_from_cache(cache_pack_list)
            
            sam_t = batch['sam_t'].to(device).squeeze(1)      # [B,256,64,64]
            dn_t  = batch['dn_t'].to(device)                  # [B,768,32,32]
            if dn_t.dim() == 5 and dn_t.size(1) == 1:         # 有時候會變 [B,1,768,32,32]
                                    dn_t = dn_t.squeeze(1)   
            pfgs  = batch['proto_fg_sam'].to(device).squeeze(1)
            pbgs  = batch['proto_bg_sam'].to(device).squeeze(1)
            pfgd  = batch['proto_fg_dn'].to(device).squeeze(1)
            pbgd  = batch['proto_bg_dn'].to(device).squeeze(1)
            # extra = batch['extra_maps'].to(device).squeeze(1)
            
            # Build GT grid labels from target GT masks (512)
            gt_mask_list = batch['tgt_mask']
            gt_mask_512 = torch.stack([torch.from_numpy((cv2.resize(m, (512,512), interpolation=cv2.INTER_NEAREST)>127).astype(np.float32)) for m in gt_mask_list], dim=0).unsqueeze(1).to(device)

            # Optional image embeddings for differentiable SAM2 path (we use cached low-res emb directly)
            img_emb = None
            if model.sam2_torch is not None:
                # load from cache dicts and stack
                emb_list = []
                for p in batch['tgt_path']:
                    key = cache_key(p); pt = os.path.join(args.cache_dir, f"{key}.pt")
                    data = torch.load(pt, map_location='cpu')
                    from train import _sam2_embed_from_cached
                    e = _sam2_embed_from_cached(data['sam2']).unsqueeze(0)  # [1,C,Hf,Wf]
                    emb_list.append(e)
                img_emb = torch.cat(emb_list, dim=0).to(device)

            out = model(sam_t, dn_t, pfgs, pbgs, pfgd, pbgd,
                        tgt_gt_mask_512=gt_mask_512,
                        extra_maps32=None,#F.interpolate(extra, size=(32,32), mode='bilinear', align_corners=False),
                        image_embeddings_for_sam2=img_emb)
                        

            loss = out['loss']
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sch.step()

            # for name, p in model.named_parameters():
            #     if p.grad is not None and p.requires_grad:
            #         print(name, p.grad.abs().mean().item())

            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            losses.append(float(loss.item()))
            dice_loss.append(float(out['loss_dice'].item()))
            ce_loss.append(float(out['loss_ce'].item()))
            aux_loss.append(float(out['loss_aux_dice'].item()))
          
        print(f"[epoch {epoch}] train loss {np.mean(losses):.4f}  dice {np.mean(dice_loss):.4f}  ce {np.mean(ce_loss):.4f} aux {np.mean(aux_loss):.4f} time {time.time()-t0:.1f}s")
        if (epoch+1) % args.save_every == 0:
            os.makedirs(args.out_dir, exist_ok=True)
            save_checkpoint(os.path.join(args.out_dir, f"ppnet_epoch{epoch}.pt"),
                            model, opt, sch, epoch, best_iou)
            print('Save checkpt.')

        # ---------- simple validation (IoU via cached predictor) ----------
        if dl_val is not None and ((epoch+1) % args.eval_every == 0):
            global t
            t=0
            model.eval()
            ious = []
            with torch.no_grad():
                for batch in dl_val:
                    sam_t = batch['sam_t'].to(device).squeeze(1)
                    dn_t  = batch['dn_t'].to(device)
                    if dn_t.dim() == 5 and dn_t.size(1) == 1:         # 有時候會變 [B,1,768,32,32]
                        dn_t = dn_t.squeeze(1)                        # → [B,768,32,32]
                    pfgs  = batch['proto_fg_sam'].to(device).squeeze(1)
                    pbgs  = batch['proto_bg_sam'].to(device).squeeze(1)
                    pfgd  = batch['proto_fg_dn'].to(device).squeeze(1)
                    pbgd  = batch['proto_bg_dn'].to(device).squeeze(1)
                    extra = batch['extra_maps'].to(device).squeeze(1)
                    out = model(sam_t, dn_t, pfgs, pbgs, pfgd, pbgd,
                                tgt_gt_mask_512=None,
                                extra_maps32=None,#F.interpolate(extra, size=(32,32), mode='bilinear', align_corners=False),
                                image_embeddings_for_sam2=None)
                    # get points and decode with fast predictor
                    pts_list, lbl_list = model.points_from_grid(out['grid_logits'])
                    pts_np_list = [p.detach().cpu().numpy().astype(np.float32) for p in pts_list]
                    lbl_np_list = [l.detach().cpu().numpy().astype(np.int32)  for l in lbl_list]

                    iou_b, _ = infer_sam2_masks_cached(pts_np_list, lbl_np_list,
                                                       batch['tgt_rgb'], batch['tgt_mask'],
                                                       batch['tgt_path'], predictor,
                                                       args.cache_dir, args.cache_long)
                    ious.extend(iou_b)
            
            print(f"\n[Val] mIoU={float(np.mean(ious)):.4f} over {len(ious)} images\n")
            if float(np.mean(ious)) > best_iou:
                os.makedirs(args.out_dir, exist_ok=True)
                best_iou = float(np.mean(ious))
                save_checkpoint(os.path.join(args.out_dir, 'best_supseg.pth'), model, opt, sch, epoch, best_iou)
                print(f"\033[94mBest IoU : {best_iou}\033[0m\n")

if __name__ == '__main__':
    main()
