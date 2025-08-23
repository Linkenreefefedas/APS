# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL.Image import Image

from sam2.modeling.sam2_base import SAM2Base

from sam2.utils.transforms import SAM2Transforms


class SAM2ImagePredictor:
    def __init__(
        self,
        sam_model: SAM2Base,
        mask_threshold=0.0,
        max_hole_area=0.0,
        max_sprinkle_area=0.0,
        **kwargs,
    ) -> None:
        """
        Uses SAM-2 to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam-2): The model to use for mask prediction.
          mask_threshold (float): The threshold to use when converting mask logits
            to binary masks. Masks are thresholded at 0 by default.
          max_hole_area (int): If max_hole_area > 0, we fill small holes in up to
            the maximum area of max_hole_area in low_res_masks.
          max_sprinkle_area (int): If max_sprinkle_area > 0, we remove small sprinkles up to
            the maximum area of max_sprinkle_area in low_res_masks.
        """
        super().__init__()
        self.model = sam_model
        self._transforms = SAM2Transforms(
            resolution=self.model.image_size,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )

        # Predictor state
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        # Whether the predictor is set for single image or a batch of images
        self._is_batch = False

        # Predictor config
        self.mask_threshold = mask_threshold

        # Spatial dim for backbone feature maps
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAM2ImagePredictor":
        """
        Load a pretrained model from the Hugging Face hub.

        Arguments:
          model_id (str): The Hugging Face repository ID.
          **kwargs: Additional arguments to pass to the model constructor.

        Returns:
          (SAM2ImagePredictor): The loaded model.
        """
        from sam2.build_sam import build_sam2_hf

        sam_model = build_sam2_hf(model_id, **kwargs)
        return cls(sam_model, **kwargs)

    @torch.no_grad()
    def set_image(
        self,
        image: Union[np.ndarray, Image],
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray or PIL Image): The input image to embed in RGB format. The image should be in HWC format if np.ndarray, or WHC format if PIL Image
          with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        self.reset_predictor()
        # Transform the image to the form expected by the model
        if isinstance(image, np.ndarray):
            logging.info("For numpy array image, we assume (HxWxC) format")
            self._orig_hw = [image.shape[:2]]
        elif isinstance(image, Image):
            w, h = image.size
            self._orig_hw = [(h, w)]
        else:
            raise NotImplementedError("Image format not supported")

        input_image = self._transforms(image)
        input_image = input_image[None, ...].to(self.device)

        assert (
            len(input_image.shape) == 4 and input_image.shape[1] == 3
        ), f"input_image must be of size 1x3xHxW, got {input_image.shape}"
        logging.info("Computing image embeddings for the provided image...")
        backbone_out = self.model.forward_image(input_image)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self._is_image_set = True
        logging.info("Image embeddings computed.")

    @torch.no_grad()
    def set_image_batch(
        self,
        image_list: List[Union[np.ndarray]],
    ) -> None:
        """
        Calculates the image embeddings for the provided image batch, allowing
        masks to be predicted with the 'predict_batch' method.

        Arguments:
          image_list (List[np.ndarray]): The input images to embed in RGB format. The image should be in HWC format if np.ndarray
          with pixel values in [0, 255].
        """
        self.reset_predictor()
        assert isinstance(image_list, list)
        self._orig_hw = []
        for image in image_list:
            assert isinstance(
                image, np.ndarray
            ), "Images are expected to be an np.ndarray in RGB format, and of shape  HWC"
            self._orig_hw.append(image.shape[:2])
        # Transform the image to the form expected by the model
        img_batch = self._transforms.forward_batch(image_list)
        img_batch = img_batch.to(self.device)
        batch_size = img_batch.shape[0]
        assert (
            len(img_batch.shape) == 4 and img_batch.shape[1] == 3
        ), f"img_batch must be of size Bx3xHxW, got {img_batch.shape}"
        logging.info("Computing image embeddings for the provided images...")
        backbone_out = self.model.forward_image(img_batch)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self._is_image_set = True
        self._is_batch = True
        logging.info("Image embeddings computed.")

    def predict_batch_logits_torch(
            self,
            point_coords_batch: List[np.ndarray] = None,
            point_labels_batch: List[np.ndarray] = None,
            box_batch: List[np.ndarray] = None,
            mask_input_batch: List[np.ndarray] = None,
            multimask_output: bool = True,
            return_logits: bool = False,
            normalize_coords=True,
        ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
            """This function is very similar to predict(...), however it is used for batched mode, when the model is expected to generate predictions on multiple images.
            It returns a tuple of lists of masks, ious, and low_res_masks_logits.
            """
            assert self._is_batch, "This function should only be used when in batched mode"
            if not self._is_image_set:
                raise RuntimeError(
                    "An image must be set with .set_image_batch(...) before mask prediction."
                )
            num_images = len(self._features["image_embed"])
            all_masks = []
            all_ious = []
            all_low_res_masks = []
            for img_idx in range(num_images):
                # Transform input prompts
                point_coords = (
                    point_coords_batch[img_idx] if point_coords_batch is not None else None
                )
                point_labels = (
                    point_labels_batch[img_idx] if point_labels_batch is not None else None
                )
                box = box_batch[img_idx] if box_batch is not None else None
                mask_input = (
                    mask_input_batch[img_idx] if mask_input_batch is not None else None
                )
                mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
                    point_coords,
                    point_labels,
                    box,
                    mask_input,
                    normalize_coords,
                    img_idx=img_idx,
                )
                masks, iou_predictions, low_res_masks = self._predict(
                    unnorm_coords,
                    labels,
                    unnorm_box,
                    mask_input,
                    multimask_output,
                    return_logits=return_logits,
                    img_idx=img_idx,
                )
                masks_np = masks.squeeze(0).float()
                iou_predictions_np = (
                    iou_predictions.squeeze(0).float()
                )
                low_res_masks_np = low_res_masks.squeeze(0).float()
                
                all_masks.append(masks_np)
                all_ious.append(iou_predictions_np)
                all_low_res_masks.append(low_res_masks_np)
    
            return all_masks, all_ious, all_low_res_masks

    def predict_batch(
        self,
        point_coords_batch: List[np.ndarray] = None,
        point_labels_batch: List[np.ndarray] = None,
        box_batch: List[np.ndarray] = None,
        mask_input_batch: List[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords=True,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """This function is very similar to predict(...), however it is used for batched mode, when the model is expected to generate predictions on multiple images.
        It returns a tuple of lists of masks, ious, and low_res_masks_logits.
        """
        assert self._is_batch, "This function should only be used when in batched mode"
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image_batch(...) before mask prediction."
            )
        num_images = len(self._features["image_embed"])
        all_masks = []
        all_ious = []
        all_low_res_masks = []
        for img_idx in range(num_images):
            # Transform input prompts
            point_coords = (
                point_coords_batch[img_idx] if point_coords_batch is not None else None
            )
            point_labels = (
                point_labels_batch[img_idx] if point_labels_batch is not None else None
            )
            box = box_batch[img_idx] if box_batch is not None else None
            mask_input = (
                mask_input_batch[img_idx] if mask_input_batch is not None else None
            )
            mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
                point_coords,
                point_labels,
                box,
                mask_input,
                normalize_coords,
                img_idx=img_idx,
            )
            masks, iou_predictions, low_res_masks = self._predict(
                unnorm_coords,
                labels,
                unnorm_box,
                mask_input,
                multimask_output,
                return_logits=return_logits,
                img_idx=img_idx,
            )
            masks_np = masks.squeeze(0).float().detach().cpu().numpy()
            iou_predictions_np = (
                iou_predictions.squeeze(0).float().detach().cpu().numpy()
            )
            low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
            all_masks.append(masks_np)
            all_ious.append(iou_predictions_np)
            all_low_res_masks.append(low_res_masks_np)

        return all_masks, all_ious, all_low_res_masks
    
    def predict_logits_torch(
        self,
        point_coords: Optional[torch.Tensor] = None,   # 允許 torch tensor
        point_labels: Optional[torch.Tensor] = None,
        box: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = False,
        normalize_coords: bool = True,
        img_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        與 predict 相同，但回傳 torch.Tensor（不 detach、不轉 numpy）。
        回傳：masks, ious, low_res_masks（皆為 torch，B=1）
        """
        # 轉/整理座標到模型內部座標系（沿用現有 _prep_prompts）
        mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
            point_coords, point_labels, box, mask_input, normalize_coords, img_idx=img_idx
        )
        masks, iou_predictions, low_res_masks = self._predict(
            unnorm_coords,
            labels,
            unnorm_box,
            mask_input,
            multimask_output,
            return_logits=True,   # 要 logits
            img_idx=img_idx,
        )
        return masks, iou_predictions, low_res_masks  # 注意：保持 tensor，別 .detach() / .numpy()

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords=True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks for the given input prompts, using the currently set image.

        Arguments:
          point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (np.ndarray or None): A length 4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where
            for SAM, H=W=256.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.
          normalize_coords (bool): If true, the point coordinates will be normalized to the range [0,1] and point_coords is expected to be wrt. image dimensions.

        Returns:
          (np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          (np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )

        # Transform input prompts

        mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
            point_coords, point_labels, box, mask_input, normalize_coords
        )

        masks, iou_predictions, low_res_masks = self._predict(
            unnorm_coords,
            labels,
            unnorm_box,
            mask_input,
            multimask_output,
            return_logits=return_logits,
        )

        masks_np = masks.squeeze(0).float().detach().cpu().numpy()
        iou_predictions_np = iou_predictions.squeeze(0).float().detach().cpu().numpy()
        low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
        return masks_np, iou_predictions_np, low_res_masks_np

    def _prep_prompts(
        self, point_coords, point_labels, box, mask_logits, normalize_coords, img_idx=-1
    ):

        unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = torch.as_tensor(
                point_coords, dtype=torch.float, device=self.device
            )
            unnorm_coords = self._transforms.transform_coords(
                point_coords, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )
            labels = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            if len(unnorm_coords.shape) == 2:
                unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]
        if box is not None:
            box = torch.as_tensor(box, dtype=torch.float, device=self.device)
            unnorm_box = self._transforms.transform_boxes(
                box, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )  # Bx2x2
        if mask_logits is not None:
            mask_input = torch.as_tensor(
                mask_logits, dtype=torch.float, device=self.device
            )
            if len(mask_input.shape) == 3:
                mask_input = mask_input[None, :, :, :]
        return mask_input, unnorm_coords, labels, unnorm_box

    #@torch.no_grad()
    def _predict(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        img_idx: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using SAM2Transforms.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          boxes (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )

        if point_coords is not None:
            concat_points = (point_coords, point_labels)
        else:
            concat_points = None

        # Embed prompts
        if boxes is not None:
            box_coords = boxes.reshape(-1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
            box_labels = box_labels.repeat(boxes.size(0), 1)
            # we merge "boxes" and "points" into a single "concat_points" input (where
            # boxes are added at the beginning) to sam_prompt_encoder
            if concat_points is not None:
                concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
                concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                concat_points = (concat_coords, concat_labels)
            else:
                concat_points = (box_coords, box_labels)

        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=mask_input,
        )

        # Predict masks
        batched_mode = (
            concat_points is not None and concat_points[0].shape[0] > 1
        )  # multi object prediction
        high_res_features = [
            feat_level[img_idx].unsqueeze(0)
            for feat_level in self._features["high_res_feats"]
        ]
        low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
            image_embeddings=self._features["image_embed"][img_idx].unsqueeze(0),
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )

        # Upscale the masks to the original image resolution
        masks = self._transforms.postprocess_masks(
            low_res_masks, self._orig_hw[img_idx]
        )
        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
        if not return_logits:
            masks = masks > self.mask_threshold

        return masks, iou_predictions, low_res_masks

    def get_image_embedding(self) -> torch.Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert (
            self._features is not None
        ), "Features must exist if an image has been set."
        return self._features["image_embed"]

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_predictor(self) -> None:
        """
        Resets the image embeddings and other state variables.
        """
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        self._is_batch = False

    # ===== NEW: cache I/O helpers =====
    @staticmethod
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
        try:
            v = np.array(val).flatten().tolist()
            if len(v) >= 2: return (int(v[-2]), int(v[-1]))
            if len(v) == 1:  return (int(v[0]), int(v[0]))
        except Exception:
            pass
        return None

    @torch.no_grad()
    def set_image_from_cache(self, cache: dict) -> None:
        """
        Restore predictor state from a cached feature pack, bypassing encoder.
        Required keys:
            - "image_embed"  (Tensor [B?,C,Hf,Wf] or [C,Hf,Wf])
            - "high_res_feats" (list[Tensor] pyramid levels)
            - "original_size" / "orig_size"  (H, W)
        """
        assert isinstance(cache, dict), "cache must be a dict"
        dev = self.device

        # resolve embedding
        img_key = None
        for k in ("image_embed", "image_embeddings", "image_features", "vision_feats"):
            if k in cache:
                img_key = k; break
        if img_key is None:
            raise KeyError("cache missing image embedding (image_embed / image_embeddings / ...)")

        image_embed = cache[img_key]
        high_res = cache.get("high_res_feats", None)
        if high_res is None:
            raise KeyError("cache missing 'high_res_feats'")

        def to_dev_fp32(x):
            return (x if isinstance(x, torch.Tensor) else torch.as_tensor(x)).to(dev).to(torch.float32)

        if isinstance(image_embed, list) or (isinstance(image_embed, torch.Tensor) and image_embed.dim()==5):
            # batched embed not needed here; squeeze
            if isinstance(image_embed, list):
                image_embed = image_embed[0]
            elif image_embed.dim()==5 and image_embed.size(0)==1:
                image_embed = image_embed[0]
        image_embed = to_dev_fp32(image_embed)

        assert isinstance(high_res, (list, tuple)) and len(high_res)>0, "high_res_feats must be list"
        high_res = [to_dev_fp32(t) for t in high_res]

        orig_hw = ( self._as_hw_tuple(cache.get("original_size"))
                    or self._as_hw_tuple(cache.get("orig_size"))
                    or self._as_hw_tuple(getattr(self, "_orig_hw", None)) )
        if orig_hw is None:
            s = int(getattr(self.model, "image_size", 1024))
            orig_hw = (s, s)

        self._features = {"image_embed": image_embed, "high_res_feats": high_res}
        self._orig_hw = [tuple(orig_hw)]
        self._is_image_set = True
        self._is_batch = False

    @torch.no_grad()
    def export_cache(self) -> dict:
        """
        Export a compact feature pack for torch.save(...).
        """
        if not self._is_image_set or self._features is None:
            raise RuntimeError("Call set_image(...) before export_cache().")

        def to_cpu_fp16(x: torch.Tensor) -> torch.Tensor:
            return x.detach().to("cpu").contiguous().to(torch.float16)

        pack = {}
        pack["image_embed"] = to_cpu_fp16(self._features["image_embed"])
        hrs = self._features.get("high_res_feats", None)
        assert isinstance(hrs, list) and len(hrs)>0, "high_res_feats missing"
        pack["high_res_feats"] = [to_cpu_fp16(t) for t in hrs]

        # sizes
        if isinstance(self._orig_hw, list) and len(self._orig_hw)>0:
            ohw = tuple(self._orig_hw[0])
        else:
            s = int(getattr(self.model, "image_size", 1024)); ohw = (s, s)
        pack["original_size"] = ohw
        pack["orig_size"] = ohw
        pack["input_size"] = ohw

        # optional meta
        pinp = getattr(self._transforms, "padded_input_image_size", None)
        if pinp is not None:
            try:
                from numpy import array
                pinp = self._as_hw_tuple(pinp) or ohw
            except Exception:
                pinp = ohw
            pack["padded_input_image_size"] = pinp
        pack["meta"] = {
            "sizes_saved_as_tuple": True,
            "original_size_hw": ohw,
            "input_size_hw": ohw,
            "has_high_res_feats": True,
            "levels": len(pack["high_res_feats"]),
        }
        return pack
    
    @torch.no_grad()
    def set_image_batch_from_cache(self, cache_list: list[dict]) -> None:
        """
        批量從快取資料載入 features，跳過影像 encoder。
        cache_list 內每個元素結構需與 set_image_from_cache(cache) 相同：
          - "image_embed": Tensor [C,Hf,Wf] 或 [1,C,Hf,Wf]
          - "high_res_feats": List[Tensor]，各層 [C,Hl,Wl] 或 [1,C,Hl,Wl]
          - "original_size"/"orig_size": (H,W)
        """
        self.reset_predictor()
        dev = self.device
        image_embeds = []
        high_res_pyr = None
        self._orig_hw = []
    
        def to_dev_fp32(x):
            return (x if isinstance(x, torch.Tensor) else torch.as_tensor(x)).to(dev).to(torch.float32)
    
        for cache in cache_list:
            # --- image embed ---
            img_key = None
            for k in ("image_embed", "image_embeddings", "image_features", "vision_feats"):
                if k in cache:
                    img_key = k; break
            assert img_key is not None, "cache missing image embedding"
    
            emb = cache[img_key]
            if isinstance(emb, list):
                emb = emb[0]
            emb = to_dev_fp32(emb)
            if emb.dim() == 3:  # [C,Hf,Wf] -> [1,C,Hf,Wf]
                emb = emb.unsqueeze(0)
            assert emb.dim() == 4 and emb.size(0) == 1
            image_embeds.append(emb)  # 暫時 [1,C,Hf,Wf]
    
            # --- high res pyramid ---
            hrs = cache.get("high_res_feats", None)
            assert isinstance(hrs, (list, tuple)) and len(hrs) > 0, "cache missing 'high_res_feats'"
            hrs = [to_dev_fp32(t) for t in hrs]
            for i in range(len(hrs)):
                if hrs[i].dim() == 3:
                    hrs[i] = hrs[i].unsqueeze(0)  # [1,C,H,W]
            if high_res_pyr is None:
                high_res_pyr = [[h] for h in hrs]  # list(level) of list(batch_items)
            else:
                for i in range(len(hrs)):
                    high_res_pyr[i].append(hrs[i])
    
            # --- original size ---
            ohw = (cache.get("original_size") or cache.get("orig_size"))
            if isinstance(ohw, torch.Tensor):
                ohw = tuple(int(v) for v in ohw.flatten().tolist()[:2])
            elif isinstance(ohw, (list, tuple)):
                ohw = (int(ohw[0]), int(ohw[1]))
            else:
                s = int(getattr(self.model, "image_size", 1024)); ohw = (s, s)
            self._orig_hw.append(ohw)
    
        # 堆成 batch
        image_embed_b = torch.cat(image_embeds, dim=0)  # [B,C,Hf,Wf]
        high_res_feats_b = [torch.cat(level_list, dim=0) for level_list in high_res_pyr]  # 每層 [B,C,H,W]
    
        self._features = {"image_embed": image_embed_b, "high_res_feats": high_res_feats_b}
        self._is_image_set = True
        self._is_batch = True
    