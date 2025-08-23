# Supervised Few-Shot Segmentation with SAM2 + Grid Prompting

本專案實作了一個 **監督式 Few-Shot Segmentation** 框架，結合 **DINOv3** 特徵與 **SAM2** 作為分割解碼器，透過 **grid-based prompt selection** 進行訓練與推論。

---

## 🎯 目標
- 在 **PASCAL-5i** few-shot segmentation 任務中，達到高品質分割。
- 利用 **support image** 建立 **prototype (前景/背景)**，指引 **query image** 的分割。
- 模型不直接輸出 segmentation mask，而是透過 **16×16 grid (pos/neg/neutral)** 選點，再交給 **SAM2 predictor** 解碼成最終 mask。

---

## 🛠 方法概述
1. **資料前處理**
   - 針對 VOC/COCO 生成 **binary segmentation masks** 與 manifest CSV。
   - 建立 episodic dataset (support/query)。
   - 使用 **SAM2 + DINOv3** 預先計算特徵，快取於 `.pt` 檔案，避免重複編碼。

2. **模型設計**
   - **Backbone features**：
     - SAM2 → `[256, 64, 64]`
     - DINOv3 → `[768, 32, 32]`
   - **Prototype 提取**：
     - Support mask → 前景/背景原型 (SAM, DINO)。
   - **Grid classification head**：
     - 輸出 `[3, 16, 16]` (pos / neutral / neg)。
   - **點選擇**：
     - 從 grid logits 中選出前景 / 背景點 → SAM2 predictor → mask。

3. **訓練流程**
   - **Loss**：
     - Grid CE loss (忽略 neutral)。
     - Dice loss (SAM2 預測 mask vs GT)。
   - **優化器與排程**：
     - AdamW + warmup + cosine annealing。
   - **驗證**：
     - IoU 計算 (SAM2 cached predictor)。