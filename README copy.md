# Abnormal-tooth detection on panoramic dental x-rays

Two Mask2Former models on the [DENTEX](https://huggingface.co/datasets/ibrahimhamamci/DENTEX)
benchmark. Given one panoramic radiograph the pipeline returns, for every
finding, **what is wrong and on which tooth**, with a bounding box and an
instance mask for both the tooth and the pathology:

```
28 teeth detected, 3 finding(s):
  - Deep Caries on tooth 47 (lower-right second molar), score 0.81
  - Deep Caries on tooth 37 (lower-left second molar), score 0.74
  - Impacted on tooth 18 (upper-right third molar), score 0.69
```

and the same thing drawn: thin teal boxes with the FDI number for every tooth,
thick coloured boxes with the diagnosis for every finding.

---

## 1. What the data actually contains

DENTEX ships three hierarchically annotated subsets. Reading the numbers before
designing anything matters, because they rule out the obvious architecture.

| subset | x-rays | instances | per image | what is annotated | labels |
|---|---|---|---|---|---|
| `quadrant` | 693 | 2 772 | 4.0 | jaw quadrants | quadrant |
| `quadrant_enumeration` | 634 | 18 095 | **28.5** | **every tooth** | quadrant, position |
| `quadrant-enumeration-disease` | 705 | 3 529 | **5.0** | **only abnormal teeth** | quadrant, position, diagnosis |
| official validation (hold-out) | 50 | 182 | 3.6 | only abnormal teeth | all three |

Diagnoses: `Impacted`, `Caries`, `Periapical Lesion`, `Deep Caries`.
An FDI number is `quadrant * 10 + position`, so tooth 36 is quadrant 3,
position 6 — the lower-left first molar.

Three facts drive every design decision below.

1. **Healthy teeth are unlabelled in the disease subset.** Training tooth
   detection there would teach the model that a healthy tooth is background.
   So tooth detection and pathology detection are trained on different subsets,
   by two models.
2. **The pathology box is the tooth.** DENTEX annotates the *whole abnormal
   tooth*, not an isolated lesion inside it, so "the box around the problem"
   and "the box around the diseased tooth" are the same rectangle. If you need
   lesion-level boxes (the carious spot inside the crown), DENTEX cannot
   supply them and a different dataset is required.
3. **The classes are long-tailed.** 2 189 caries against 158 periapical
   lesions, a 14:1 ratio; among teeth, third molars (18/28/38/48) are the
   rarest at ~350 examples because they are often missing. Both are handled —
   inverse-frequency class weights and repeat-factor sampling.

A trap: in `train_quadrant.json` the `categories` list is **not sorted by id**
(id 0 is named `"2"`). Every mapping in `dentex/fdi.py` resolves by `name`.

## 2. Architecture

```
                    panoramic x-ray
                           |
      +--------------------+--------------------+
      |                                         |
  TOOTH MODEL                              PATHOLOGY MODEL
  Mask2Former R50                          Mask2Former R50 + hierarchical heads
  32 classes (FDI 11..48)                  primary : diagnosis (4)
  trained on quadrant_enumeration          aux     : quadrant (4)
  (634 x-rays, all teeth)                  aux     : position (8)
      |                                    trained on the disease subset
      |                                    (705 x-rays, abnormal teeth)
      |                                         |
  mask + box + FDI per tooth               mask + box + diagnosis + its own FDI
      |                                         |
      +------------------+----------------------+
                         |
                    LINKING (IoU)
         the tooth detector has seen 18k teeth against
         the pathology model's 3.5k, so when the two
         disagree about the tooth number, it wins
                         |
             StudyReport -> json / markdown / annotated png
```

### Why hierarchical heads rather than one flat label

A single detector over "tooth x diagnosis" would need 128 classes with ~27
training examples each. Instead the pathology model keeps one query per
abnormal tooth and reads three heads off it, so the quadrant head sees all
3 529 instances rather than 1/32 of them. This is the DENTEX baseline's
structure (`HierarchicalDet`), carried over to Mask2Former.

The tooth model needs no auxiliary heads: its flat 32-way label is laid out as
`(quadrant - 1) * 8 + (position - 1)`, so the quadrant and the position are
recoverable from the class index, and the checkpoint stays byte-compatible
with stock Mask2Former.

### Multi-stage transfer

`quadrant (693) -> enumeration (634) -> diagnosis (705)`, each stage
warm-starting the next. `--init-from` strips the classification heads (their
shape changes between stages) and transfers the backbone, pixel decoder and
transformer decoder, which is where the pre-training value sits.

### Two changes to upstream Mask2Former

* **Boxes.** Stock Mask2Former returns `pred_boxes = Boxes(zeros(N, 4))` — the
  line deriving boxes from masks is commented out upstream because COCO
  instance segmentation is scored on masks. This project is specified in
  boxes, so it is turned back on (`MODEL.DENTEX.PREDICT_BOXES`).
* **Mirror-aware flipping.** A horizontal flip of a panoramic x-ray swaps the
  left and right sides of the jaw: quadrant 1 becomes quadrant 2, quadrant 4
  becomes quadrant 3. Flipping without relabelling teaches the model that
  tooth 18 can appear in the lower-left. `INPUT.XRAY.MIRROR_AWARE_FLIP`
  relabels; leave it on whenever `HFLIP_PROB > 0`.

## 3. Layout

```
dentex/
  fdi.py                 every label mapping, FDI <-> quadrant/position <-> class id
  config.py              MODEL.DENTEX + INPUT.XRAY nodes, setup_cfg, validate_cfg
  data/
    download.py          Hub download; streams only the needed zip members
    convert.py           DENTEX json -> detectron2 dicts, stable hash splits, stats
    register.py          DatasetCatalog registration
    mapper.py            augmentation, hierarchy labels, mirror-aware flip
  modeling/
    hierarchy.py         the four levels in one table
    hier_decoder.py      transformer decoder + auxiliary class heads
    hier_criterion.py    matched-query cross-entropy per level, class weights
    hier_maskformer.py   meta-arch: boxes, hierarchy decoding, criterion wiring
  engine/
    trainer.py           DefaultTrainer + Mask2Former optimiser + repeat sampling
    evaluator.py         COCO AP plus the DENTEX triple metric
  inference/
    pipeline.py          both models, deduplication, tooth<->pathology linking
    report.py            ToothFinding / ProblemFinding / StudyReport
    visualize.py         the two-layer figure
configs/                 Base + one yaml per stage
scripts/                 prepare_data.py, train.py, predict.py
tests/                   logic tests that need no GPU and no detectron2
```

`dentex.fdi`, `dentex.data.convert`, `dentex.inference.report` and
`dentex.inference.visualize` import without detectron2, so the annotations and
the figures can be worked on from a plain Python session.

## 4. Installation

Training needs a CUDA GPU. Colab T4 is enough (~1.5 h per stage).

```bash
pip install -r requirements.txt
pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation
git clone https://github.com/facebookresearch/Mask2Former.git
cd Mask2Former/mask2former/modeling/pixel_decoder/ops && python setup.py build install && cd -
export PYTHONPATH=$PYTHONPATH:$PWD/Mask2Former
```

Notes:

* Do **not** reinstall torch on Colab. detectron2 compiles against the torch
  already present; replacing it afterwards breaks the extension.
* The `ops` build compiles the MSDeformAttn CUDA kernel and needs `nvcc`.
  If it fails, Mask2Former falls back to a pure-PyTorch implementation that
  works but is slower — training still runs.
* `notebooks/dentex_colab.ipynb` does all of the above in order.

## 5. Running it

```bash
# 1. data (streams ~5 GB instead of the full 10.9 GB archive)
python scripts/prepare_data.py --root dataset/dentex --holdout-root dataset/dentex_val

# 2. sanity check: build the model and one batch, train nothing
python scripts/train.py --config configs/dentex_tooth_R50.yaml --dry-run

# 3. stage 2 -- tooth detection with FDI enumeration
python scripts/train.py --config configs/dentex_tooth_R50.yaml

# 4. stage 3 -- hierarchical pathology, warm-started from stage 2
python scripts/train.py --config configs/dentex_disease_R50_hier.yaml \
    --init-from outputs/stage2_tooth/model_final.pth

# 5. evaluate on the untouched official hold-out
python scripts/train.py --config configs/dentex_disease_R50_hier.yaml --eval-only \
    MODEL.WEIGHTS outputs/stage3_disease/model_final.pth \
    DATASETS.TEST '("dentex_disease_holdout",)'

# 6. predictions and figures
python scripts/predict.py \
    --tooth-config configs/dentex_tooth_R50.yaml   --tooth-weights outputs/stage2_tooth/model_final.pth \
    --disease-config configs/dentex_disease_R50_hier.yaml --disease-weights outputs/stage3_disease/model_final.pth \
    --images dataset/dentex_val --out outputs/predictions --contact-sheet
```

Stage 1 (`configs/dentex_quadrant_R50.yaml`) is an optional warm-up; it is
cheap but the gain is small, so start at stage 2 if time is short.

## 6. Splits and metrics

The 705 released training x-rays are split 90/10 into `train`/`val` by a stable
hash of the file name (`dentex/data/convert.py`), so the split is identical on
every machine and across reruns. The **official 50-image validation release**
is registered as `dentex_disease_holdout` and is never used for model
selection — that is the number to quote.

Two evaluators run together:

* `COCOEvaluator` — box AP and mask AP on the primary head.
* `HierarchyEvaluator` — what DENTEX is actually scored on:
  * `detection/recall`, `detection/precision` — localisation alone;
  * `level/quadrant`, `level/position`, `level/disease` — per-level accuracy on
    true positives;
  * `fdi/accuracy` — full tooth number correct;
  * `triple/accuracy` and `triple/recall` — quadrant + position + diagnosis all
    correct. **`triple/recall` is the headline number.**

It also logs the most frequent enumeration confusions (`47->46 x5`), which are
usually off-by-one neighbours and tell you whether the quadrant or the position
head is the weak one.

## 7. Known limitations

* **Pathology boxes outline whole teeth**, not lesions — a property of DENTEX,
  not of the model. See §1.
* **Periapical lesions are rare** (158 instances). Expect low AP on that class
  whatever the weighting; report it per class rather than only as a mean.
* **The tooth detector is evaluated on a 56-image internal split.** The
  official hold-out only annotates abnormal teeth, so it cannot measure
  enumeration over a full mouth; `dentex_tooth_holdout_abnormal` is registered
  for completeness but is not a full tooth benchmark.
* **Missing teeth.** The tooth model may hallucinate an extracted tooth to fill
  an expected FDI slot. The one-instance-per-FDI rule in `deduplicate` bounds
  the damage but does not remove it.
* **Not a medical device.** DENTEX is licensed CC BY-NC-SA 4.0: research and
  teaching only, no clinical or commercial use.

## 8. Tests

```bash
python tests/test_pipeline_logic.py      # or: python -m pytest tests -q
```

Covers the FDI round-trip, the out-of-order category remap, mirror-aware
flipping, deduplication and the tooth/pathology linking rules. No GPU, no
detectron2, no dataset needed.

## 9. Citation

```bibtex
@article{hamamci2023dentex,
  title   = {DENTEX: An Abnormal Tooth Detection with Dental Enumeration and
             Diagnosis Benchmark for Panoramic X-rays},
  author  = {Hamamci, Ibrahim Ethem and others},
  journal = {arXiv preprint arXiv:2305.19112},
  year    = {2023}
}
@inproceedings{cheng2022mask2former,
  title     = {Masked-attention Mask Transformer for Universal Image Segmentation},
  author    = {Cheng, Bowen and Misra, Ishan and Schwing, Alexander G. and
               Kirillov, Alexander and Girdhar, Rohit},
  booktitle = {CVPR},
  year      = {2022}
}
```
