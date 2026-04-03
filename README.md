# SwinTR — Scene Text Recognition (Swin Transformer)

SwinTR is an implementation of Scene Text Recognition (STR) built on the Swin Transformer backbone. This repository contains end-to-end code for data preparation, augmentation, training, validation and evaluation, plus utilities for visualization and logging.

Key features
- Supports multiple Transformer/Swin variants as backbones (`--TransformerModel`)
- Supports character-level, BPE and WordPiece label encodings
- Data augmentation, LMDB/raw image loading and batch-balancing strategies
- Integrated training / validation / test workflows with checkpointing and logging

Dependencies
- See `requirements.txt` for a minimal dependency list. Pin specific `torch`/`torchvision` versions according to your CUDA runtime.

Quick start
1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Data loading:

```bash
python create_lmdb_dataset.py --data_dir /path/to/data --lmdb_dir /path/to/lmdb
```

3. Training example:

```bash
python main.py --train_data /path/to/train --valid_data /path/to/valid --test_data /path/to/test \
  --select_data inno --batch_ratio 1 \
  --Transformer swin-str-fusion --TransformerModel swin_small_patch4_window7_224_fusion \
  --imgH 224 --imgW 224 --manualSeed=226 --workers=12 \
  --scheduler --batch_size=32 --rgb \
  --valInterval 109 --num_iter 21800 --lr 0.0001 \
  --exp_name my_exp --saved_path ./saved_models
```

4. Evaluation / inference example:

```bash
python main.py --eval_data /path/to/test \
  --Transformer swin-str-fusion --TransformerModel swin_small_patch4_window7_224_fusion \
  --benchmark_all_eval \
  --data_filtering_off --rgb \
  --imgH 224 --imgW 224 \
  --model_dir ./saved_models/best_accuracy.pth \
  --exp_name test
```

Data preparation
- The repo supports LMDB-format datasets and raw image folders. Use `create_lmdb_dataset.py` to convert labeled images into an LMDB dataset.
- Dictionaries and token files are under `data/dict/`.

Saving and logs
- Checkpoints, logs and visualization outputs are saved under `./saved_models/<exp_name>/` by default.

Repository layout (brief)
- `main.py`: entry point (train or test mode)
- `train.py`: training loop and checkpointing
- `test.py`: validation / evaluation scripts
- `data/`: dataset loading, augmentation and LMDB helpers
- `modules/`: model implementations (Swin/TR variants and helpers)
- `utils/`: utility functions (plotting, label converters, arg parsing, etc.)

Notes
- Choose `torch`/`torchvision` versions compatible with your CUDA and GPU drivers.
- Large model or dictionary files generated during training should be excluded from the repository via `.gitignore` or hosted separately.

License
- This repository's code (except where otherwise noted for third-party components) is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
- Third-party components retain their original licenses; copies are collected in [licenses/](licenses/):
  - AdvancedLiterateMachinery (MGP-STR): [Apache-2.0](licenses/AlibabaResearch-AdvancedLiterateMachinery-LICENSE-APACHE-2.0.txt)
  - Microsoft Swin-Transformer: [MIT](licenses/Microsoft-Swin-Transformer-LICENSE-MIT.txt)