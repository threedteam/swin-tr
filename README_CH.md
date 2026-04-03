# SwinTR — Scene Text Recognition with Swin Transformer

SwinTR 是一个基于 Swin Transformer 的场景文本识别（STR）实现，包含训练、验证和测试/推理的完整代码。此仓库提供从数据准备、增强、模型训练到评估与可视化的一套工具。

主要功能
- 支持多种 Transformer/Swin 变体作为特征/编码器（通过 `--TransformerModel` 指定）
- 支持字符级、BPE 与 WordPiece 多种标签编码方式
- 数据增强、LMDB/RawImage 数据加载及批次平衡策略
- 训练/验证/测试流程一体化，支持保存模型与训练日志

依赖（推荐）：请参阅 `requirements.txt`。

快速开始
1. 安装依赖（示例）：

```bash
python -m pip install -r requirements.txt
```

2. 数据加载：
```bash
python create_lmdb_dataset.py --data_dir /path/to/data --lmdb_dir /path/to/lmdb
```

3. 训练示例：

```bash
python main.py --train_data /path/to/train --valid_data /path/to/valid --test_data /path/to/test \
  --select_data inno --batch_ratio 1 \
  --Transformer swin-str-fusion --TransformerModel swin_small_patch4_window7_224_fusion \
  --imgH 224 --imgW 224 --manualSeed=226 --workers=12 \
  --scheduler --batch_size=32 --rgb \
  --valInterval 109 --num_iter 21800 --lr 0.0001 \
  --exp_name my_exp --saved_path ./saved_models
```

4. 测试/评估示例：

```bash
python main.py --eval_data /path/to/test \
  --Transformer swin-str-fusion --TransformerModel swin_small_patch4_window7_224_fusion \
  --benchmark_all_eval \
  --data_filtering_off --rgb \
  --imgH 224 --imgW 224 \
  --model_dir ./saved_models/best_accuracy.pth \
  --exp_name test
```

数据准备
- 支持 LMDB 格式与 Raw image 目录。可使用 `create_lmdb_dataset.py` 将原始图片与标注转换为 LMDB。
- 词典与模型配置位于 `data/dict/`。

保存与日志
- 训练过程中模型权重、日志与绘图文件默认保存在 `./saved_models/<exp_name>/`。

文件结构（简要）
- `main.py`：入口，选择训练或测试。
- `train.py`：训练脚本。
- `test.py`：测试 / 验证脚本。
- `data/`：数据加载、增强与 LMDB 支持。
- `modules/`：模型实现（包括 Swin-TR 变体与辅助模块）。
- `utils/`：工具函数（包含绘图、标签转换、参数解析等）。

注意事项
- 请根据目标环境选择合适的 `torch`/`torchvision` 版本与 CUDA 版本。
- 模型与字典文件较大，请在提交仓库前将训练生成的模型置于 `.gitignore` 或另行下载说明中。

许可证与致谢
- 本项目基于并借鉴以下开源项目：
  - AdvancedLiterateMachinery / MGP-STR — https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/OCR/MGP-STR — 许可证：Apache License 2.0
  - Microsoft / Swin-Transformer — https://github.com/microsoft/Swin-Transformer — 许可证：MIT License

请在使用或发布本仓库前，查阅并遵守上述原仓库的 LICENSE/NOTICE 要求（例如 Apache-2.0 要求保留 NOTICE 内容）。