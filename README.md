# 项目目录结构

## 主要目录

- **finetune_src** - 微调后的test验证部分
  - models
  - r2r - test核心代码
  - scripts - test执行脚本
  - utils

- **LLaMA2-Accessory**
  - accessory
    - configs
      - data/finetune/sg
    - exps/finetune/sg - 对预训练模型微调的执行脚本
    - model
    - output_dir - tensorboard
    - util - 微调的核心代码 main_finetune.py
  - docs
  - light-eval
  - SPHINX

## 关键组件

1. **微调部分**：
   - 微调脚本：`LLaMA2-Accessory/accessory/exps/finetune/sg`
   - 微调核心代码：`LLaMA2-Accessory/accessory/util/main_finetune.py`

2. **测试部分**：
   - 测试核心代码：`finetune_src/r2r`
   - 测试执行脚本：`finetune_src/scripts`

3. **可视化**：
   - Tensorboard：`LLaMA2-Accessory/accessory/output_dir`
