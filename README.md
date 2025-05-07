.
├── files
├── finetune_src               # 微调后的test验证部分
│   ├── models
│   │   └── __pycache__
│   ├── r2r                    # test核心代码
│   │   └── __pycache__
│   ├── scripts                # test执行脚本
│   └── utils
│       └── __pycache__
└── LLaMA2-Accessory
    ├── accessory
    │   ├── configs
    │   │   ├── data
    │   │   │   └── finetune
    │   │   │       ├── mm
    │   │   │       └── sg
    │   │   ├── model
    │   │   │   ├── convert
    │   │   │   │   └── falcon
    │   │   │   ├── finetune
    │   │   │   │   ├── mm
    │   │   │   │   └── sg
    │   │   │   └── pretrain
    │   │   └── __pycache__
    │   ├── data
    │   │   ├── conversation
    │   │   │   └── __pycache__
    │   │   └── __pycache__
    │   ├── demos
    │   ├── exps
    │   │   ├── finetune
    │   │   │   ├── mm
    │   │   │   └── sg            # 对预训练模型微调的执行脚本
    │   │   └── pretrain
    │   ├── model
    │   │   ├── LLM
    │   │   │   └── __pycache__
    │   │   └── __pycache__
    │   ├── output_dir              # tensorboard
    │   ├── __pycache__
    │   ├── tools
    │   │   └── data_conversion
    │   │       └── to_alpaca
    │   └── util                 # 微调的核心代码 main_finetune.py
    │       └── __pycache__
    ├── asset
    ├── data_example
    ├── docs
    │   ├── examples
    │   │   └── finetune
    │   │       ├── mm
    │   │       └── sg
    │   ├── finetune
    │   ├── light-eval
    │   └── _static
    │       ├── css
    │       └── images
    ├── light-eval
    │   ├── data
    │   │   ├── gsm8k
    │   │   ├── LLaVA-benchmark
    │   │   ├── math
    │   │   └── MM-Vet
    │   ├── prompt
    │   ├── scripts
    │   └── src
    │       └── eval_utils
    └── SPHINX
        └── figs

69 directories
