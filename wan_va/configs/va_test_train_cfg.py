# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_test_cfg import va_test_cfg

va_test_train_cfg = EasyDict(__name__='Config: VA test train')
va_test_train_cfg.update(va_test_cfg)

va_test_train_cfg.dataset_path = '/apdcephfs_gy5/share_303588738/leoyizhang/datasets/robotwin'
va_test_train_cfg.enable_wandb = True
va_test_train_cfg.load_worker = 16
va_test_train_cfg.save_interval = 1000
va_test_train_cfg.gc_interval = 50
va_test_train_cfg.cfg_prob = 0.1

# Training parameters
va_test_train_cfg.learning_rate = 4e-5
va_test_train_cfg.beta1 = 0.9
va_test_train_cfg.beta2 = 0.95
va_test_train_cfg.weight_decay = 0.1
va_test_train_cfg.warmup_steps = 10
va_test_train_cfg.gradient_accumulation_steps = 1
va_test_train_cfg.num_steps = 50000
