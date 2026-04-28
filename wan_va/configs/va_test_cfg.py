# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import va_shared_cfg

va_test_cfg = EasyDict(__name__='Config: VA test')
va_test_cfg.update(va_shared_cfg)
va_test_cfg.infer_mode = 'server'

va_test_cfg.wan22_pretrained_model_name_or_path = "/apdcephfs_gy5/share_303588738/leoyizhang/model/lingbot-va-base"

va_test_cfg.attn_window = 72
va_test_cfg.frame_chunk_size = 2
va_test_cfg.env_type = 'none'

va_test_cfg.height = 256
va_test_cfg.width = 256
va_test_cfg.action_dim = 30
va_test_cfg.action_per_frame = 16
va_test_cfg.obs_cam_keys = [
    'observation.images.cam_high', 'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist'
]
va_test_cfg.guidance_scale = 5
va_test_cfg.action_guidance_scale = 1

va_test_cfg.num_inference_steps = 25
va_test_cfg.video_exec_step = -1
va_test_cfg.action_num_inference_steps = 50

va_test_cfg.snr_shift = 5.0
va_test_cfg.action_snr_shift = 1.0

va_test_cfg.used_action_channel_ids = list(range(0, 7)) + list(
    range(28, 29)) + list(range(7, 14)) + list(range(29, 30))
inverse_used_action_channel_ids = [
    len(va_test_cfg.used_action_channel_ids)
] * va_test_cfg.action_dim
for i, j in enumerate(va_test_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_test_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_test_cfg.action_norm_method = 'quantiles'
va_test_cfg.norm_stat = {
    "q01": [0.] * 30,
    "q99": [0.] * 30,
}
