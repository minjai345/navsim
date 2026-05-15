"""Truck TransFuser config.

Ported from transfuser-truckscenes/configs/_base.py. Strictly the model
side -- adapter and training-loop concerns are stripped:

  - Adapter (driving_command derivation, hitch geometry): handled in
    `navsim/common/truckscenes/` and not stored here.
  - Training (optimizer / weight_decay / lr warmup): in navsim convention
    these are passed to `TruckTransfuserAgent.__init__` and the Lightning
    trainer config, not the model config. Left out here.

Everything that affects model architecture or in-forward behavior
(including `status_dropout_p`, which is consumed inside
`TruckTransfuserModel.forward`) stays.

Field defaults mirror the v3 / v9 baselines from transfuser-truckscenes.
Each downstream experiment overrides via dataclass replace or, once
Lightning wiring lands, a Hydra config file.
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class TruckTransfuserConfig:
    """Global TruckTransFuser model config."""

    # === Backbone ===
    image_architecture: str = "resnet34"
    lidar_architecture: str = "resnet34"

    # Latent TF mode -- restricts agent loss to a forward cone.
    latent: bool = False
    latent_rad_thresh: float = 4 * np.pi / 9

    # === LiDAR pre-processing ===
    max_height_lidar: float = 100.0     # ego z upper bound (~no filter)
    pixels_per_meter: float = 4.0       # BEV resolution
    hist_max_per_pixel: int = 5         # per-pixel histogram clip

    # BEV range. ±32 m symmetric matches the v3 baseline; v6+ widens to
    # x in [-32, 48] etc. via override.
    lidar_min_x: float = -32
    lidar_max_x: float = 32
    lidar_min_y: float = -32
    lidar_max_y: float = 32

    # below(<= split) / above(> split) channel split, ego ground plane z=0.
    lidar_split_height: float = 0.2
    # True -> [below, above] 2-channel; False -> above only.
    use_ground_plane: bool = False

    lidar_seq_len: int = 1   # no multi-frame LiDAR

    # === Image / BEV input grids ===
    # camera_width 1536 = NavSim 1024 extended for TruckScenes 4-cam
    # raw layout (cam_l0/r0/l2/r2) stitched on the front pair inside the
    # FeatureBuilder.
    camera_width: int = 1536
    camera_height: int = 256
    lidar_resolution_width: int = 256
    lidar_resolution_height: int = 256

    img_vert_anchors: int = 256 // 32
    img_horz_anchors: int = 1536 // 32
    lidar_vert_anchors: int = 256 // 32
    lidar_horz_anchors: int = 256 // 32

    # === GPT fusion ===
    block_exp = 4
    n_layer = 2
    n_head = 4
    n_scale = 4
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    gpt_linear_layer_init_mean = 0.0
    gpt_linear_layer_init_std = 0.02
    gpt_layer_norm_init_weight = 1.0

    perspective_downsample_factor = 1
    transformer_decoder_join = True
    detect_boxes = True
    # No HD map -> bev_semantic disabled by default for TruckScenes.
    use_bev_semantic = False
    use_semantic = False
    use_depth = False
    add_features = True

    # === Trajectory / agent decoder transformer ===
    tf_d_model: int = 256
    tf_d_ffn: int = 1024
    tf_num_layers: int = 3
    tf_num_head: int = 8
    tf_dropout: float = 0.0

    # === Detection (auxiliary) ===
    num_bounding_boxes: int = 30

    # === Loss weights ===
    trajectory_weight: float = 10.0
    agent_class_weight: float = 10.0
    agent_box_weight: float = 1.0
    bev_semantic_weight: float = 0.0  # disabled: no HD map
    # ego_trailer future L1 loss weight. Trailer-free samples are mask=0
    # so per-batch supervision strength scales with present-trailer rate.
    trailer_weight: float = 10.0

    # Trailer trajectory head toggle. True -> trailer query slot + head
    # built and a "trailer_trajectory" key is emitted. False -> trailer
    # slot removed from query embedding entirely (strict truck-only
    # ablation per the paper baseline). Setting `trailer_weight=0` alone
    # does NOT remove the head -- the capacity stays. Use this flag for
    # apples-to-apples ablation.
    use_trailer_head: bool = True

    # === BEV semantic head (structurally present, weight=0) ===
    bev_pixel_width: int = 256
    bev_pixel_height: int = 128
    bev_pixel_size: float = 0.25

    num_bev_classes = 7
    bev_features_channels: int = 64
    bev_down_sample_factor: int = 4
    bev_upsample_factor: int = 2

    # === Trajectory target ===
    num_poses: int = 8                          # 8 future poses
    trajectory_sampling_time: float = 4.0       # 4 s horizon
    trajectory_sampling_interval: float = 0.5   # keyframe interval

    # Status feature dropout. Used only during training, only when
    # use_ego_status=True. See TruckTransfuserModel.forward for the
    # rationale: vx*dt produces a trivial mapping to trajectory and
    # crowds out image/lidar gradients without dropout. Default 0.5
    # forces the vision branch to carry signal.
    status_dropout_p: float = 0.5

    # NavSim TransFuser parity: concat the 3-way one-hot driving_command
    # ([Turn Right, Turn Left, Go Straight]) into the status encoder
    # input. The command itself is derived by the adapter (heading-mode
    # threshold over future trajectory).
    use_driving_command: bool = False

    # VAD §4.2: drop ego status (vx, vy, ax, ay) to avoid shortcut
    # learning in open-loop planning eval. Default True for v3~v7
    # parity; set False in configs that want VAD-style omission. When
    # False the status encoder drops the 4 ego-status channels and
    # status_dropout has no effect.
    use_ego_status: bool = True

    # === Derived properties ===
    @property
    def bev_semantic_frame(self) -> Tuple[int, int]:
        return (self.bev_pixel_height, self.bev_pixel_width)

    @property
    def bev_radius(self) -> float:
        values = [
            self.lidar_min_x,
            self.lidar_max_x,
            self.lidar_min_y,
            self.lidar_max_y,
        ]
        return max([abs(value) for value in values])
