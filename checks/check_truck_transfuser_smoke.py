"""Self-contained smoke test for TruckTransfuserModel + truck_transfuser_loss.

Verifies, with no TruckScenes data and no Lightning wiring, that:
  1. Default `TruckTransfuserConfig` produces an instantiable model.
  2. Forward pass on random input tensors returns the expected output
     keys with the expected shapes.
  3. Loss computation on random targets returns a finite scalar plus a
     populated component dict.

Run from the navsim-truckscenes repo root:

    conda activate truckscenes    # or any env with torch + timm + scipy
    PYTHONPATH=. python tests/agents/truck_transfuser/test_smoke.py

Exits with code 0 on success, 1 on any failed assertion or unexpected
exception. Intended for quick post-port sanity checks; the full
transfuser-truckscenes vs navsim parity test lives elsewhere (#18).
"""
from __future__ import annotations

import sys
import traceback

import torch

from navsim.agents.truck_transfuser.truck_transfuser_config import TruckTransfuserConfig
from navsim.agents.truck_transfuser.truck_transfuser_loss import truck_transfuser_loss
from navsim.agents.truck_transfuser.truck_transfuser_model import TruckTransfuserModel


def _build_random_features(config: TruckTransfuserConfig, batch_size: int):
    """Build feature dict matching the shapes the model's forward expects.

    Channel count for `lidar_feature` flips with `use_ground_plane`
    (1 channel above-only vs 2 channels [below, above]).
    """
    in_channels = 2 if config.use_ground_plane else 1
    features = {
        "camera_feature": torch.randn(batch_size, 3, config.camera_height, config.camera_width),
        "lidar_feature": torch.randn(
            batch_size, in_channels, config.lidar_resolution_height, config.lidar_resolution_width
        ),
        "status_feature": torch.randn(batch_size, 4),
    }
    if config.use_driving_command:
        # One-hot over [Turn Right, Turn Left, Go Straight].
        cmd = torch.zeros(batch_size, 3)
        cmd[:, 2] = 1.0
        features["driving_command"] = cmd
    return features


def _build_random_targets(config: TruckTransfuserConfig, batch_size: int):
    """Build target dict matching loss expectations.

    Always supplies trailer targets; loss masking handles trailer-absent
    samples. Agent labels are binary (0 / 1) with about half present.
    """
    targets = {
        "trajectory": torch.randn(batch_size, config.num_poses, 3),
        "agent_states": torch.randn(batch_size, config.num_bounding_boxes, 5),
        "agent_labels": torch.randint(0, 2, (batch_size, config.num_bounding_boxes)).float(),
    }
    if config.use_trailer_head:
        targets["trailer_trajectory"] = torch.randn(batch_size, config.num_poses, 3)
        # Alternate present (1) / absent (0) so we exercise the masked
        # mean path; the first sample has a trailer, the second does not.
        targets["trailer_mask"] = torch.tensor(
            [1.0 if i % 2 == 0 else 0.0 for i in range(batch_size)]
        )
    return targets


def main() -> int:
    try:
        config = TruckTransfuserConfig()
        # Eval mode so status dropout (training-only) does not change the
        # forward path and assertions stay deterministic over the input.
        torch.manual_seed(0)
        model = TruckTransfuserModel(config).eval()

        batch_size = 2
        features = _build_random_features(config, batch_size)
        with torch.no_grad():
            output = model(features)

        # Expected keys when config defaults stand (use_trailer_head=True,
        # bev_semantic_weight=0 -> no bev_semantic_map).
        expected_keys = {"trajectory", "agent_states", "agent_labels"}
        if config.use_trailer_head:
            expected_keys.add("trailer_trajectory")
        if config.bev_semantic_weight > 0:
            expected_keys.add("bev_semantic_map")
        missing = expected_keys - set(output.keys())
        unexpected = set(output.keys()) - expected_keys
        assert not missing, f"missing forward output keys: {missing}"
        assert not unexpected, f"unexpected forward output keys: {unexpected}"

        # Shape spot-checks.
        assert output["trajectory"].shape == (batch_size, config.num_poses, 3)
        if config.use_trailer_head:
            assert output["trailer_trajectory"].shape == (batch_size, config.num_poses, 3)
        assert output["agent_states"].shape == (batch_size, config.num_bounding_boxes, 5)
        assert output["agent_labels"].shape == (batch_size, config.num_bounding_boxes)

        # Loss path.
        targets = _build_random_targets(config, batch_size)
        loss, components = truck_transfuser_loss(targets, output, config)
        assert torch.isfinite(loss), f"loss is not finite: {loss}"
        assert "truck_l1" in components and "agent_cls" in components and "agent_box" in components
        if config.use_trailer_head and config.trailer_weight > 0:
            assert "trailer_l1" in components, "trailer_l1 missing despite use_trailer_head=True"
        if config.bev_semantic_weight <= 0:
            assert "bev_semantic" not in components

        print("=== smoke test passed ===")
        print(f"forward output keys: {sorted(output.keys())}")
        for k in sorted(output.keys()):
            print(f"  {k}: shape={tuple(output[k].shape)}")
        print(f"\nloss = {float(loss):.6f}")
        print("components:")
        for k, v in components.items():
            print(f"  {k}: {float(v):.6f}")
        return 0
    except Exception:
        print("=== smoke test FAILED ===", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
