#!/usr/bin/env python3
"""Tests for the GroundingDINO locator hook in turbovla_worker.py.

The hook reuses the training-matched ItemsHandoverInferenceAdapter: cam_high
is converted to a CHW float tensor, blended via adapter.prepare, converted
back to PIL, and the phase prompt replaces the incoming task text.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


def _load_worker():
    module_path = Path(__file__).with_name("turbovla_worker.py")
    spec = importlib.util.spec_from_file_location("turbovla_worker", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeAdapter:
    def __init__(self, prompt="Stage GRASP_PEN: locate and pick up the pen"):
        self.prompt = prompt
        self.seen_shapes = []
        self.seen_ranges = []
        self.advance_calls = []

    def prepare(self, obs, task):
        tensor = obs["observation.images.cam_high"]
        self.seen_shapes.append(tuple(tensor.shape))
        self.seen_ranges.append((float(tensor.min()), float(tensor.max())))
        # Deterministic "detection" effect: paint the first pixel cyan-ish so
        # the test can observe that the blended tensor is what gets returned.
        tensor = tensor.clone()
        tensor[:, 0, 0] = torch.tensor([0.0, 1.0, 1.0])
        obs["observation.images.cam_high"] = tensor
        return obs, self.prompt

    def advance(self, n_actions):
        self.advance_calls.append(int(n_actions))


class LocatorHookTest(unittest.TestCase):
    def setUp(self):
        self.worker = _load_worker()
        height, width = 8, 12
        high = np.zeros((height, width, 3), dtype=np.uint8)
        high[..., 0] = 200
        self.cam_high = high
        self.wrist = np.full((height, width, 3), 90, dtype=np.uint8)
        self.images = [
            self.worker.np_to_pil(self.cam_high),
            self.worker.np_to_pil(self.wrist),
            self.worker.np_to_pil(self.wrist),
        ]

    def test_apply_locator_converts_blends_and_replaces_lang(self):
        adapter = FakeAdapter()
        images, lang = self.worker.apply_locator(
            adapter, self.images, "pick up the pen"
        )
        self.assertEqual(lang, adapter.prompt)
        self.assertEqual(adapter.seen_shapes, [(3, 8, 12)])
        lo, hi = adapter.seen_ranges[0]
        self.assertTrue(0.0 <= lo <= hi <= 1.0)
        # cam_high rebuilt from the blended tensor: pixel (0,0) is cyan.
        first = np.asarray(images[0])
        self.assertLess(int(first[0, 0, 0]), 60)
        self.assertGreater(int(first[0, 0, 1]), 200)
        self.assertGreater(int(first[0, 0, 2]), 200)
        # untouched background keeps the original red channel value.
        self.assertEqual(int(first[4, 6, 0]), 200)

    def test_apply_locator_leaves_wrist_cameras_untouched(self):
        adapter = FakeAdapter()
        images, _ = self.worker.apply_locator(adapter, self.images, "task")
        for wrist_image in images[1:]:
            self.assertTrue(
                (np.asarray(wrist_image) == 90).all(),
                "wrist cameras must pass through unchanged",
            )

    def test_apply_locator_without_adapter_is_identity(self):
        images, lang = self.worker.apply_locator(None, self.images, "task")
        self.assertEqual(lang, "task")
        self.assertIs(images, self.images)

    def test_apply_locator_rejects_non_cam_high_first(self):
        adapter = FakeAdapter()
        with self.assertRaises(ValueError):
            self.worker.apply_locator(
                adapter,
                list(reversed(self.images)),
                "task",
                cam_order=["cam_left_wrist", "cam_high", "cam_right_wrist"],
            )


if __name__ == "__main__":
    unittest.main()
