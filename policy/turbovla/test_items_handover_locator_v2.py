#!/usr/bin/env python3
"""Unit tests for the sparse v2 items_handover visual prompt."""

import unittest

import torch

from items_handover_locator import ItemsHandoverInferenceAdapter


class SparseLocatorV2Test(unittest.TestCase):
    def make_adapter(self, phase: int):
        adapter = ItemsHandoverInferenceAdapter.__new__(ItemsHandoverInferenceAdapter)
        adapter.mode = "v2_sparse"
        adapter.overlay_strength = 0.18
        adapter.action_count = (0, 100, 210, 225, 300)[phase]
        adapter.targets = {
            "pen": {"cx": 0.75, "cy": 0.40, "width": 0.08, "height": 0.20},
            "holder": {"cx": 0.30, "cy": 0.30, "width": 0.14, "height": 0.24},
        }
        return adapter

    def prepare(self, phase: int):
        adapter = self.make_adapter(phase)
        original = torch.full((3, 48, 64), 0.5)
        obs = {adapter.camera_key: original.clone()}
        output, prompt = adapter.prepare(obs, "Hand the pen to the other hand")
        return original, output[adapter.camera_key], prompt

    def test_phase_zero_marks_pen_and_keeps_original_task(self):
        original, output, prompt = self.prepare(0)
        self.assertFalse(torch.equal(original, output))
        self.assertTrue(prompt.startswith("Hand the pen to the other hand."))
        self.assertIn("grasp the pen", prompt)

    def test_transport_and_handover_phases_keep_rgb_unchanged(self):
        for phase in (1, 2, 4):
            with self.subTest(phase=phase):
                original, output, _ = self.prepare(phase)
                self.assertTrue(torch.equal(original, output))

    def test_insert_phase_marks_holder_only(self):
        original, output, prompt = self.prepare(3)
        self.assertFalse(torch.equal(original, output))
        self.assertIn("insert the pen into the holder", prompt)
        # Holder is left of the pen; magenta cue raises red at its centre.
        self.assertGreater(float(output[0, 14, 19]), float(original[0, 14, 19]))

    def test_v2_cue_is_weaker_than_legacy_default(self):
        image = torch.full((3, 48, 64), 0.5)
        target = {"cx": 0.5, "cy": 0.5, "width": 0.10, "height": 0.20}
        weak = ItemsHandoverInferenceAdapter._blend(
            image, target, (0.0, 1.0, 1.0), 0.18
        )
        dense = ItemsHandoverInferenceAdapter._blend(
            image, target, (0.0, 1.0, 1.0)
        )
        self.assertLess(float((weak - image).abs().max()), float((dense - image).abs().max()))


if __name__ == "__main__":
    unittest.main()
