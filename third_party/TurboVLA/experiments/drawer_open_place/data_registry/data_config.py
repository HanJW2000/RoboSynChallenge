"""CobotMagic click_bell (RoboSynChallenge) data configuration."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


class CobotMagicDrawer_open_placeDataConfig:
    """Three-camera CobotMagic bimanual data, 14-dim absolute qpos (L7+R7,
    gripper dims 6/13 are constant 0 in this sim), 50-step action horizon."""

    video_keys = ["video.cam_high", "video.cam_left_wrist", "video.cam_right_wrist"]
    state_keys = ["state.qpos"]
    action_keys = ["action.qpos"]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(50))

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self):
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={"state.qpos": "min_max"},
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={"action.qpos": "min_max"},
                ),
            ]
        )


ROBOT_TYPE_CONFIG_MAP = {"cobotmagic_drawer_open_place": CobotMagicDrawer_open_placeDataConfig()}
ROBOT_TYPE_TO_EMBODIMENT_TAG = {}

DATASET_NAMED_MIXTURES = {
    # Hard450-only variant: the 1000-episode base overlay was dropped; the
    # mixture is just the pruned 450-episode hard dataset.
    "cobotmagic_drawer_open_place_1000": [
        ("cobotmagic_Sim_drawer_open_place_tv", 1.0, "cobotmagic_drawer_open_place"),
    ],
}
