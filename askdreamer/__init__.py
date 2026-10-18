"""AskDreamer 包：一个 CPU 可训练的小型潜空间智能体原型。"""

from .env import (
    ABSTAIN_ACTION,
    ACTION_NAMES,
    ACTION_NAMES_WITH_ABSTAIN,
    ASK_ACTION,
    GRID_SIZE,
    MSG_DIM,
    N_ACTIONS,
    N_ACTIONS_WITH_ABSTAIN,
    OBS_DIM,
    HiddenGoalGrid,
)

__all__ = [
    "ABSTAIN_ACTION",
    "ACTION_NAMES",
    "ACTION_NAMES_WITH_ABSTAIN",
    "ASK_ACTION",
    "GRID_SIZE",
    "MSG_DIM",
    "N_ACTIONS",
    "N_ACTIONS_WITH_ABSTAIN",
    "OBS_DIM",
    "HiddenGoalGrid",
]
