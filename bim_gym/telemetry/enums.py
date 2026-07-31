from enum import Enum


class EpisodePhase(str, Enum):
    Train = "train"
    Eval = "eval"
    Debug = "debug"
