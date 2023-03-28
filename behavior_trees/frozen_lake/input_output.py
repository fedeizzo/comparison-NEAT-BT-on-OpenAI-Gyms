from enum import IntEnum


class LakeInputIndex(IntEnum):
    position = 0

class LakeInputProperties:
    position = {"min": 0., "max": 15., "type": int}


class LakeOutputIndex(IntEnum):
    move_left = 0
    move_down = 1
    move_right = 2
    move_up = 3
