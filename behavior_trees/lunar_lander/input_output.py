from enum import IntEnum


class LanderInputIndex(IntEnum):
    x_position = 0
    y_position = 1
    x_velocity = 2
    y_velocity = 3
    angle = 4
    angular_velocity = 5
    left_ground_contact = 6
    right_ground_contact = 7


class LanderInputProperties:
    x_position = {"min": -1.5, "max": 1.5, "type": float}
    y_position = {"min": -1.5, "max": 1.5, "type": float}
    x_velocity = {"min": -5, "max": 5, "type": float}
    y_velocity = {"min": -5, "max": 5, "type": float}
    angle = {"min": -3.14, "max": 3.14, "type": float}  # radians
    angular_velocity = {"min": 0, "max": 5, "type": float}
    left_ground_contact = {"min": 0, "max": 1, "type": bool}
    right_ground_contact = {"min": 0, "max": 1, "type": bool}


class LanderOutputIndex(IntEnum):
    nothing = 0
    left_engine = 1
    main_engine = 2
    right_engine = 3
