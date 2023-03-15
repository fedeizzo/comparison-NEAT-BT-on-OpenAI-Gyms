import numpy as np

from argparse import ArgumentParser
from itertools import product


def generate_actions(actions_space, move=-1, rot=-1, chase=-1):
    if actions_space == "linespace" and move > 0 and rot > 0 and chase > 0:
        move_actions = np.linspace(-1, 1, num=move)
        rot_actions = np.linspace(-1, 1, num=rot)
        chase_actions = np.linspace(0, 1, num=chase)
        cast_actions = np.arange(1, 2)
        focus_actions = np.arange(0, 8)
        actions_map = list(
            product(
                move_actions, rot_actions, chase_actions, cast_actions, focus_actions
            )
        )
    elif actions_space == "ranged_weapon":
        # pistol, magnum, blaster
        move_actions = np.linspace(-1, 1, num=move)
        rot_actions = np.linspace(-1, 1, num=rot)
        chase_actions = np.linspace(0, 1, num=chase)
        cast_actions = np.array([1])
        focus_actions = np.array([0, 4, 5, 7])
        actions_map = list(
            product(
                move_actions, rot_actions, chase_actions, cast_actions, focus_actions
            )
        )
    elif actions_space == "melee_weapon":
        # talon, blood claws, cleavers, crippers
        move_actions = np.linspace(-1, 1, num=move)
        rot_actions = np.linspace(-1, 1, num=rot)
        chase_actions = np.linspace(0, 1, num=chase)
        cast_actions = np.array([1])
        focus_actions = np.array([0, 4, 5, 7])
        actions_map = list(
            product(
                move_actions, rot_actions, chase_actions, cast_actions, focus_actions
            )
        )
    elif actions_space == "restricted":
        actions_map = [
            [0, 0, 0, 0, 0],       # nothing
            [1, 0, 0, 0, 0],       # move forward
            [1, 1, 0, 0, 0],       # move forward, turn right
            [1, -1, 0, 0, 0],      # move forward, turn left
            [-1, 0, 0, 0, 0],      # move backward
            [-1, 1, 0, 0, 0],      # move backward, turn right
            [-1, -1, 0, 0, 0],     # move backward, turn left
            [0, 1, 0, 0, 0],       # turn right
            [0, 0.5, 0, 0, 0],     # turn half right
            [0, -0.5, 0, 0, 0],    # turn half left
            [0, -1, 0, 0, 0],      # turn left
            [0, 0, 0, 0, 0],       # not chase focus
            [0, 0, 0.3, 0, 0],     # chase 1/3 focus
            [0, 0, 0.7, 0, 0],     # chase 2/3 focus
            [0, 0, 1, 0, 0],       # chase focus
            [0, 0, 0, 1, 0],       # cast 1
            [0, 0, 0, 2, 0],       # cast 2
            [0, 0, 0, 3, 0],       # cast 3
            [0, 0, 0, 0, 1],       # focus 1
            [0, 0, 0, 0, 2],       # focus friend 2
            [0, 0, 0, 0, 3],       # focus friend 3
            [0, 0, 0, 0, 4],       # focus enemy statue
            [0, 0, 0, 0, 5],       # focus enemy 1
            [0, 0, 0, 0, 6],       # focus enemy 2
            [0, 0, 0, 0, 7],       # focus enemy 3
        ]
    else:
        raise ValueError(
            "Possible actions_space are: linespace and restricted, with linespace pass positive arguemnts"
        )

    return actions_map


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument(
        "-m",
        "--movement-split",
        help="Discrete representation number for movement",
        type=int,
        required=True,
    )
    p.add_argument(
        "-r",
        "--rotation-split",
        help="Discrete representation number for rotation",
        type=int,
        required=True,
    )
    p.add_argument(
        "-c",
        "--chase-focus-split",
        help="Discrete representation number for chase focus",
        type=int,
        required=True,
    )

    args = p.parse_args()
    all_actions = generate_actions(
        args.movement_split,
        args.rotation_split,
        args.chase_focus_split,
    )
