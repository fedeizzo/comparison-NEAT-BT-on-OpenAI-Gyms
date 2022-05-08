import numpy as np

from argparse import ArgumentParser
from itertools import product

def generate_actions(move, rot, chase):
    move_actions = np.linspace(-1, 1, num=move)
    rot_actions = np.linspace(-1, 1, num=rot)
    chase_actions = np.linspace(0, 1, num=chase)
    cast_actions = np.arange(0, 4)
    focus_actions = np.arange(0, 8)
    
    return list(product(move_actions, rot_actions, chase_actions, cast_actions, focus_actions))

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
