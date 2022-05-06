import neat
import numpy as np

from typing import Dict, Callable

from agent.pytorch_neat import recurrent_net

input_node_names = [
    "Hitpoints",
    "Ability0Ready",
    "FriendStatueDistance",
    "FriendStatueAngle",
    "Friend1Distance",
    "Friend1Angle",
    "Friend2Distance",
    "Friend2Angle",
    "EnemyStatueDistance",
    "EnemyStatueAngle",
    "Enemy1Distance",
    "Enemy1Angle",
    "Enemy2Distance",
    "Enemy2Angle",
    "Enemy3Distance",
    "Enemy3Angle",
    "HasFocus",
    "FocusRelativeRotation",
    "FocusFacingUs",
    "FocusFocusingBack",
    "FocusHitpoints",
    "Ability1Ready",
    "Ability2Ready",
    "FocusDazed",
    "FocusCrippled",
    "HeightFront1",
    "HeightFront5",
    "HeightBack2",
    "PositionLeftRight",
    "PositionUpDown",
    "Stuck",
    "UnusedSense31",
    "HasTalons",
    "HasBloodClaws",
    "HasCleavers",
    "HasCripplers",
    "HasHealingGland",
    "HasVampireGland",
    "HasFrogLegs",
    "HasPistol",
    "HasMagnum",
    "HasBlaster",
    "HasParalyzingDart",
    "HasIronBubblegum",
    "HasHeliumBubblegum",
    "HasShell",
    "HasTrombone",
    "FocusHasTalons",
    "FocusHasBloodClaws",
    "FocusHasCleavers",
    "FocusHasCripplers",
    "FocusHasHealingGland",
    "FocusHasVampireGland",
    "FocusHasFrogLegs",
    "FocusHasPistol",
    "FocusHasMagnum",
    "FocusHasBlaster",
    "FocusHasParalyzingDart",
    "FocusHasIronBubblegum",
    "FocusHasHeliumBubblegum",
    "FocusHasShell",
    "FocusHasTrombone",
    "UnusedExtraSense30",
    "UnusedExtraSense31",
]

output_node_names = [
    "movement",
    "rotation",
    "chase",
    "cast0",
    "cast1",
    "cast2",
    "cast3",
    "focus0",
    "focus1",
    "focus2",
    "focus3",
    "focus4",
    "focus5",
    "focus6",
    "focus7",
]


class DerkNeatNNPlayer:
    def __init__(
        self,
        genome,
        config: neat.Config,
        activation_functions: Dict[int, Callable[[float], float]],
        verbose: bool = False,
    ):
        self.genome = genome
        self.config = config
        # self.network = neat.nn.FeedForwardNetwork.create(self.genome, self.config)
        self.network = recurrent_net.RecurrentNet.create(self.genome, self.config)
        self.activation_functions = activation_functions
        self.verbose = verbose
        if self.verbose:
            print(f"Agent with {len(activation_functions)} output")

    def forward(self, observations: np.ndarray) -> np.ndarray:
        if self.verbose:
            print(
                f"Observation received\n\tshape: {observations.shape}\n\tcontent: {observations}"
            )

        output = np.array(self.network.activate(observations.reshape(1, -1)).squeeze(0))
        cast, focus = (np.argmax(output[-12:-8]), np.argmax(output[-8:]))
        output = np.array(
            [
                *output[:-12],
                *[self.activation_functions[i](o) for i, o in enumerate([cast, focus])],
            ]
        ).flatten()
        if self.verbose:
            print(f"Actions taken: {output}")
        return output
