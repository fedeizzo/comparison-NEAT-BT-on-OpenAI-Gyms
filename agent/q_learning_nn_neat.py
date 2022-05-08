import neat
import numpy as np

from typing import List, Union

from agent.pytorch_neat import recurrent_net
from scipy.special import softmax

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


class DerkQLearningNEATPlayer:
    def __init__(
        self,
        genome,
        config: neat.Config,
        all_actions: List[np.ndarray],
        verbose: bool = False,
    ):
        self.genome = genome
        self.config = config
        self.network = recurrent_net.RecurrentNet.create(self.genome, self.config)
        self.all_actions = all_actions
        self.verbose = verbose
        if self.verbose:
            print(f"Q learning agent with {len(all_actions)} output")

    def forward(self, observations: np.ndarray) -> np.ndarray:
        if self.verbose:
            print(
                f"Observation received\n\tshape: {observations.shape}\n\tcontent: {observations}"
            )

        output = np.array(self.network.activate(observations.reshape(1, -1)).squeeze(0))
        output = softmax(output)
        output = self.all_actions[np.argmax(output)]

        if self.verbose:
            print(f"Actions taken: {output}")
        return output
