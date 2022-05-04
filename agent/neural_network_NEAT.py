import neat
import numpy as np

from typing import Dict, Callable


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
        self.network = neat.nn.FeedForwardNetwork.create(self.genome, self.config)
        self.activation_functions = activation_functions
        self.verbose = verbose
        if self.verbose:
            print(f"Agent with {len(activation_functions)} output")

    def forward(self, observations: np.ndarray) -> np.ndarray:
        if self.verbose:
            print(
                f"Observation received\n\tshape: {observations.shape}\n\tcontent: {observations}"
            )

        output = np.array(self.network.activate(observations))
        cast, focus = (np.argmax(output[-12:-8]), np.argmax(output[-8:]))
        output = np.array(
            [self.activation_functions[i](o) for i, o in enumerate([cast, focus])]
        ).flatten()
        if self.verbose:
            print(f"Actions taken: {output}")
        return output
