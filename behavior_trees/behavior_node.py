from enum import IntEnum, Enum
import numpy as np
from abc import ABC, abstractmethod
import itertools


class BehaviorStates(Enum):
    SUCCESS = 0
    RUNNING = 1
    FAILURE = 2


class BehaviorNodeTypes(Enum):
    ACT = 0
    SEQ = 1
    FALL = 2
    COND = 3
    DEC = 4  # ? will use decorators?


class InputIndex(IntEnum):
    Hitpoints = 0
    Ability0Ready = 1
    FriendStatueDistance = 2
    FriendStatueAngle = 3
    Friend1Distance = 4
    Friend1Angle = 5
    Friend2Distance = 6
    Friend2Angle = 7
    EnemyStatueDistance = 8
    EnemyStatueAngle = 9
    Enemy1Distance = 10
    Enemy1Angle = 11
    Enemy2Distance = 12
    Enemy2Angle = 13
    Enemy3Distance = 14
    Enemy3Angle = 15
    HasFocus = 16
    FocusRelativeRotation = 17
    FocusFacingUs = 18
    FocusFocusingBack = 19
    FocusHitpoints = 20
    Ability1Ready = 21
    Ability2Ready = 22
    FocusDazed = 23
    FocusCrippled = 24
    HeightFront1 = 25
    HeightFront5 = 26
    HeightBack2 = 27
    PositionLeftRight = 28
    PositionUpDown = 29
    Stuck = 30
    UnusedSense31 = 31
    HasTalons = 32
    HasBloodClaws = 33
    HasCleavers = 34
    HasCripplers = 35
    HasHealingGland = 36
    HasVampireGland = 37
    HasFrogLegs = 38
    HasPistol = 39
    HasMagnum = 40
    HasBlaster = 41
    HasParalyzingDart = 42
    HasIronBubblegum = 43
    HasHeliumBubblegum = 44
    HasShell = 45
    HasTrombone = 46
    FocusHasTalons = 47
    FocusHasBloodClaws = 48
    FocusHasCleavers = 49
    FocusHasCripplers = 50
    FocusHasHealingGland = 51
    FocusHasVampireGland = 52
    FocusHasFrogLegs = 53
    FocusHasPistol = 54
    FocusHasMagnum = 55
    FocusHasBlaster = 56
    FocusHasParalyzingDart = 57
    FocusHasIronBubblegum = 58
    FocusHasHeliumBubblegum = 59
    FocusHasShell = 60
    FocusHasTrombone = 61
    UnusedExtraSense30 = 62
    UnusedExtraSense31 = 63


class OutputIndex(IntEnum):
    MoveX = 0
    Rotate = 1
    ChaseFocus = 2
    CastingSlot = 3
    ChangeFocus = 4


class BehaviorNode(ABC):
    BEHAVIOR_NODE_ID = itertools.count()

    def __init__(self, type, parameters) -> None:
        self.type = type
        self.parameters = parameters
        self.status = None
        self.id = next(BehaviorNode.BEHAVIOR_NODE_ID)

    def tick(self, input):
        """Launches the tick to the node, it implements the starndard routine
        with the check of applicable and the call to method run adterwards.

        Args:
            input (np.ndarray): observations input array.

        Returns:
            tuple[BehaviorStates, np.ndarray]: the result of the run method is
            a tuple with the return state (SUCCESS, RUNNING or FAILURE) and the
            action to perform.
        """
        if self.applicable(input):
            result = self.run(input)
        else:
            result = (BehaviorStates.FAILURE, np.zeros(shape=(5,)))
        self.status = result[0]
        return result

    @abstractmethod
    def applicable(self, input):
        """Check if the node run is executable.

        Args:
            input (np.ndarray): observations input array.
        """
        pass

    @abstractmethod
    def run(self, input):
        """Runs the node.
        In composite nodes it will tick the children and implement the logic of
        the switch between the various children and the combination of returns.

        In action nodes it will return the action and the state.

        In condition nodes it will return the result of the condition (the
        state) and an empty action.

        Args:
            input (np.ndarray): observations input array.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_random_node():
        """Generate a random instance of the BehaviorNode."""
        pass

    @abstractmethod
    def mutate(self, prob):
        """Randomly mutates the node with probability prob.

        Args:
            prob (float): probability of the mutation, between 0 and 1.
        """
        pass

    def __str__(self, indent=0) -> str:
        string_form = "\t" * indent
        string_form += f"{self.__class__.__name__}#{self.id}\tparams: {self.parameters}"
        return string_form

    def is_regular(self):
        """Checks if the tree is regular, namely if it does not contain two 
        times the very same node.
        """
        tree_nodes: set[BehaviorNode] = set()
        nodes = [self]
        while len(nodes) > 0:
            node = nodes.pop()
            if hasattr(node, "children"):
                for child in node.children:
                    nodes.append(child)
            if node.id in tree_nodes:
                return False
            tree_nodes.add(node.id)
        return True

    def copy(self):
        """Manual implementation of deepcopy.
        """
        pass


if __name__ == "__main__":
    print(BehaviorStates.SUCCESS)
    print(BehaviorStates.RUNNING)
    print(BehaviorStates.FAILURE)
    print(InputIndex.__getitem__("Ability0Ready"))
