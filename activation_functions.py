from typing import Union
def identity(x: Union[float, int]) -> Union[float, int]:
    return x


def ReLU(x: Union[float, int]) -> Union[float, int]:
    if x < 0:
        return 0
    else:
        return x
