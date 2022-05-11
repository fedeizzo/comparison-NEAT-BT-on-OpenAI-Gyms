import numpy as np

def tournament(individuals, size):
    partecipants = list(
        np.random.choice(a=individuals, size=size, replace=False)
    )
    partecipants.sort(key=lambda x: x.fitness, reverse=True)
    return partecipants[0]

