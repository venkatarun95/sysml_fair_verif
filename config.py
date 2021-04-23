class Config:
    # Number of nodes
    N: int = 3
    # Number of iterations (S stands for steps)
    S: int = 3
    # Link rate. Units are arbitrary
    C: int = 1
    # Maximum number of flows (including ours)
    F: int = 2
    # Maximum number of events per link
    E: int = 20
    # Whether or not to keep track of unsat core while solving
    unsat_core: bool = False

    def __init__(self):
        pass
