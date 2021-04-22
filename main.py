from utils import ModelDict, model_to_dict

import matplotlib.pyplot as plt
from typing import Tuple
from z3 import And, If, Implies, Not, Or, Real, Solver

'''# A node's local view

A node goes through the following phases. Training (Tr) -> Summing (Su) ->
Broadcast (Br). The compute times for these are C_tr, C_su and 0
respectively. The model parameters have size P bytes.

In the summing phase, it sends and receives in N blocks of size B. The size of
the model is B * N bytes, where N is the number of nodes. The nodes are
arranged in a ring which determines who sends blocks to whom. It can send the
first block as soon as training is done, but sending every subsequent block is
contingent on receiving the previous block C_su seconds ago and having finished
transmitting the last block.

When a node receives its last block, it moves into the broadcast phase. Here,
each node has one block which it wants to broadcast to everybody. Blocks go in
a cycle, with nodes transmitting as soon as they finished transmitting the last
block and they have the next block to transmit. Once a node has all the blocks,
it can begin the training phase of the next iterations

'''


class Config:
    # Number of nodes
    N: int = 4
    # Number of iterations (S stands for steps)
    S: int = 5
    # Number of timesteps we compute cross-traffic over
    T: int = 10
    # Link rate. Units are arbitrary
    C: int = 1

    def __init__(self):
        pass


class Variables:
    def __init__(self, c: Config):
        self.C_tr = Real("C_tr")
        self.C_su = Real("C_su")
        self.B = Real("B")

        # The time at which the s^th iteration's training starts
        self.tr = [[Real(f"tr_{n},{s}") for s in range(c.S)] for n in
                   range(c.N)]
        # The time at which we start trasmitting the i^th block when summing in
        # the s^th iteration. Note, the last sum event is virtual: it doesn't
        # really happen, it merely indicates when the first broadcast started
        self.su = [[[Real(f"su_{n},{s},{i}") for i in range(c.N)] for s in
                    range(c.S)] for n in range(c.N)]
        # The time at which we start trasmitting the n^th block when
        # broadcasting in the s^th iteration. Similar to su, the last event is
        # virtual
        self.br = [[[Real(f"br_{n},{s},{i}") for i in range(c.N)] for s in
                    range(c.S)] for n in range(c.N)]

        # Transmit time for transmitting during summing
        self.su_tx = [[[Real(f"su_tx_{n},{s},{i}") for i in range(c.N)] for s
                       in range(c.S)] for n in range(c.N)]
        # Transmit time for transmitting during broadcast
        self.br_tx = [[[Real(f"br_tx_{n},{s},{i}") for i in range(c.N)] for s
                       in range(c.S)] for n in range(c.N)]

        # Number of our own bytes waiting to be transmitted at the given link
        self.our_bytes = [[Real("our_bytes_{n},{t}") for t in range(c.T)] for n
                          in range(c.N)]
        # Number of competing bytes waiting to be transmitted at the given link
        self.their_bytes = [[Real("their_bytes_{n},{t}") for t in range(c.T)]
                            for n in range(c.N)]
        # Number of bytes competing flows sent
        self.their_arr = [[Real("their_arr_{n},{t}") for t in range(c.T)] for n
                          in range(c.N)]


def phases(c: Config, o: Solver, v: Variables):
    ''' Constraints for when the various phases happen for each node '''
    for n in range(c.N):
        pre = (n - 1) % c.N
        nex = (n + 1) % c.N
        for s in range(c.S):
            if s == 0:
                o.add(v.tr[n][s] == 0)
            else:
                # Note, the last broadcast is a pseudo-event
                o.add(v.tr[n][s] == v.br[pre][s-1][-1])

            # First sum transmit starts after training is done
            o.add(v.su[n][s][0] == v.tr[n][s] + v.C_tr)

            # First broadcast happens when the last sum transmit starts because
            # the last sum transmit is not really needed. It is a pseudo-event
            o.add(v.br[n][s][0] == v.su[n][s][-1])

            for i in range(1, c.N):
                # Summing phase:
                # The next block is ready to be sent
                ready = v.su[pre][s][i-1] + v.su_tx[pre][s][i-1] + v.C_su
                # Tx link is clear for sending
                clear = v.su[n][s][i-1] + v.su_tx[n][s][i-1]
                # When are we sending the next block
                o.add(v.su[n][s][i] == If(ready > clear, ready, clear))

                # Broadcast phase: analogous to summing phase
                ready = v.br[pre][s][i-1] + v.br_tx[pre][s][i-1]
                clear = v.br[n][s][i-1] + v.br_tx[n][s][i-1]
                o.add(v.br[n][s][i] == If(ready > clear, ready, clear))

            # For now, all except one transmit times are equal
            for i in range(c.N):
                if n == 0:
                    o.add(v.su_tx[n][s][i] == 2)
                    o.add(v.br_tx[n][s][i] == 2)
                else:
                    o.add(v.su_tx[n][s][i] == 1)
                    o.add(v.br_tx[n][s][i] == 1)


def tx_times(c: Config, o: Solver, v: Variables):
    ''' Figure out transmit times based on fair sharing policy '''
    for n in range(c.N):
        o.add(v.our_bytes[n][0] == 0)
        o.add(v.their_bytes[n][0] == v.their_arr[n][0])

        # This is the list of events that occured on this link. We have this for
        # convenience. Each event is a time instant where a flow started
        events = []
        for s in range(c.S):
            events.extend(v.su[n][s][:-1])
            events.extend(v.br[n][s][:-1])

        for t in range(1, c.T):
            # Figure out when flows active at time t-1 end
            s.add(Implies(v.their_bytes[t-1] > v.our_bytes[t-1]))

            # Find whether there was an event between t-1 and t (there can only
            # one such event)
            found = Or(*[And(e > t-1, e <= t) for e in events])
            # If one was there, find out when it started
            start_time = Real(f"our_start_time_{n},{t}")
            for e in events:
                o.add(Implies(And(e > t-1, e <= t), start_time == e))


def make_solver(c: Config) -> Tuple[Solver, Variables]:
    v = Variables(c)
    o = Solver()
    o.add(v.C_tr > 0)
    o.add(v.C_su > 0)
    # We don't want time discretization to be too fine. Helps with constraints
    # to determine our_bytes since we can have only one event per timestep
    o.add(v.B >= c.C)

    phases(c, o, v)

    return (o, v)


def plot_model(m: ModelDict, c: Config):
    ax, fig = plt.subplots()
    args = {
        "linewidth": 8,
        "markersize": 0,
        "solid_capstyle": "butt"
    }
    for n in range(c.N):
        for s in range(c.S):
            # Plot training
            y = 3 * n + 0.8
            start = m[f"tr_{n},{s}"]
            end = start + m["C_tr"]
            plt.plot([start, end], [y, y], **args, color="black")

            for i in range(c.N-1):
                # Plot summing
                y = 3 * n + 1 + i / (c.N - 1)
                start = m[f"su_{n},{s},{i}"]
                end_1 = start + m[f"su_tx_{n},{s},{i}"]
                end_2 = end_1 + m["C_tr"]
                plt.plot([start, end_1], [y, y], **args, color="red")
                plt.plot([end_1, end_2], [y, y], **args, color="tomato")

                # Plot broadcast
                y = 3 * n + 2 + i / (c.N - 1)
                start = m[f"br_{n},{s},{i}"]
                end = start + m[f"su_tx_{n},{s},{i}"]
                plt.plot([start, end], [y, y], **args, color="blue")
    plt.show()


if __name__ == "__main__":
    c = Config()
    o, v = make_solver(c)
    sat = o.check()
    print(sat)
    if str(sat) == "sat":
        m = model_to_dict(o.model())
        plot_model(m, c)
