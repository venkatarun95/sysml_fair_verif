from config import Config
from my_solver import MySolver
from utils import ModelDict, model_to_dict
from variables import Variables

import matplotlib.pyplot as plt
from typing import Tuple
from z3 import And, If, Implies, Not, Or

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


def phases(c: Config, o: MySolver, v: Variables):
    ''' Constraints for when the various phases happen for each node '''
    for n in range(c.N):
        pre = (n - 1) % c.N
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

            # # For now, all except one transmit times are equal
            # for i in range(c.N):
            #     if n == 0:
            #         o.add(v.su_tx[n][s][i] == 2)
            #         o.add(v.br_tx[n][s][i] == 2)
            #     else:
            #         o.add(v.su_tx[n][s][i] == 1)
            #         o.add(v.br_tx[n][s][i] == 1)


def tx_times(c: Config, o: MySolver, v: Variables):
    ''' Figure out transmit times based on fair sharing policy '''
    for n in range(c.N):
        # First, events are monotonic. For simplicity, two events cannot happen
        # at the same time. They *can* happen arbitrarily close to each other
        # though
        for s in range(1, c.E):
            o.add(v.events[n][s-1] < v.events[n][s])

        # Event type, num flows and outstanding must be within range
        for e in range(c.E):
            o.add(v.event_type[n][e] <= 2 * c.F)
            o.add(0 <= v.event_type[n][e])

            o.add(v.num_flows[n][e] <= c.F)
            o.add(0 <= v.num_flows[n][e])

            for f in range(c.F):
                o.add(0 <= v.outstanding[n][f][e])

        # Some initial conditions
        for f in range(c.F):
            o.add(v.outstanding[n][f][0] == 0)

        # This is the list of flow events that occured on this link. We have
        # this for convenience. Each event is a time instant where a flow
        # started or ended
        our_starts = []
        our_ends = []
        for s in range(c.S):
            our_starts.extend(v.su[n][s])
            our_starts.extend(v.br[n][s])
            for i in range(c.N):
                our_ends.append(v.su[n][s][i] + v.su_tx[n][s][i])
                our_ends.append(v.br[n][s][i] + v.br_tx[n][s][i])

        for (start, end) in zip(our_starts, our_ends):
            # Make sure each of our_events are represented in v.events
            o.add(Or(*[start == event for event in v.events[n]]))
            o.add(Or(*[end == event for event in v.events[n]]))

            # Further, ensure that `v.outstanding` and `v.event_type` are
            # correct when our event starts/ends
            for (e, event) in enumerate(v.events[n]):
                o.add(Implies(
                    event == start,
                    And(v.outstanding[n][0][e] == v.B,
                        v.event_type[n][e] == 0)))

                o.add(Implies(
                    event == end,
                    And(v.outstanding[n][0][e] == 0.0,
                        v.event_type[n][e] == 1)))

        # Start and stop conditions for other flows
        for f in range(c.F):
            # Ensure a flow starts only after the previous sub-flow ended
            for e in range(1, c.E):
                o.add(Implies(v.event_type[n][e] == 2 * f,
                              v.outstanding[n][f][e-1] == 0))
            # Flow only ends when it has transmitted all bytes
            for e in range(1, c.E):
                o.add(Implies(v.event_type[n][e] == 2 * f + 1,
                              v.outstanding[n][f][e] == 0))

        # Continuation equation for `v.outstanding`
        for e in range(1, c.E):
            # First compute how much each flow should decrease its
            # outstanding. Need to do it in a loop to avoid multiplications
            # with `v.num_flows`
            decrease = o.Real(f"outstanding_decrease_{n},{e}")
            for f in range(1, c.F + 1):
                o.add(Implies(v.num_flows[n][e] == f, decrease == c.C / f))
            o.add(Implies(v.num_flows[n][e] == 0, decrease == 0))

            # If the event does not correspond to this flow, it transmits bytes
            # depending on number of flows and transmission policy
            for f in range(c.F):
                new = v.outstanding[n][f][e] - decrease
                o.add(Implies(
                    And(v.event_type[n][e] != 2 * f,
                        v.event_type[n][e] != 2 * f + 1),
                    v.outstanding[n][f][e] == If(new >= 0, new, 0)))

        # Calculate `v.num_flows`
        o.add(v.num_flows[n][0] == 0)
        for e in range(1, c.E):
            start = Or(*[v.event_type[n][e] == 2 * f for f in range(c.F)])
            end = Or(*[v.event_type[n][e] == 2 * f + 1 for f in range(c.F)])
            o.add(Implies(start, v.num_flows[n][e] == v.num_flows[n][e-1] + 1))
            o.add(Implies(end,   v.num_flows[n][e] == v.num_flows[n][e-1] - 1))
            o.add(Implies(Not(Or(start, end)),
                          v.num_flows[n][e] == v.num_flows[n][e-1]))


def make_solver(c: Config) -> Tuple[MySolver, Variables]:
    o = MySolver()
    o.set(unsat_core=c.unsat_core)
    v = Variables(c, o)
    o.add(v.C_tr > 0)
    o.add(v.C_su > 0)
    # We don't want time discretization to be too fine. Helps with constraints
    # to determine our_bytes since we can have only one event per timestep
    # o.add(v.B >= c.C)

    phases(c, o, v)
    tx_times(c, o, v)

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
    print(f"Num constraints = {o.num_constraints}, "
          f"num variables = {len(o.variables)}")
    sat = o.check()
    print(sat)
    if str(sat) == "sat":
        m = model_to_dict(o.model())
        print(m)
        plot_model(m, c)
