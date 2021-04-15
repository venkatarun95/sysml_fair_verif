from utils import model_to_dict

import matplotlib.pyplot as plt
from z3 import And, If, Implies, Not, Real, Solver

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

# Number of nodes
N = 4
# Number of iterations (S stands for steps)
S = 5
# Number of timesteps we compute cross-traffic over
T = 10

C_tr = Real("C_tr")
C_su = Real("C_su")
B = Real("B")
# Link rate. Units are arbitrary
C = 1

# The time at which the s^th iteration's training starts
tr = [[Real(f"tr_{n},{s}") for s in range(S)] for n in range(N)]
# The time at which we start trasmitting the i^th block when summing in the
# s^th iteration. Note, the last sum event is virtual: it doesn't really
# happen, it merely indicates when the first broadcast started
su = [[[Real(f"su_{n},{s},{i}") for i in range(N)] for s in range(S)] for
      n in range(N)]
# The time at which we start trasmitting the n^th block when broadcasting in
# the s^th iteration. Similar to su, the last event is virtual
br = [[[Real(f"br_{n},{s},{i}") for i in range(N)] for s in range(S)] for
      n in range(N)]

# Transmit time for transmitting during summing
su_tx = [[[Real(f"su_tx_{n},{s},{i}") for i in range(N)] for s in range(S)] for
         n in range(N)]
# Transmit time for transmitting during broadcast
br_tx = [[[Real(f"br_tx_{n},{s},{i}") for i in range(N)] for s in range(S)] for
         n in range(N)]

# Number of our own bytes waiting to be transmitted at the given link
our_bytes = [[Real("our_bytes_{n},{t}") for t in range(T)] for n in range(N)]
# Number of competing bytes waiting to be transmitted at the given link
their_bytes = [[Real("their_bytes_{n},{t}") for t in range(T)] for n in range(N)]
# Number of bytes competing flows sent
their_arr = [[Real("their_arr_{n},{t}") for t in range(T)] for n in range(N)]

o = Solver()
o.add(C_tr > 0)
o.add(C_su > 0)
# We don't want time discretization to be too fine. Helps with constraints to
# determine our_bytes
o.add(B >= C)

for n in range(N):
    pre = (n - 1) % N
    nex = (n + 1) % N
    for s in range(S):
        if s == 0:
            o.add(tr[n][s] == 0)
        else:
            # Note, the last broadcast is a pseudo-event
            o.add(tr[n][s] == br[pre][s-1][-1])

        # First sum transmit starts after training is done
        o.add(su[n][s][0] == tr[n][s] + C_tr)

        # First broadcast happens when the last sum transmit starts because
        # the last sum transmit is not really needed. It is a pseudo-event
        o.add(br[n][s][0] == su[n][s][-1])

        for i in range(1, N):
            # Summing phase:
            # The next block is ready to be sent
            ready = su[pre][s][i-1] + su_tx[pre][s][i-1] + C_su
            # Tx link is clear for sending
            clear = su[n][s][i-1] + su_tx[n][s][i-1]
            # When are we sending the next block
            o.add(su[n][s][i] == If(ready > clear, ready, clear))

            # Broadcast phase: analogous to summing phase
            ready = br[pre][s][i-1] + br_tx[pre][s][i-1]
            clear = br[n][s][i-1] + br_tx[n][s][i-1]
            o.add(br[n][s][i] == If(ready > clear, ready, clear))

        # For now, all transmit times are equal
        for i in range(N):
            if n == 0:
                o.add(su_tx[n][s][i] == 2)
                o.add(br_tx[n][s][i] == 2)
            else:
                o.add(su_tx[n][s][i] == 1)
                o.add(br_tx[n][s][i] == 1)

for n in range(N):
    s.add(our_bytes[n][0] == 0)
    s.add(their_bytes[n][0] == their_arr[n][0])

    # This is the list of events that occured on this link. We have this for
    # convenience
    events = []
    for s in range(S):
        events.extend(su[n][s][:-1])
        events.extend(br[n][s][:-1])

    for t in range(1, T):
        # Find the latest event between t-1 and t.

        # Temporary variables to compute max
        max_tmp = [Real(f"max_tmp_{n},{t},{j}") for j in range(len(events))]
        o.add(max_tmp[0] == events[0])
        for j in range(1, len(events)):
            o.add(max_tmp[j] == If(
                And(events[j] > i-1, events[j] <= i),
                If(events[j] >= max_tmp[j-1], events[j], max_tmp[j-1]),
                max_tmp[j-1]))

        # Was there an event in this time interval?
        found = And(max_tmp[-1] > i-1, max_tmp[-1] <= i)

        # If such an event was found, then our_bytes is determined by that
        drained = B - (t - max_tmp[-1]) * C
        s.add(Implies(found, our_bytes[n][t] == If(drained >= 0, drained, 0)))
        # If not, we continue draining what was there before
        drained = B - C
        s.add(Implies(Not(found), our_bytes[n][t] == If(drained >= 0, drained,
                                                        0)))

sat = o.check()
print(sat)
if str(sat) == "sat":
    m = model_to_dict(o.model())
    ax, fig = plt.subplots()
    args = {
        "linewidth": 8,
        "markersize": 0,
        "solid_capstyle": "butt"
    }
    for n in range(N):
        for s in range(S):
            # Plot training
            y = 3 * n + 0.8
            start = m[f"tr_{n},{s}"]
            end = start + m["C_tr"]
            plt.plot([start, end], [y, y], **args, color="black")

            for i in range(N-1):
                # Plot summing
                y = 3 * n + 1 + i / (N - 1)
                start = m[f"su_{n},{s},{i}"]
                end_1 = start + m[f"su_tx_{n},{s},{i}"]
                end_2 = end_1 + m["C_tr"]
                plt.plot([start, end_1], [y, y], **args, color="red")
                plt.plot([end_1, end_2], [y, y], **args, color="tomato")

                # Plot broadcast
                y = 3 * n + 2 + i / (N - 1)
                start = m[f"br_{n},{s},{i}"]
                end = start + m[f"su_tx_{n},{s},{i}"]
                plt.plot([start, end], [y, y], **args, color="blue")
    plt.show()
