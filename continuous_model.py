from pyz3_utils import IfStmt, Max, Min, MySolver, Variables, run_query
from typing import Optional, Tuple
from z3 import And, Implies, Not

'''Each node starts doing backprop. At some arbitrary point in between, and
certainly by the end of backprop, it will have the ability to send its (1/n)^th
share of the weights. Likewise when it receives somebody else's share, it can
only add its share depending on whether its backprop is finished. We model all
of this by saying that before backprop is finished, a node can arbitrarily
rate-limit what it sends. After backprop is finished, it must send.

The broadcast phase begins only after a node has finished
summing.

'''

class Config(Variables):
    num_timesteps: int = 5
    num_rings: int = 3
    num_nodes_per_ring: int = 3

class Node(Variables):
    # r and n of my the node with which we share the uplink bottleneck
    neighbor: Optional[Tuple[int, int]]

    def __init__(self, c: Config, s: MySolver, name: str,
                 neighbor: Optional[Tuple[int, int]] = None):
        # No neighbor by default. Can be filled in by invoker
        self.neighbor = neighbor
        # Progress of the backprop step
        self.backprop = s.Real(f"{name}_backprop")
        # Amount of data sent for summing step
        self.sum_sent = s.Real(f"{name}_sumSent")
        # Amount of data sent for the broadcast step
        self.broad_sent = s.Real(f"{name}_broadSent")
        # Amount of data (sum or broadcast) the app was ready to send in the
        # time interval leading up to this
        self.ready_to_send = s.Real(f"{name}_readyToSend")
        # The amount of data it actually sent based on bandwidth available
        # (sum_sent + broad_sent) in the time interval leading up to this.
        self.tot_data_sent = s.Real(f"{name}_totDataSent")


class Ring(Variables):
    def __init__(self, c: Config, s: MySolver, name: str):
        self.nodes = [Node(c, s, f"{name}_node{n}")
                      for n in range(c.num_nodes_per_ring)]

class Timestep(Variables):
    def __init__(self, c: Config, s: MySolver, name: str):
        self.time = s.Real(f"{name}_time")
        self.rings = [Ring(c, s, f"{name}_ring{r}") for r in range(c.num_rings)]

class GlobalVars(Variables):
    def __init__(self, c: Config, s: MySolver):
        # Total amount of computation to be done for backprop
        self.tot_backprop = [s.Real(f"totBackprop{r}") for r in range(c.num_rings)]
        # Total size of the model
        self.tot_size = [s.Real(f"totSize{r}") for r in range(c.num_rings)]

        self.times = [Timestep(c, s, f"time{t}") for t in range(c.num_timesteps)]


def tick(t1: Timestep, t2: Timestep, t2_id: int, c: Config, s: MySolver,
         v: GlobalVars):
    ''' Constrain how things evolve in time '''
    # Assumption is t2 is at a later time than t1
    delta_t = t2.time - t1.time
    s.add(delta_t > 0)
    for r in range(c.num_rings):
        for n in range(c.num_nodes_per_ring):
            nt1 = t1.rings[r].nodes[n]
            nt2 = t2.rings[r].nodes[n]

            # Do backprop if necessary
            s.add(nt2.backprop ==
                  Min(s, nt1.backprop + delta_t,
                      v.tot_backprop[r]))

            # How much do we send for summing?
            ## Cap due to backprop, we'll let z3 pick if backprop is not done
            sum_backprop_cap = s.Real(f"backprop_cap{t2_id},{r},{n}")
            ### No caps if backprop is not done
            s.add(Implies(nt2.backprop >= v.tot_backprop[r],
                  sum_backprop_cap == v.tot_size[r]))
            s.add(sum_backprop_cap >= 0)

            ## Cap because of whether or not we received from the previous node
            sum_recv_cap = Max(s, v.tot_size[r] / c.num_nodes_per_ring,
                           t1.rings[r].nodes[n-1].sum_sent)

            # Update ready_to_send, sum_sent and broad_sent
            IfStmt(
                t1.rings[r].nodes[n].sum_sent < v.tot_size[r],
                # ^ We are still summing
                nt2.ready_to_send ==
                Max(s, 0,
                    Min(s, sum_recv_cap, sum_backprop_cap, v.tot_size[r])
                    - nt1.sum_sent),
                nt2.sum_sent == nt1.sum_sent + nt2.tot_data_sent,
                nt2.broad_sent == nt1.broad_sent
            ).Else(
                # We are broadcasting
                nt2.ready_to_send ==
                Max(s, 0,
                    Min(s, v.tot_size[r],
                         t1.rings[r].nodes[n-1].broad_sent
                        + v.tot_size[r] / c.num_nodes_per_ring)
                    - nt1.broad_sent),
                nt2.broad_sent == nt1.broad_sent + nt2.tot_data_sent,
                nt2.sum_sent == nt1.sum_sent
            ).add_to_solver(s)

            # Decide tot_data_sent based on ready_to_send
            assert nt1 != nt2
            if nt2.neighbor is None:
                s.add(nt2.tot_data_sent == Min(s, nt2.ready_to_send, delta_t))
            else:
                # Only one of the neighbors needs to do this
                if r > nt2.neighbor[0] or n > nt2.neighbor[1]:
                    other = t1.rings[nt2.neighbor[0]].nodes[nt2.neighbor[1]]
                    # Should we dominate in TCP or should `other` dominate?
                    dom_cond = nt1.sum_sent + nt1.broad_sent\
                        > other.sum_sent + other.broad_sent
                    # Neither should dominate the other
                    eq_cond = nt1.sum_sent + nt1.broad_sent\
                        == other.sum_sent + other.broad_sent
                    IfStmt(
                        nt2.ready_to_send + other.ready_to_send < delta_t,
                        nt2.tot_data_sent == nt2.ready_to_send,
                        other.tot_data_sent == other.ready_to_send
                    ).Else(
                        nt2.tot_data_sent + other.tot_data_sent == delta_t,
                        Implies(dom_cond,
                                nt2.tot_data_sent > other.tot_data_sent),
                        Implies(And(Not(dom_cond), Not(eq_cond)),
                                nt2.tot_data_sent < other.tot_data_sent),
                        Implies(eq_cond,
                                nt2.tot_data_sent == other.tot_data_sent)
                    ).add_to_solver(s)


def plot(c: Config, v: Variables):
    print("Format: backprop,sum_sent,broad_sent")
    print(f"tot_backprop: {[float(x) for x in v.tot_backprop]}, "
          f"tot_size: {[float(x) for x in v.tot_size]}")
    for t in range(c.num_timesteps):
        line = f"{float(v.times[t].time):.2} "
        for r in range(c.num_rings):
            line += '\t: '.join(
                ["{:.2},{:.2},{:.2}".format(
                    float(n.backprop),
                    float(n.sum_sent),
                    float(n.broad_sent))
                 for n in v.times[t].rings[r].nodes])
            line += " --- "
        print(line)

if __name__ == "__main__":
    c = Config()
    s = MySolver()
    v = GlobalVars(c, s)

    # Tell nodes about their neighbors. They are indexed by (ring_id, node_id)
    neighbors = [((0, 0), (1, 0)), ((1, 1), (2, 1))]
    for t in range(c.num_timesteps):
        for r in range(c.num_rings):
            for ((r1, n1), (r2, n2)) in neighbors:
                v.times[t].rings[r1].nodes[n1].neighbor = (r2, n2)
                v.times[t].rings[r2].nodes[n2].neighbor = (r1, n1)

    # Do all the ticks
    for t in range(1, c.num_timesteps):
        tick(v.times[t-1], v.times[t], t, c, s, v)

    # Initial conditions
    ## Starting time is arbitrary. Set it to something nice
    s.add(v.times[0].time == 0)
    for r in range(c.num_rings):
        for n in range(c.num_nodes_per_ring):
            n = v.times[0].rings[r].nodes[n]
            s.add(Implies(n.broad_sent > 0,
                          And(n.sum_sent == v.tot_size[r],
                              n.backprop == v.tot_backprop[r])))

    # Basic conditions that hold at all times
    for r in range(c.num_rings):
        s.add(v.tot_size[r] > 0)
        s.add(v.tot_backprop[r] > 0)
        for t in range(c.num_timesteps):
            for n in v.times[t].rings[r].nodes:
                s.add(n.backprop >= 0)
                s.add(n.backprop <= v.tot_backprop[r])
                s.add(n.sum_sent >= 0)
                s.add(n.broad_sent >= 0)
                s.add(n.sum_sent <= v.tot_size[r])
                s.add(n.broad_sent <= v.tot_size[r])

    # Just so the example has nice values
    for r in range(c.num_rings):
        s.add(v.tot_size[r] <= 5)
        s.add(v.tot_backprop[r] <= 5)

    # Just for kicks
    s.add(v.times[-1].time > 10)

    res = run_query(c, s, v, timeout=3600)
    print(res.satisfiable)

    if res.satisfiable == "sat":
        for k in res.model:
            if type(res.model[k]) is not bool:
                pass #print(k, ":::", res.model[k])
        # print(res.model)
        plot(res.c, res.v)

    # print(s.to_smt2())
