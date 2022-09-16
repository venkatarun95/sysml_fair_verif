import csv
from pyz3_utils import IfStmt, Max, Min, MySolver, Variables, run_query
from typing import List, Optional, Tuple
import z3
z3.set_option("parallel.threads.max", 4)
z3.set_option("parallel.enable", "true")
from z3 import And, If, Implies, Not, Or

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
    neighbors: List[Tuple[Tuple[int, int], Tuple[int, int]]] = \
        [((0, 0), (1, 0)), ((1, 1), (2, 1))]

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
        # Progresses in rounds. This is the round we are currently in
        self.round = s.Int(f"{name}_round")


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


def tick(t2_id: int, c: Config, s: MySolver,
         v: GlobalVars):
    ''' Constrain how things evolve in time '''
    assert t2_id > 0
    t1, t2 = v.times[t2_id-1], v.times[t2_id]
    delta_t = t2.time - t1.time
    s.add(delta_t >= 0)
    for r in range(c.num_rings):
        for n in range(c.num_nodes_per_ring):
            nt1 = t1.rings[r].nodes[n]
            nt2 = t2.rings[r].nodes[n]

            # Are we done with this round?
            new_round = And(nt1.backprop == v.tot_backprop[r],
                            nt1.sum_sent == v.tot_size[r],
                            nt1.broad_sent == v.tot_size[r])

            IfStmt(new_round,
                   # Increment round
                   nt2.round == nt1.round + 1,
                   # Do backprop
                   nt2.backprop == Min(s, delta_t, v.tot_backprop[r]),
                   # No time has elapsed
                   t2.time == t1.time,
            ).Else(
                # Same old, same old
                nt2.round == nt1.round,
                # Do backprop
                nt2.backprop ==
                  Min(s, nt1.backprop + delta_t,
                      v.tot_backprop[r])).add_to_solver(s)

            # How much do we send for summing?
            ## Cap due to backprop, we'll let z3 pick if backprop is not done
            sum_backprop_cap = s.Real(f"backprop_cap{t2_id},{r},{n}")
            ### No caps if backprop is done
            s.add(Implies(nt2.backprop >= v.tot_backprop[r],
                  sum_backprop_cap == v.tot_size[r]))
            s.add(sum_backprop_cap >= 0)

            ## Cap because of whether or not we received from the previous node
            node_round_diff = t1.rings[r].nodes[n-1].round - nt1.round
            sum_recv_cap = v.tot_size[r] / c.num_nodes_per_ring\
                + If(node_round_diff > 0, v.tot_size[r],
                     If(node_round_diff < 0, 0,
                        t1.rings[r].nodes[n-1].sum_sent))
            broad_recv_cap = v.tot_size[r] / c.num_nodes_per_ring\
                + If(node_round_diff > 0, v.tot_size[r],
                     If(node_round_diff < 0, 0,
                        t1.rings[r].nodes[n-1].broad_sent))

            # Update ready_to_send, sum_sent and broad_sent
            IfStmt(new_round,
                   # No summing or broadcast has started. If z3 wants to sum,
                   # it can always use one extra timestamp
                   nt2.sum_sent == 0,
                   nt2.broad_sent == 0,
                   nt2.ready_to_send == 0,
            ).Elif(
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
                    Min(s, broad_recv_cap, v.tot_size[r])
                    - nt1.broad_sent),
                nt2.broad_sent == nt1.broad_sent + nt2.tot_data_sent,
                nt2.sum_sent == nt1.sum_sent
            ).add_to_solver(s)

            # Decide tot_data_sent based on ready_to_send
            assert nt1 != nt2
            if nt2.neighbor is None:
                s.add(nt2.tot_data_sent == Min(s, nt2.ready_to_send, delta_t))
                # This ensures that the timesteps are such that link
                # utilization is 100% or 0%. Sure we could simplify above due
                # to this constraint, but meh. Let's keep the flexibility to
                # enable/disable this for now
                s.add(Or(nt2.ready_to_send >= delta_t,
                         nt2.ready_to_send == 0))
            else:
                # Only one of the neighbors needs to do this
                assert r != nt2.neighbor[0]
                if r > nt2.neighbor[0]: # or (r == nt2.neighbor[0] and n > nt2.neighbor[1]):                    print(t2_id, (r, n), nt2.neighbor, )
                    oth1 = t1.rings[nt2.neighbor[0]].nodes[nt2.neighbor[1]]
                    oth2 = t2.rings[nt2.neighbor[0]].nodes[nt2.neighbor[1]]

                    # Should we dominate in TCP or should `other` dominate?
                    dom_cond = nt1.sum_sent + nt1.broad_sent\
                        > oth1.sum_sent + oth1.broad_sent
                    undom_cond = nt1.sum_sent + nt1.broad_sent\
                        < oth1.sum_sent + oth1.broad_sent
                    # Neither should dominate the other
                    eq_cond = nt1.sum_sent + nt1.broad_sent\
                        == oth1.sum_sent + oth1.broad_sent
                    IfStmt(
                        nt2.ready_to_send + oth2.ready_to_send < delta_t,
                        nt2.tot_data_sent == nt2.ready_to_send,
                        oth2.tot_data_sent == oth2.ready_to_send
                    ).Else(
                        nt2.tot_data_sent + oth2.tot_data_sent == delta_t,
                        nt2.tot_data_sent <= nt2.ready_to_send,
                        oth2.tot_data_sent <= oth2.ready_to_send,
                        Implies(dom_cond,
                                nt2.tot_data_sent > oth2.tot_data_sent),
                        Implies(undom_cond,
                                nt2.tot_data_sent < oth2.tot_data_sent),
                        Implies(eq_cond,
                                nt2.tot_data_sent == oth2.tot_data_sent)
                    ).add_to_solver(s)

                    # This ensures that the timesteps are such that link
                    # utilization is 100% or 0%
                    s.add(Or(
                        oth2.ready_to_send + nt2.ready_to_send >= delta_t,
                        oth2.ready_to_send + nt2.ready_to_send == 0))
                    # We can additionally enforce that each sender can
                    # individually fill up the time. Z3 will be forced to pick
                    # smaller time gaps if needed
                    s.add(Or(nt2.ready_to_send >= delta_t,
                             nt2.ready_to_send == 0))
                    s.add(Or(oth2.ready_to_send >= delta_t,
                             oth2.ready_to_send == 0))


def make_solver(c: Config, s: MySolver) -> GlobalVars:
    v = GlobalVars(c, s)

    # Tell nodes about their neighbors. They are indexed by (ring_id, node_id)
    for t in range(c.num_timesteps):
        for ((r1, n1), (r2, n2)) in c.neighbors:
            v.times[t].rings[r1].nodes[n1].neighbor = (r2, n2)
            v.times[t].rings[r2].nodes[n2].neighbor = (r1, n1)

    # Do all the ticks
    for t in range(1, c.num_timesteps):
        tick(t, c, s, v)

    # Initial conditions
    ## Starting time is arbitrary. Set it to something nice
    s.add(v.times[0].time == 0)
    ## One starting round per ring is arbitrary. Let it be nice
    for r in range(c.num_rings):
        s.add(v.times[t].rings[r].nodes[0].round == 0)

    for r in range(c.num_rings):
        for nid in range(c.num_nodes_per_ring):
            n = v.times[0].rings[r].nodes[nid]
            # Broadcast can only start when sum and backprop is finished
            s.add(Implies(n.broad_sent > 0,
                          And(n.sum_sent == v.tot_size[r],
                              n.backprop == v.tot_backprop[r])))

            # We should not have sent more than we had received
            s.add(n.sum_sent <= v.times[0].rings[r].nodes[nid-1].sum_sent
                  + v.tot_size[r] / c.num_nodes_per_ring)
            s.add(n.broad_sent <= v.times[0].rings[r].nodes[nid-1].broad_sent
                  + v.tot_size[r] / c.num_nodes_per_ring)

            # If the rounds are different:
            for nid2 in range(c.num_nodes_per_ring):
                n2 = v.times[0].rings[r].nodes[nid2]
                s.add(Implies(
                    n.round != n2.round,
                    # The difference can be at most 1. We'll take each case at
                    # a time and assert that we cannot progress farther than
                    # doing our own backprop and sending our sum
                    Or(And(n2.round == n.round + 1,
                           n2.sum_sent <= v.tot_size[r] / c.num_nodes_per_ring),
                       And(n2.round == n.round - 1,
                           n.sum_sent <= v.tot_size[r] / c.num_nodes_per_ring),
                )))

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
                s.add(n.tot_data_sent >= 0)
    return v


def plot(c: Config, v: Variables):
    print("Format: round,backprop,sum_sent,broad_sent")
    print(f"tot_backprop: {[float(x) for x in v.tot_backprop]}, "
          f"tot_size: {[float(x) for x in v.tot_size]}")
    def pprint(n: Node, name: str) -> float:
        if name in n.__dict__:
            return float(n.__dict__[name])
        return -1.0
    writer = csv.writer(open("output.csv", "w"))

    for t in range(c.num_timesteps):
        line = f"{float(v.times[t].time):4.2} "
        row = [float(v.times[t].time)]
        for r in range(c.num_rings):
            # line += '\t: '.join(
            #     ["{:.2},{:.2},{:.2}".format(
            #         float(n.backprop),
            #         float(n.sum_sent),
            #         float(n.broad_sent))
            #         # pprint(n, "tot_data_sent"),
            #         # pprint(n, "ready_to_send"))
            #      for n in v.times[t].rings[r].nodes])
            line += ' : '.join(
                [f"{int(n.round)},"
                 f"{float(n.backprop):4.2},"
                 f"{float(n.sum_sent):4.2},"
                 f"{float(n.broad_sent):4.2},"
                 f"{pprint(n, 'tot_data_sent'):4.2},"
                 f"{pprint(n, 'ready_to_send'):4.2}"
                 for n in v.times[t].rings[r].nodes])

            for n in v.times[t].rings[r].nodes:
                row.extend([
                    int(n.round),
                    float(n.backprop),
                    float(n.sum_sent),
                    float(n.broad_sent),
                    pprint(n, "tot_data_sent"),
                    pprint(n, "ready_to_send"),
                    "---"])
            row[-1] += ":---"
        print(line)

        writer.writerow(row)


def verify_sudarsanan_is_genius(c: Config):
    s = MySolver()
    v = make_solver(c, s)

    if False:
        # Just so the example has nice values
        for r in range(c.num_rings):
            s.add(v.tot_size[r] <= 5)
            s.add(v.tot_backprop[r] <= 5)

    # Let's ask the big question
    cond = []
    for ((r1, n1), (r2, n2)) in c.neighbors:
        n1i = v.times[0].rings[r1].nodes[n1]
        n1f = v.times[-1].rings[r1].nodes[n1]
        n2i = v.times[0].rings[r2].nodes[n2]
        n2f = v.times[-1].rings[r2].nodes[n2]

        # Overlap in communication increases. This is bad
        cond.append(And(
            n1i.sum_sent + n1i.broad_sent > n2i.sum_sent + n2i.broad_sent,
            sum([v.times[t].rings[r1].nodes[n1].tot_data_sent for t in range(1, c.num_timesteps)]) <
            sum([v.times[t].rings[r2].nodes[n2].tot_data_sent for t in range(1, c.num_timesteps)])
        ))
        cond.append(And(
            n1i.sum_sent + n1i.broad_sent < n2i.sum_sent + n2i.broad_sent,
            sum([v.times[t].rings[r1].nodes[n1].tot_data_sent for t in range(1, c.num_timesteps)]) >
            sum([v.times[t].rings[r2].nodes[n2].tot_data_sent for t in range(1, c.num_timesteps)])
        ))

        # There is enough space for communication to fit inside computation
        s.add(2 * v.tot_size[r1] > v.tot_backprop[r2])
        s.add(2 * v.tot_size[r2] > v.tot_backprop[r1])

    s.add(Or(*cond))

    res = run_query(c, s, v, timeout=3600)
    print(res.satisfiable)

    if res.satisfiable == "sat":
        plot(res.c, res.v)

if __name__ == "__main__":
    configs = [
        {
            "num_rings": 2,
            "num_nodes_per_ring": 3,
            "neighbors": [((0, 0), (1, 0))],
            "num_timesteps": 5,
        },
        {
            "num_rings": 2,
            "num_nodes_per_ring": 4,
            "neighbors": [((0, 0), (1, 0))],
            "num_timesteps": 5,
        },
        {
            "num_rings": 2,
            "num_nodes_per_ring": 4,
            "neighbors": [((0, 0), (1, 0)), ((0, 2), (1, 2))],
            "num_timesteps": 5,
        },
        {
            "num_rings": 3,
            "num_nodes_per_ring": 3,
            "neighbors": [((0, 0), (1, 0)), ((1, 1), (2, 1)), ((2, 2), (0, 2))],
            "num_timesteps": 10,
        },
        {
            "num_rings": 3,
            "num_nodes_per_ring": 3,
            "neighbors": [((0, 0), (1, 0)), ((1, 1), (2, 1))],
            "num_timesteps": 10,
        },
        {
            "num_rings": 4,
            "num_nodes_per_ring": 3,
            "neighbors": [((0, 0), (1, 0)), ((1, 1), (2, 1)), ((2, 2), (3, 2))],
            "num_timesteps": 10,
        }
    ]

    for conf in configs:
        print(conf)
        c = Config()
        c.num_rings = conf["num_rings"]
        c.num_nodes_per_ring = conf["num_nodes_per_ring"]
        c.neighbors = conf["neighbors"]
        c.num_timesteps = conf["num_timesteps"]
        verify_sudarsanan_is_genius(c)

    # c.num_nodes_per_ring = 3
    # c.neighbors = [((0, 0), (1, 0)), ((1, 1), (2, 1)), ((2, 2), (0, 2))]
    # c.num_timesteps = 5
    # c.unsat_core = False
