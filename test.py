from continuous_model import GlobalVars, Config, make_solver
from pyz3_utils import MySolver, run_query
import unittest
from z3 import And, Or

class TestContinuousModel(unittest.TestCase):
    def test_exists(self):
        '''If we don't add any fancy constraints, there better exist a
        solution!'''
        c = Config()
        s = MySolver()
        v = make_solver(c, s)
        res = run_query(c, s, v, timeout=60)
        self.assertEqual(res.satisfiable, "sat")


    def test_monotone(self):
        c = Config()
        s = MySolver()
        v = make_solver(c, s)

        # Can any of the variables be decreasing?
        cond = []
        for t in range(1, c.num_timesteps):
            for r in range(c.num_rings):
                for n in range(c.num_nodes_per_ring):
                    nt1 = v.times[t-1].rings[r].nodes[n]
                    nt2 = v.times[t].rings[r].nodes[n]
                    eq = nt1.round == nt2.round
                    cond.append(nt1.round > nt2.round)
                    cond.append(And(nt1.backprop > nt2.backprop, eq))
                    cond.append(And(nt1.sum_sent > nt2.sum_sent, eq))
                    cond.append(And(nt1.broad_sent > nt2.broad_sent, eq))
                    cond.append(And(nt2.ready_to_send < 0, eq))
                    cond.append(And(nt2.tot_data_sent < 0, eq))
        s.add(Or(*cond))
        res = run_query(c, s, v, timeout=60)
        if res.satisfiable == "sat":
            from continuous_model import plot
            plot(res.c, res.v)
        self.assertEqual(res.satisfiable, "unsat")

    def test_operation_order(self):
        c = Config()
        s = MySolver()
        v = make_solver(c, s)

        cond = []
        for t in range(1, c.num_timesteps):
            for r in range(c.num_rings):
                for n in range(c.num_nodes_per_ring):
                    nt1 = v.times[t-1].rings[r].nodes[n]
                    nt2 = v.times[t].rings[r].nodes[n]

                    # Broadcast can only start if summing is over
                    cond.append(
                        And(nt2.broad_sent > nt1.broad_sent,
                            nt1.sum_sent != v.tot_size[r]))

                    # We cannot send more data than we have
                    cond.append(nt2.tot_data_sent > nt2.ready_to_send)

                    # We ought not be sending more sum data than we have
                    cond.append(And(
                        nt2.sum_sent > v.times[t-1].rings[r].nodes[n-1].sum_sent
                        + v.tot_size[r] / c.num_nodes_per_ring,
                        nt2.round >= v.times[t-1].rings[r].nodes[n-1].round))
                    cond.append(And(
                        nt2.broad_sent
                        > v.times[t-1].rings[r].nodes[n-1].broad_sent
                        + v.tot_size[r] / c.num_nodes_per_ring,
                        nt2.round >= v.times[t-1].rings[r].nodes[n-1].round))

        s.add(Or(*cond))

        # Just so the example has nice values
        # for r in range(c.num_rings):
        #     s.add(v.tot_size[r] <= 5)
        #     s.add(v.tot_backprop[r] <= 5)

        res = run_query(c, s, v, timeout=60)
        if res.satisfiable == "sat":
            from continuous_model import plot
            plot(res.c, res.v)
        self.assertEqual(res.satisfiable, "unsat")

if __name__ == '__main__':
    unittest.main()
