from config import Config
from my_solver import MySolver


class Variables:
    def __init__(self, c: Config, o: MySolver):
        self.C_tr = o.Real("C_tr")
        self.C_su = o.Real("C_su")
        self.B = o.Real("B")

        # The time at which the s^th iteration's training starts
        self.tr = [[o.Real(f"tr_{n},{s}") for s in range(c.S)] for n in
                   range(c.N)]
        # The time at which we start trasmitting the i^th block when summing in
        # the s^th iteration. Note, the last sum event is virtual: it doesn't
        # really happen, it merely indicates when the first broadcast started
        self.su = [[[o.Real(f"su_{n},{s},{i}") for i in range(c.N)] for s in
                    range(c.S)] for n in range(c.N)]
        # The time at which we start trasmitting the n^th block when
        # broadcasting in the s^th iteration. Similar to su, the last event is
        # virtual
        self.br = [[[o.Real(f"br_{n},{s},{i}") for i in range(c.N)] for s in
                    range(c.S)] for n in range(c.N)]

        # Transmit time for transmitting during summing
        self.su_tx = [[[o.Real(f"su_tx_{n},{s},{i}") for i in range(c.N)] for s
                       in range(c.S)] for n in range(c.N)]
        # Transmit time for transmitting during broadcast
        self.br_tx = [[[o.Real(f"br_tx_{n},{s},{i}") for i in range(c.N)] for s
                       in range(c.S)] for n in range(c.N)]

        # A sorted list of events that happen on each link
        self.events = [[o.Real(f"events_{n},{e}") for e in range(c.E)]
                       for n in range(c.N)]
        # The type of event. For flow i, event type 2*i denotes the start of
        # that flow while 2*i+1 denotes the end
        self.event_type = [[o.Int(f"event_type_{n},{e}") for e in range(c.E)]
                           for n in range(c.N)]
        # Number of flows active between events[e] and events[e+1]
        self.num_flows = [[o.Real(f"num_flows_{n},{e}") for e in range(c.E)]
                          for n in range(c.N)]
        # For each flow on each link, the number of outstanding bytes when the
        # t^th `event` happened. The 0^th flow is us
        self.outstanding = [[[o.Real(f"outstanding_{n},{f},{e}") for e in
                              range(c.E)] for f in range(c.F)] for n in
                            range(c.N)]
