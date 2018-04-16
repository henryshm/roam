import pickle
import time

import numpy as np

from macrel import graphs
from macrel import vast11data as vast


INTERVAL_SEC = 900.0


N = len(vast.NODES)

SERVICES = {
    1: "Mux",
    17: "Quote",
    21: "FTP",
    22: "SSH",
    23: "Telnet",
    25: "SMTP",
    53: "DNS",
    80: "HTTP",
    88: "Kerberos",
    123: "NTP",
    135: "DCE",
    139: "NETBIOS",
    255: "Reserved",
    389: "LDAP",
    443: "HTTPS",
    445: "Microsoft-DS",
    464: "kpasswd",
    481: "ph",
}

tally_map = {port: graphs.ConnectionTally(N) for port in SERVICES.keys()}
other_tally = graphs.ConnectionTally(N)

def add_tally(event):
	src = vast.NODE_BY_IP.get(event.source_ip)
	dest = vast.NODE_BY_IP.get(event.dest_ip)
	if not src or not dest: return
	port = event.dest_port
	tally = tally_map.get(port, other_tally)
	tally.connect(src.id, dest.id, event.conn_built)

start_times = []
snapshots = {port: [] for port in SERVICES}
snapshots["other"] = []
snapshots["all"] = []

def take_snapshot():
	start_times.append(start_time)
	all_totals = None
	for port, tally in tally_map.items():
		totals = tally.to_sparse_matrix()
		snapshots[port].append(totals)
		if all_totals is None:
			all_totals = totals.copy()
		else:
			all_totals += totals

	snapshots["other"].append(other_tally.to_sparse_matrix())
	snapshots["all"].append(all_totals)
	

parser = vast.FWEventParser()
events = parser.parse_all_fw_events()

first_event = next(events)
start_time = first_event.time
tt = time.gmtime(start_time)
start_time -= (tt.tm_min * 60)  # align to hour
end_time = start_time + INTERVAL_SEC

t = first_event.time
while t > end_time:
	take_snapshot()
	start_time = end_time
	end_time = start_time + INTERVAL_SEC
add_tally(first_event)

for event in events:
	t = event.time
	while t > end_time:
		take_snapshot()
		start_time = end_time
		end_time = start_time + INTERVAL_SEC
	add_tally(event)

take_snapshot()

data = dict(
	services = SERVICES,
	start_times = np.asarray(start_times),
	snapshots = snapshots,
)

pickle.dump(data, open("vast11-connections-by-port.pickle", "wb"))

