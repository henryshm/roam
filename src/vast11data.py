import math
import time
from collections import namedtuple
import csv
from glob import glob

import numpy as np


# Structure for node metadata
NodeMeta = namedtuple("NodeMeta", ["id", "name", "ip", "kind"])

# kind should be one of:
class NodeKind:
    Server = "Server"
    External = "External"
    Workstation = "Workstation"
    Vulnerable_Workatation = "Vulnerable Workstation"
    Unknown = "Unknown"

def build_nodes():

    nodes = []

    # Function to add a new node to the list
    def add(ip, name, kind):
        id = len(nodes)
        nodes.append(NodeMeta(id, name, ip, kind))

    # Servers (incl. firewall and Interfaces)
    servers = [
        ("10.200.150.1", "Firewall"),
        ("192.168.1.1", "Vlan10"),
        ("192.168.2.1", "Vlan20"),
        ("172.20.1.1", "DMZ"),
        ("172.20.1.5", "ExtWeb"),
        ("192.168.1.2", "DHCP"),
        ("192.168.1.3", "HR DB"),
        ("192.168.1.4", "Ship DB"),
        ("192.168.1.5", "IntWeb"),
        ("192.168.1.6", "Mail"),
        ("192.168.1.7", "FileServ"),
        ("192.168.1.14", "DNS"),
        ("192.168.1.16", "Snort"),
        ("192.168.1.50", "FWLog"),
    ]
    for ip, name in servers:
        add(ip, name, NodeKind.Server)

    # unknown (server IP space)
    for i in list(range(8,14)) + [15] + list(range(17, 50)) + list(range(51, 256)):
        add(f"192.168.1.{i}", f"us{i}", kind=NodeKind.Unknown)

    # external machines (public internet)
    for i in range(2, 256):
        add(f"10.200.150.{i}", f"x{i}", kind=NodeKind.External)

    # internal workstations
    for i in range(10, 251):
        if 171 <= i <= 175:
            kind = NodeKind.Vulnerable_Workatation
        else:
            kind = NodeKind.Workstation
        add(f"192.168.2.{i}", f"w{i}", kind=kind)
        
    # unknown (workstation IP space)
    for i in list(range(2, 10)) + list(range(251, 256)):
        add(f"192.168.2.{i}", f"uw{i}", kind=NodeKind.Unknown)

    return nodes


# Global list of all nodes
NODES = build_nodes()
    
# lookup table into nodes by IP address
NODE_BY_IP = {node.ip: node for node in NODES}

# Colors for node classes
# Colorbrewer2 Qualitative/Dark/5
# http://colorbrewer2.org/#type=qualitative&scheme=Dark2&n=5
COLOR_BY_KIND = {
    NodeKind.Server: "#d95f02",
    NodeKind.External: "#e7298a",
    NodeKind.Workstation: "#7570b3",
    NodeKind.Vulnerable_Workatation: "#e6ab02",
    NodeKind.Unknown: "#66a61e"
}


# Structure for connection events
FWEvent = namedtuple("FWEvent", [
    "time",
    "source_ip",
    "dest_ip",
    "source_port",
    "dest_port",
    "conn_built",
    "conn_teardown"
])


class FWEventParser:

    # Identify Firewall Log files
    DATES = ["20110413", "20110414", "20110415"]
    FWFILES = [fname
               for date in DATES
               for fname in glob(f"/Users/dbeach/Desktop/MiniChallenge2 Core Data/{date}/firewall/csv/{date}_VAST11MC2_firewall_log*.csv")
    ]

    # Firewall event fields
    N_FIELDS = 15
    (
        f_time, f_priority, f_operation, f_code, f_protocol,
        f_source_ip, f_dest_ip, f_source_host, f_dest_host,
        f_source_port, f_dest_port, f_dest_service, f_direction,
        f_num_conn_built, f_num_conn_teardown
    ) = range(N_FIELDS)

    DATE_FORMAT = "%d/%b/%Y %H:%M:%S"

    @staticmethod
    def _tryparseint(x, dft=0):
        try:
            return int(x)
        except ValueError:
            return dft

    @staticmethod
    def _parse_time(timestamp):
        return time.mktime(time.strptime(timestamp, FWEventParser.DATE_FORMAT))
    
    def parse_fw_events(self, fname):
        tryparseint = self._tryparseint
        parse_time = self._parse_time
        N_FIELDS = self.N_FIELDS

        reader = csv.reader(open(fname))
        header = next(reader) # throw away fields
        assert len(header) == N_FIELDS
        for rec in reader:
            if len(rec) != N_FIELDS:
                print(f"Skipping INVALID LINE: {rec}")
                continue
            t = parse_time(rec[self.f_time])
            source_ip = rec[self.f_source_ip]
            dest_ip = rec[self.f_dest_ip]
            if not source_ip or source_ip == "(empty)": continue
            if not dest_ip or dest_ip == "(empty)": continue
            source_port = tryparseint(rec[self.f_source_port])
            dest_port = tryparseint(rec[self.f_dest_port])
            conn_built = tryparseint(rec[self.f_num_conn_built])
            conn_teardown = tryparseint(rec[self.f_num_conn_teardown])
            yield FWEvent(t, source_ip, dest_ip, source_port, dest_port, conn_built, conn_teardown)
        
    def parse_all_fw_events(self):
        for fname in self.FWFILES:
            print(f"parsing file: {fname}")
            for event in self.parse_fw_events(fname):
                yield event


def parse_all_fw_events():
    parser = FWEventParser()
    return parser.parse_all_fw_events()


class VAST11GraphProps:
    """Scatterplot properties for VAST'11 data."""

    def __init__(self, nodeweights, wscale=400):
        self._w = nodeweights
        self._medw = np.median(list(self._w.values()))
        self._wscale = wscale

    def get_size(self, idx):
        # strange math, but better visual dynamics
        w = self._w[idx]
        return self._wscale * math.log(1 + (w / self._medw))

    def get_color(self, idx):
        return COLOR_BY_KIND[NODES[idx].kind]

    def get_alpha(self):
        return 0.5

    def get_label(self, idx):
        node = NODES[idx]
        if node.kind in (NodeKind.Server, NodeKind.External, NodeKind.Vulnerable_Workatation):
            return node.name
        else:
            return None

    def get_label_size(self, idx):
        node = NODES[idx]
        if node.kind == NodeKind.Server:
            return 18
        else:
            return 12


# Network interfaces for Cisco switch
INTERFACES = ["10.200.150.1", "192.168.1.1", "192.168.2.1", "172.20.1.1"]


def vast11_topology():

    idx = lambda ip: NODE_BY_IP[ip].id

    for if1 in INTERFACES:
        for if2 in INTERFACES:
            if if1 != if2:
                yield (idx(if1), idx(if2))

    # connect External Web to DMZ
    yield (idx("172.20.1.1"), idx("172.20.1.5"))

    # connect IP ranges in other subnets
    for i in range(2, 256):
        yield (idx("192.168.1.1"), idx(f"192.168.1.{i}"))
        yield (idx("192.168.2.1"), idx(f"192.168.2.{i}"))
        yield (idx("10.200.150.1"), idx(f"10.200.150.{i}"))



