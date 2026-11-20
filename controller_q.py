"""
controller_q.py - Ryu controller (v2, FIXED PATHS) for the SDN Q-Learning
project. This file replaces the v1 learning-switch controller.

WHY v2 EXISTS (the bug it fixes):
    v1 was the textbook Ryu learning switch: a switch that does not know a
    destination FLOODS the packet out of every port, and learns host
    locations from whatever comes back. Our topology is a LOOP
    (s1 - s2 - s4 - s3 - s1), so flooded frames (above all ARP requests)
    circle the loop and come back in through other ports. Switches then
    learn host MAC addresses on WRONG ports and install wrong flows.
    In practice pingall failed deterministically for exactly one pair,
    h1 <-> h4 (the longest diagonal), on every run, while all other
    pairs passed. More warm-up runs did not help - it is not a warm-up
    problem, it is a flooding-in-a-loop problem.

WHAT v2 DOES INSTEAD:
    The topology is small and fixed, so the controller keeps a FIXED
    next-hop table: for every switch and every destination host, the one
    correct output port. No flooding ever happens for our traffic:
      - unicast IP/ARP frames follow the table;
      - an ARP request (broadcast) is NOT flooded either - the controller
        reads its target IP and sends the single frame only toward that
        target host.
    A frame can therefore never circle the loop, and forwarding is the
    same on every run. This is also the more honest SDN design for the
    report/viva: the controller computes the paths; only the Q-Learning
    agent's choice is dynamic.

PROJECT BEHAVIOUR (unchanged from v1):
    The main flow h1 (10.0.0.1) -> h2 (10.0.0.2) does not use the default
    table entry on s1. It follows the agent's action in current_action.txt:
        0 = Path 1 (s1 -> s2 -> s4)     1 = Path 2 (s1 -> s3 -> s4)
    NEW in v2: the return flow h2 -> h1 is routed on s4 along the SAME
    path, so a ping's round trip uses one path and delay/loss numbers
    stay meaningful. (In v1 the reply came back by whatever the learning
    switch happened to know.)
    The background flow h3 <-> h4 always uses Path 1 (the default table).

PORT MAP (verified live in the VM on 2026-11-13 with the Mininet CLI
'net' command):
    s1: 1=h1, 2=h3, 3=s2 (Path 1), 4=s3 (Path 2)
    s2: 1=s1, 2=s4
    s3: 1=s1, 2=s4
    s4: 1=h2, 2=h4, 3=s2 (Path 1 side), 4=s3 (Path 2 side)

Run INSIDE the Mininet VM, before starting the topology:
    ryu-manager controller_q.py
After changing current_action.txt, clear stale flows before measuring:
    sh ovs-ofctl del-flows s1     (and: sh ovs-ofctl del-flows s4)
"""

import os

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, arp, ether_types

ACTION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "current_action.txt")

MAIN_SRC_HOST = 1          # h1 = 10.0.0.1
MAIN_DST_HOST = 2          # h2 = 10.0.0.2
S1_DPID = 1
S4_DPID = 4
S1_PORT_PATH1 = 3          # s1 -> s2
S1_PORT_PATH2 = 4          # s1 -> s3
S4_PORT_PATH1 = 3          # s4 -> s2
S4_PORT_PATH2 = 4          # s4 -> s3

BROADCAST_MAC = "ff:ff:ff:ff:ff:ff"

# Fixed next-hop table: NEXT_PORT[dpid][destination host number] = port.
# Host number = last octet of the 10.0.0.x address (h1=1, h2=2, h3=3, h4=4).
# Cross-diamond destinations default to Path 1; the main flow's entries on
# s1 (to host 2) and s4 (to host 1) are overridden by the agent's action
# inside decide_out_port().
NEXT_PORT = {
    1: {1: 1, 3: 2, 2: 3, 4: 3},   # s1
    2: {1: 1, 3: 1, 2: 2, 4: 2},   # s2
    3: {1: 1, 3: 1, 2: 2, 4: 2},   # s3
    4: {2: 1, 4: 2, 1: 3, 3: 3},   # s4
}


def read_current_action():
    """Agent's current action for the main flow: 0 = Path 1, 1 = Path 2."""
    try:
        with open(ACTION_FILE) as f:
            return int(f.read().strip()) % 2
    except Exception:
        return 0  # file missing/unreadable -> safe default: shortest path


def host_of_ip(ip):
    """'10.0.0.4' -> 4. Returns None if the address is not one of ours."""
    try:
        host = int(ip.split(".")[3])
        return host if host in (1, 2, 3, 4) else None
    except Exception:
        return None


def host_of_mac(mac):
    """autoSetMacs gives hosts 00:00:00:00:00:0X -> X. Fallback only."""
    try:
        host = int(mac.split(":")[5], 16)
        return host if host in (1, 2, 3, 4) else None
    except Exception:
        return None


def decide_out_port(dpid, src_host, dst_host, action):
    """The one correct output port for a frame, or None to drop it.

    Pure function of the fixed table plus the agent's action for the
    main flow (forward on s1, return on s4, same path both ways).
    """
    if dst_host is None:
        return None
    if dpid == S1_DPID and src_host == MAIN_SRC_HOST \
            and dst_host == MAIN_DST_HOST:
        return S1_PORT_PATH1 if action == 0 else S1_PORT_PATH2
    if dpid == S4_DPID and src_host == MAIN_DST_HOST \
            and dst_host == MAIN_SRC_HOST:
        return S4_PORT_PATH1 if action == 0 else S4_PORT_PATH2
    return NEXT_PORT.get(dpid, {}).get(dst_host)


class QLearningSwitch(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(QLearningSwitch, self).__init__(*args, **kwargs)

    # --- table-miss: send unknown frames to the controller -----------------
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)

    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match,
                                    instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst)
        datapath.send_msg(mod)

    # --- packet-in: fixed-path forwarding + project override ---------------
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match["in_port"]

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return
        dpid = datapath.id

        # Whose frame is it, and where is it going? IP first, ARP next,
        # MAC (autoSetMacs) as a last resort. For an ARP request the
        # ethernet destination is broadcast, so the TARGET comes from
        # inside the ARP packet - that is what kills the flooding.
        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        arp_pkt = pkt.get_protocol(arp.arp)
        if ip_pkt is not None:
            src_host = host_of_ip(ip_pkt.src)
            dst_host = host_of_ip(ip_pkt.dst)
        elif arp_pkt is not None:
            src_host = host_of_ip(arp_pkt.src_ip)
            dst_host = host_of_ip(arp_pkt.dst_ip)
        else:
            src_host = host_of_mac(eth.src)
            dst_host = host_of_mac(eth.dst)

        action = read_current_action()
        out_port = decide_out_port(dpid, src_host, dst_host, action)
        if out_port is None or out_port == in_port:
            # Nowhere sensible to send it. Dropping beats looping:
            # a frame sent round the diamond would corrupt nothing here,
            # but it would also never arrive - so end it quietly.
            return

        if dpid == S1_DPID and src_host == MAIN_SRC_HOST \
                and dst_host == MAIN_DST_HOST:
            self.logger.info("Q-action on s1: main flow -> Path %d",
                             action + 1)

        actions = [parser.OFPActionOutput(out_port)]

        # Install a flow so later packets of this pair skip the controller.
        # NEVER for broadcast frames: one (in_port, src, broadcast-dst)
        # flow would hijack ARP requests meant for other targets.
        if eth.dst != BROADCAST_MAC:
            match = parser.OFPMatch(in_port=in_port, eth_dst=eth.dst,
                                    eth_src=eth.src)
            if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                self.add_flow(datapath, 1, match, actions, msg.buffer_id)
                return
            self.add_flow(datapath, 1, match, actions)

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data
        out = parser.OFPPacketOut(datapath=datapath,
                                  buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions,
                                  data=data)
        datapath.send_msg(out)
