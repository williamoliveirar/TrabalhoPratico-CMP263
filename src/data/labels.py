"""Label mapping (docs/PLANO.md §3.3): capture folder → label34 → category8 → y.

In the per-capture release the label is the folder of each capture (there is no label column). Three folders
are spelled differently from the capture files inside them (e.g. `Benign_Final/BenignTraffic1.pcap.csv`);
label34 follows the file names, which is also the spelling of the MERGED_CSV release used in the literature.
Categories are the 7 attack groups of the dataset paper plus Benign.
"""
FOLDER_RENAMES = {
    "Benign_Final": "BenignTraffic",
    "DDoS-PSHACK_FLOOD": "DDoS-PSHACK_Flood",
    "DDoS-RSTFINFLOOD": "DDoS-RSTFINFlood",
}
BENIGN = "BenignTraffic"

CATEGORY = {
    "BenignTraffic": "Benign",
    "DDoS-ACK_Fragmentation": "DDoS",
    "DDoS-HTTP_Flood": "DDoS",
    "DDoS-ICMP_Flood": "DDoS",
    "DDoS-ICMP_Fragmentation": "DDoS",
    "DDoS-PSHACK_Flood": "DDoS",
    "DDoS-RSTFINFlood": "DDoS",
    "DDoS-SYN_Flood": "DDoS",
    "DDoS-SlowLoris": "DDoS",
    "DDoS-SynonymousIP_Flood": "DDoS",
    "DDoS-TCP_Flood": "DDoS",
    "DDoS-UDP_Flood": "DDoS",
    "DDoS-UDP_Fragmentation": "DDoS",
    "DoS-HTTP_Flood": "DoS",
    "DoS-SYN_Flood": "DoS",
    "DoS-TCP_Flood": "DoS",
    "DoS-UDP_Flood": "DoS",
    "Recon-HostDiscovery": "Recon",
    "Recon-OSScan": "Recon",
    "Recon-PingSweep": "Recon",
    "Recon-PortScan": "Recon",
    "VulnerabilityScan": "Recon",
    "Backdoor_Malware": "Web",
    "BrowserHijacking": "Web",
    "CommandInjection": "Web",
    "SqlInjection": "Web",
    "Uploading_Attack": "Web",
    "XSS": "Web",
    "DictionaryBruteForce": "BruteForce",
    "DNS_Spoofing": "Spoofing",
    "MITM-ArpSpoofing": "Spoofing",
    "Mirai-greeth_flood": "Mirai",
    "Mirai-greip_flood": "Mirai",
    "Mirai-udpplain": "Mirai",
}


# Dataset paper, §4: features are summarised over windows of 100 packets for the DDoS, DoS and Mirai attacks and of
# 10 packets for benign traffic and every other attack (D12: window size is a shortcut candidate).
WINDOW_100_CATEGORIES = frozenset({"DDoS", "DoS", "Mirai"})


def window_of(label: str) -> int:
    """Packet-window size the dataset paper reports for this label (10 or 100)."""
    return 100 if CATEGORY[label] in WINDOW_100_CATEGORIES else 10


def label_of_folder(folder: str) -> str:
    label = FOLDER_RENAMES.get(folder, folder)
    if label not in CATEGORY:
        raise KeyError(f"unknown capture folder: {folder!r}")
    return label


def target_of(label: str) -> int:
    """D1: attack = 1, benign = 0."""
    return int(label != BENIGN)
