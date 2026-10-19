"""Dataset schema per platform and the physics baseline (minutes)."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data_test"
MODEL_DIR = ROOT / "models"
TARGET = "Restore_Time_Min"

FEEDBACK_DIR = ROOT / "data_feedback"

# controllable: inputs a team can change; only these are flagged as causes or tried in what-ifs.
# location: where a restore ran; used to look for slow regions.
PLATFORMS = {
    "aws": {"name": "AWS", "csv": "aws_restore_data_cleaned.csv", "id": "Instance_ID",
            "num": ["VM_Size_GB", "Provisioned_IOPS", "Snapshot_Age_Days"],
            "cat": ["EBS_Volume_Type", "Region"],
            "controllable": ["EBS_Volume_Type", "Provisioned_IOPS", "Region"], "location": ["Region"]},
    "azure": {"name": "Azure", "csv": "azure_restore_data_cleaned.csv", "id": "VM_Resource_ID",
              "num": ["VM_Size_GB", "Network_Bandwidth_Mbps"],
              "cat": ["Disk_Tier", "Vault_Redundancy", "Restore_Method", "Region"],
              "controllable": ["Disk_Tier", "Vault_Redundancy", "Restore_Method", "Region", "Network_Bandwidth_Mbps"],
              "location": ["Region"]},
    "vmware": {"name": "VMware", "csv": "vmware_restore_data_cleaned.csv", "id": "VM_ID",
               "num": ["VM_Size_GB", "Network_Gbps", "Concurrency_Level"],
               "cat": ["Transport_Mode", "Disk_Provisioning", "Target_Storage"],
               "controllable": ["Transport_Mode", "Disk_Provisioning", "Target_Storage", "Network_Gbps", "Concurrency_Level"],
               "location": []},
}

# The slowest AWS restore in the data: the demo story's example.
DEMO_CASE = {"platform": "aws", "workload": {"VM_Size_GB": 1971, "Provisioned_IOPS": 64000, "Snapshot_Age_Days": 340,
                                             "EBS_Volume_Type": "st1", "Region": "us-east-1"}}

HELP = {
    "VM_Size_GB": "Total size of the VM or dataset to restore.",
    "Provisioned_IOPS": "IOPS provisioned on the restored EBS volume.",
    "Snapshot_Age_Days": "Age of the snapshot being restored.",
    "EBS_Volume_Type": "st1 is throughput HDD; gp2/gp3 general SSD; io2 provisioned-IOPS SSD.",
    "Network_Bandwidth_Mbps": "Raw bandwidth allocated to the restore job.",
    "Disk_Tier": "Target managed-disk tier. Faster tiers reduce write latency.",
    "Vault_Redundancy": "Backup vault replication: LRS (local), ZRS (zone), GRS (geo).",
    "Restore_Method": "Instant mounts the snapshot; Vault copies the data back (slower).",
    "Region": "Where the restore runs.",
    "Network_Gbps": "Network link between backup proxy and target storage.",
    "Concurrency_Level": "Restores running in parallel on the same infrastructure.",
    "Transport_Mode": "NBD goes over the management network; HotAdd and SAN read disks directly.",
    "Disk_Provisioning": "Thin, Thick-Lazy or Thick-Eager zeroed disks.",
    "Target_Storage": "Datastore media the VM is restored to.",
}


def label(col: str) -> str:
    """Display name: VM_Size_GB → VM Size (GB)."""
    out = col.replace("_", " ")
    for unit in ("GB", "Mbps", "Gbps"):
        out = out.replace(f" {unit}", f" ({unit})")
    return out


def features(platform: str) -> list:
    cfg = PLATFORMS[platform]
    return cfg["num"] + cfg["cat"]


def feedback_path(platform: str) -> Path:
    return FEEDBACK_DIR / f"{platform}.csv"


def load_data(platform: str, include_feedback: bool = True) -> pd.DataFrame:
    """Restore records for one platform, plus real restores logged through feedback."""
    cfg = PLATFORMS[platform]
    path = DATA_DIR / cfg["csv"]
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")
    df = pd.read_csv(path)
    missing = sorted(set(features(platform) + [TARGET]) - set(df.columns))
    if missing:
        raise ValueError(f"{cfg['csv']} is missing columns {missing}")
    fb = feedback_path(platform)
    if include_feedback and fb.exists():
        df = pd.concat([df, pd.read_csv(fb)[[cfg["id"], *features(platform), TARGET]]], ignore_index=True)
    df = df.dropna(subset=features(platform) + [TARGET])
    df = df[df[TARGET] > 0].reset_index(drop=True)
    if len(df) < 50:
        raise ValueError(f"{cfg['csv']} has only {len(df)} usable rows; at least 50 are needed to train.")
    return df


def physics_baseline(platform: str, df: pd.DataFrame) -> np.ndarray:
    """Theoretical restore time: data size ÷ nominal throughput, in minutes."""
    size = df["VM_Size_GB"].to_numpy(float)
    if platform == "aws":
        # Datasheet throughput: every IO moves 256 KB, so MB/s = IOPS × 0.25.
        return size * 1024 / (df["Provisioned_IOPS"].to_numpy(float) * 0.25) / 60
    if platform == "azure":
        # Instant restore mounts the snapshot instead of copying it: no bulk transfer to time.
        copy = size * 8192 / df["Network_Bandwidth_Mbps"].to_numpy(float) / 60
        return np.where(df["Restore_Method"].to_numpy() == "Instant", 0.0, copy)
    if platform == "vmware":
        return size * 8192 / (df["Network_Gbps"].to_numpy(float) * 1000) / 60
    raise ValueError(f"Unknown platform {platform!r}")
