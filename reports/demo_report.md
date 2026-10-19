# GMCR incident pipeline: demo on held-out VMs

Synthetic telemetry. Each case: failure risk as the failure approaches (A), then root cause (B) and recovery plan (C) one minute after the VM went down.

## inc-0003: vm-001 (azure), true cause: disk_exhaustion

- **A, risk at 15/10/5/1 min before:** 9% → 33% → 82% → 86%
- **A, first warning:** Elevated risk of failure within the next 15 minutes (33%). Signals: disk latency at 1.3× normal; free disk changed by -24% in 15 min; disk full in about 16 min at the current rate.
- **B, after it went down:** Likely cause: disk exhaustion (92%); supporting evidence: free disk fell to 4%, 10 disk-full errors in the last hour, disk latency at 1.4× normal.
- **C:** SEV2. Restore ETA: 20 minutes; investigate disk usage (logs, temp files) first; extend the volume or free space, then restart services.

## inc-0001: vm-000 (aws), true cause: storage_io

- **A, risk at 15/10/5/1 min before:** 16% → 12% → 77% → 77%
- **A, first warning:** High risk of failure within the next 15 minutes (77%). Signals: disk latency at 43 ms; 1 disk I/O error in the last 15 minutes; CPU at 1.4× normal.
- **B, after it went down:** Likely cause: storage I/O failure (99%); supporting evidence: disk latency peaked at 8.3× normal, 5 disk I/O errors in the last hour.
- **C:** SEV2. Restore ETA: 103 minutes (P90 105); investigate storage first; restore the disk onto healthy storage; prepare an alternative recovery target.

## inc-0004: vm-001 (azure), true cause: network

- **A, risk at 15/10/5/1 min before:** 0% → 19% → 94% → 98%
- **A, first warning:** High risk of failure within the next 15 minutes (94%). Signals: network errors at 5.0× normal; 1 failed health check; 4 earlier failures on this VM.
- **B, after it went down:** Likely cause: network failure (96%); supporting evidence: network errors peaked at 24/min, 2 failed health checks, 4 network timeouts in the last hour.
- **C:** SEV2. Restore ETA: 25 minutes; investigate the network path (NIC, security rules, routes) first; fail traffic over and reattach or replace the network interface.

## inc-0033: vm-014 (vmware), true cause: host

- **A, risk at 15/10/5/1 min before:** 0% → 0% → 0% → 0%
- **A, first warning:** Low risk of failure within the next 15 minutes (0%).
- **B, after it went down:** Likely cause: host issue (98%); supporting evidence: 1 host-degraded platform event in the last hour, disk latency peaked at 1.4× normal, 1 failed health check.
- **C:** SEV2. Restore ETA: 82 minutes (P90 84); investigate the host first; redeploy the VM onto a healthy host; prepare an alternative recovery target.

## inc-0006: vm-002 (vmware), true cause: os_crash

- **A, risk at 15/10/5/1 min before:** 7% → 9% → 30% → 36%
- **A, first warning:** Elevated risk of failure within the next 15 minutes (30%). Signals: memory at 1.3× normal; 1 out-of-memory kill in the last hour.
- **B, after it went down:** Likely cause: OS crash (99%); supporting evidence: memory peaked at 1.7× normal, 1 out-of-memory kill in the last hour, 1 kernel panic in the last hour.
- **C:** SEV2. Restore ETA: 15 minutes; investigate the OS (memory use, kernel logs) first; reboot; restore from the last good snapshot if it recurs.

## inc-0000: vm-000 (aws), true cause: app

- **A, risk at 15/10/5/1 min before:** 0% → 6% → 29% → 85%
- **A, first warning:** Elevated risk of failure within the next 15 minutes (29%). Signals: 3 HTTP 5xx errors in the last 15 minutes; 1 failed health check.
- **B, after it went down:** Likely cause: application failure (99%); supporting evidence: 10 HTTP 5xx errors in the last hour, 1 application crash in the last hour, 2 failed health checks.
- **C:** SEV2. Restore ETA: 20 minutes; investigate the application (recent deploys, error logs) first; roll back the last deploy and restart the service.

## inc-0005: vm-001 (azure), true cause: security

- **A, risk at 15/10/5/1 min before:** 0% → 1% → 93% → 99%
- **A, first warning:** High risk of failure within the next 15 minutes (93%). Signals: network errors at 2.5× normal; CPU changed by +27% in 15 min; 5 earlier failures on this VM.
- **B, after it went down:** Likely cause: suspected security incident (99%); supporting evidence: 2 security alerts in the last hour, 11 failed logins in the last hour, network errors at 1.4× normal.
- **C:** SEV1. Restore ETA: 214 minutes; investigate the security incident first; isolate the VM, preserve forensic evidence, rebuild from a clean snapshot; prepare an alternative recovery target.

## Fleet triage at minute 1709 (held-out VMs)

| vm_id   | severity   |   priority | tier     | status            | likely_cause   |   eta_p50_min | next_step                                                               |
|:--------|:-----------|-----------:|:---------|:------------------|:---------------|--------------:|:------------------------------------------------------------------------|
| vm-098  | SEV1       |      10.4  | standard | down              | security       |           577 | investigate the security incident first                                 |
| vm-088  | SEV2       |       3.09 | standard | down              | storage_io     |           127 | investigate storage first                                               |
| vm-011  | SEV3       |       0.89 | standard | Elevated risk 24% | -              |           202 | no outage yet: act on the warning signals and pre-stage a restore point |
