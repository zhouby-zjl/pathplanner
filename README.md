# PathPlanner: ECMP Path Planning with Minimal Overlap for Efficient Cross-Host Collective Communications

## Overview

In data center networks, cross-host collective communications (CC) for LLM training often suffer from ECMP's hash-based randomness. This randomness funnels flows onto overlapping spine–leaf links, causing network hotspots, rank stragglers, and degraded CC efficiency.  

A promising yet underexplored approach is to plan cross-host paths ahead of CC initialization, leveraging host-side steering and ECMP hash linearity via lightweight packet-header modification. Assigning cross-host paths to minimize link overlap and balance load is NP-complete for large-scale networks.  

PathPlanner is a centralized service that:  
- Heuristically selects near-optimal paths with minimal spine–leaf overlap.  
- Generates multiple valid source ports via host-side steering.  
- Distributes source ports to workers so that each flow follows its intended path without modifying application logic.  

Results from high-fidelity SimAI simulations with realistic LLM workloads:  
- CC primitive flow completion times reduced by up to 44.7%.  
- Overall execution times reduced by up to 21.35%.  
- In 32-rank Mixtral training, per-iteration runtimes shortened by 1.6–2.1s, with estimated cumulative savings of 2.65–3.52 days over a complete training run.

---

## Setup

PathPlanner is built on the Host-Based Path Selector (HPS), available at [https://github.com/zhouby-zjl/hps](https://github.com/zhouby-zjl/hps).  

### Prerequisites
- HPS (follow instructions in the HPS repository)  
- SimAI simulator  
- Python 3.8+ (tested with 3.8.3)

### Installation Steps

1. Install HPS and SimAI according to the instructions in the HPS repository.  

2. Copy the network extension to SimAI and recompile NS-3:
   ```bash
   cp LinearityNetwork.cc \
       SimAI/astra-sim-alibabacloud/extern/network_backend/ns3-interface/simulation/scratch/
   # Then recompile SimAI NS-3
````

3. Ensure Python interpreter is set up for executing the provided scripts.

4. Generate a customized SimAI topology:

   ```bash
   ./dcn-sim.py -t simai_topo -k 16 -z 32 -n 100 -o valid \
       -l /home/zby/ext/SimAI/ -y /home/zby/ext/SimAI/
   ```

5. Run SimAI to start the data plane for the DCN:

   ```bash
   build/scratch/ns3.36.1-LinearityNetwork-debug \
       -t 10 \
       -w example/workload_analytical.txt \
       -n ./simai-topo \
       -c SimAI/astra-sim-alibabacloud/inputs/config/SimAI.conf \
       -l ./ -s
   ```

   * For a full list of options:

     ```bash
     build/scratch/ns3.36.1-LinearityNetwork-debug -h
     ```

6. Run PathPlanner with HPS and SimAI:

   ```bash
   ./dcn-sim.py -t simai_run -k 16 -z 32 -n 100 -o valid \
       -l /home/zby/ext/SimAI/ -y /home/zby/ext/SimAI/
   ```

   * SimAI and PathPlanner communicate via RESTful APIs.
   * Results are output to a designated directory.

---

## Paper

[1] Boyang Zhou, Chunming Wu, Qiang Yang, and Bing Hu. Planning ECMP Paths with Minimal Overlap for Efficient Cross-Host Collective Communications. Submitted to IEEE ICPADS 2025.

---

## License

This work is licensed under CC BY-NC-SA 4.0
[https://creativecommons.org/licenses/by-nc-sa/4.0/](https://creativecommons.org/licenses/by-nc-sa/4.0/)

Copyright (c) 2025 Boyang Zhou

This file is part of ["PathPlanner"](https://github.com/zhouby-zjl/pathplanner/).

---

