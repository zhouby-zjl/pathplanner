from flask import Flask, request, jsonify
import random as rand
from collections import defaultdict, deque
from controller import controller
from dcn_networks import dcn_network
import socket
import struct
import threading
import time
import numpy as np
import networkx as nx
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import random
import config
from tracer import tracer
import pandas as pd
import seaborn as sns
import csv
from datetime import datetime

class RestApiServer:
    class flow_allocation_state:
        def __init__(self, path, path_assigned, flow_size, flow_allocation_time, layer_num):
            self.path = path
            self.path_assigned = path_assigned
            self.flow_size = flow_size
            self.flow_allocation_time = flow_allocation_time
            self.layer_num = layer_num
            self.allocated = True

    def __init__(self, net : dcn_network, ctl: controller, log_prefix):
        self.net = net
        self.app = Flask(__name__)
        self.app.add_url_rule('/api/planpaths', view_func=self.planpaths, methods=["POST"])
        self.app.add_url_rule('/api/flowinitiated', view_func=self.cross_node_flow_initiated)
        self.app.add_url_rule('/api/flowfinish', view_func=self.flowfinish)
        self.app.add_url_rule('/api/reportpath', view_func=self.reportpath)
        self.app.add_url_rule('/api/reportqbb', view_func=self.reportqbb, methods=["POST"])
        self.sport_id_map = defaultdict(lambda: defaultdict(int))
        self.sport_id_map_rand = defaultdict(lambda: defaultdict(list))
        self.ctl = ctl
        self.max_retrying_time = 3

        self.history = deque(maxlen=2)
        self.throughput = {}  # 当前瞬时吞吐量
        self.hot_links = set()  # 记录热点链路
        self.lock = threading.Lock()

        self.link_load = defaultdict(float)  # 全局链路负载状态
        self.alpha = 2  # 控制重叠链路惩罚权重
        self.link_forward_speed = 40e9

        self.path_allocation_states = {}  # map from the hash of a flow header to its path designated states

        self.log_prefix = log_prefix
        self.flow_log_file = open(log_prefix + 'flows-changes.csv', 'w')
        self.qbb_log_file = open(log_prefix + 'qbb-changes.csv', 'w')
        self.time_log_file = open(log_prefix + 'exe-time.csv', 'w')
        self.progression_file = open(log_prefix + 'progression', 'w')

        self.path_assigned_all = defaultdict(lambda : defaultdict(list))

    def planpaths(self):
        json_data = request.get_json()
        if not json_data:
            return "Invalid JSON", 400

        start_time_ns = time.time() * 1e9
        result_map = {}
        flow_idx = 0
        max_disjoint_paths_all_across_ps = {}
        src_dst_flows_all_across_ps = {}
        exe_time_records = []
        for ps in json_data.keys():
            start_time_ns_comp_paths = time.time() * 1e9
            src_dst_flows = []
            max_disjoint_paths_all_across_ps[ps] = {}
            for ring_id in json_data[ps].keys():
                for pair_str in json_data[ps][ring_id]:
                    src, dst = map(int, pair_str.split("-"))
                    node_src_id = self.net.id_rmap[src]
                    node_dst_id = self.net.id_rmap[dst]
                    if int(np.min([len(p) for p in self.net.paths_all[node_src_id][node_dst_id]])) == 3:
                        continue       # skip the pair directly connected via a leaf switch

                    flow = controller.flow_info()
                    flow.idx = flow_idx
                    flow.node_src_id = node_src_id
                    flow.node_dst_id = node_dst_id
                    flow.port_src = rand.randint(1000,65535)
                    flow.port_dst = 100
                    flow.flow_size = 1000
                    src_dst_flows.append(flow)
                    flow_idx += 1

            src_dst_flows_all_across_ps[ps] = src_dst_flows
            if len(src_dst_flows) == 0:
                continue

            max_disjoint_paths = self.ctl.gen_max_disjoint_paths_for_flows(src_dst_flows)
            max_disjoint_paths_all_across_ps[ps] = max_disjoint_paths
            exe_time_ns_comp_paths =  int(time.time() * 1e9 - start_time_ns_comp_paths)

            start_time_ns_comp_sp = time.time() * 1e9
            m_addresses = 100
            result_map[ps] = {}
            for flow in src_dst_flows:
                valid_sport_all = self.gen_m_diff_valid_sport(flow, max_disjoint_paths, m_addresses)
                self.path_assigned_all[flow.node_src_id][flow.node_dst_id] = max_disjoint_paths[flow.idx][0]
                result_map[ps][f"{self.net.id_map[flow.node_src_id]}-{self.net.id_map[flow.node_dst_id]}"] = valid_sport_all

            exe_time_ns_sp = int(time.time() * 1e9 - start_time_ns_comp_sp)
            exe_time_records.append([exe_time_ns_comp_paths, exe_time_ns_sp])

        execution_time_ns = int(time.time() * 1e9 - start_time_ns)
        self.time_log_file.write(f"total_ns:{execution_time_ns}\n")
        for t_path, t_sp in exe_time_records:
            self.time_log_file.write(f"{t_path},{t_sp}\n")
        self.time_log_file.flush()

        for ps, max_disjoint_paths in max_disjoint_paths_all_across_ps.items():
            if len(max_disjoint_paths) == 0:
                continue
            orig_ecmp_paths_info = self.ctl.gen_orig_ecmp_paths(src_dst_flows_all_across_ps[ps])
            paths = [paths_info[1] for paths_info in orig_ecmp_paths_info]

            md_path_vectors = [max_disjoint_paths[flow_idx][0] for flow_idx in max_disjoint_paths]
            output_fig_filename = f'{self.log_prefix}link-density-{ps}.eps'
            self.gen_heatmap_link_density(paths, md_path_vectors, output_fig_filename)
            self.save_paths_to_csv(paths, f'{self.log_prefix}paths-ecmp-{ps}.csv')
            self.save_paths_to_csv(md_path_vectors, f'{self.log_prefix}paths-hps-{ps}.csv')

        return jsonify(result_map)


    def save_paths_to_csv(self, paths, filename):
        with open(filename, mode='w', newline='') as f:
            writer = csv.writer(f)

            for path in paths:
                if not path or len(path) < 2:
                    continue  # 跳过无效路径
                src = path[0]
                dst = path[-1]
                path_str = '-'.join(map(str, path))
                writer.writerow([src, dst, path_str])

    def gen_heatmap_link_density(self, paths_ecmp, paths_disjoint, output_fig_filename):
        link_weights_ecmp = defaultdict(int)
        link_weights_disjoint = defaultdict(int)
        heatmap_all = []
        for link_weights, paths in zip([link_weights_ecmp, link_weights_disjoint], [paths_ecmp, paths_disjoint]):
            for p in paths:
                links = [(p[i], p[i + 1]) for i in range(len(p) - 1)]
                for link in links:
                    link_weights[link] += 1

            df = pd.DataFrame(list(link_weights.items()), columns=['Link', 'FlowCount'])
            df['Link'] = df['Link'].astype(str)
            heatmap = df.pivot_table(index='Link', values='FlowCount', aggfunc='sum').sort_values(by='FlowCount',
                                                                                                 ascending=False)
            heatmap_all.append(heatmap)

        fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(heatmap_all[0]) * 0.25)))

        sns.heatmap(heatmap_all[0], annot=True, fmt=".0f", cmap='viridis', cbar=True, ax=axes[0])
        axes[0].set_title("Flow-Size-Weighted Link Usage - Random Src Ports")
        axes[0].set_xlabel("Flow Size Sum (bytes)")
        axes[0].set_ylabel("Link (node1, node2)")

        sns.heatmap(heatmap_all[1], annot=True, fmt=".0f", cmap='viridis', cbar=True, ax=axes[1])
        axes[1].set_title("Flow-Size-Weighted Link Usage - Predetermined Src Ports")
        axes[1].set_xlabel("Flow Size Sum (bytes)")
        axes[1].set_ylabel("")

        plt.tight_layout()
        plt.savefig(output_fig_filename, bbox_inches='tight')



    def gen_m_diff_valid_sport(self, flow, max_disjoint_paths, m, need_valid=False):
        valid_sport_all = set()
        while len(valid_sport_all) < m:
            sport, succ, retrying_times = self.gen_valid_sport(flow, max_disjoint_paths)
            if succ and sport not in valid_sport_all:
                valid_sport_all.add(sport)
                if need_valid:
                    valid, path_fd, designated_path = self.ctl.validate_designated_flow(flow, sport,
                                                                                   max_disjoint_paths, True)
                    tracer.log(f"validation for the flow {flow}: {valid}")

        return list(valid_sport_all)

    def gen_valid_sport(self, flow, max_disjoint_paths):
        port_srcs_to_retry = []
        port_src_max = 2 ** config.HEADER_CHANGE_BITS - 1
        port_src_min = 1
        retried_port_srcs = set()

        for i in range(self.max_retrying_time):
            while True:
                port_src = random.randint(port_src_min, port_src_max)
                if port_src not in retried_port_srcs:
                    break
            port_srcs_to_retry.append(port_src)

        ep_port_src, succ, retrying_times = self.ctl.find_ep_port_src_uha(flow,
                                                                          max_disjoint_paths, self.max_retrying_time,
                                                                          False,
                                                                          port_srcs_to_retry, True)
        tracer.retrying_times(retrying_times)
        if succ:
            sport = ep_port_src
            return sport, True, retrying_times
        else:
            sport = random.randint(port_src_min, port_src_max)
            tracer.log(f"obtaining valid delta is failed for the flow {flow}")
            return sport, False, retrying_times

    def gen_port_srcs_to_retry_for_flows(self, max_retrying_time, n_flows):
        retried_port_srcs = set()
        port_src_max = 2 ** config.HEADER_CHANGE_BITS - 1
        port_src_min = 1

        port_srcs_to_retry_all = []
        for k in range(n_flows):
            port_srcs_to_retry = []
            for i in range(max_retrying_time):
                while True:
                    port_src = random.randint(port_src_min, port_src_max)
                    if port_src not in retried_port_srcs:
                        break
                port_srcs_to_retry.append(port_src)

            port_srcs_to_retry_all.append(port_srcs_to_retry)
        return port_srcs_to_retry_all


    def cross_node_flow_initiated(self):
        try:
            flowId = int(request.args.get('flowId'))
            src = int(request.args.get('src'))
            dst = int(request.args.get('dst'))
            sport = int(request.args.get('sport'))
            dport = int(request.args.get('dport'))
            packetCount = int(request.args.get('packetCount'))
            layerNum = int(request.args.get('layerNum'))
            time = int(request.args.get('time'))
            print(f"[API Server] cross node flow initiated with ({src}:{sport} -> {dst}:{dport}, {flowId}, {packetCount}, {layerNum}, {time})")
            print(f"!!!!!! ===> PROGRESSION at LayerNum: {layerNum}")
            self.progression_file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: ({src}:{sport} -> {dst}:{dport}, {flowId}, {packetCount}) @ layerNum: {layerNum}\n")
            self.progression_file.flush()
        except (TypeError, ValueError):
            return "-1", 400

        flow = controller.flow_info()
        flow.node_src_id = self.net.id_rmap[src]
        flow.node_dst_id = self.net.id_rmap[dst]
        flow.port_dst = dport
        flow.flow_size = packetCount

        f_str = self.flow_info_to_str(flow.node_src_id, flow.node_dst_id, sport, flow.port_dst)
        if f_str not in self.path_allocation_states:
            path_assigned = self.path_assigned_all[flow.node_src_id][flow.node_dst_id]
            fas = RestApiServer.flow_allocation_state(None, path_assigned, packetCount, time, layerNum)
            self.path_allocation_states[f_str] = fas
        else:
            fas = self.path_allocation_states[f_str]
            fas.allocated = False
            fas.flow_size = packetCount
            fas.flow_allocation_time = time
        return "0", 200

    def flowfinish(self):
        try:
            flowId = int(request.args.get('flowId'))
            src = int(request.args.get('src'))
            dst = int(request.args.get('dst'))
            sport = int(request.args.get('sport'))
            dport = int(request.args.get('dport'))
            time = int(request.args.get('time'))
            print(f"[API Server] flow finish with ({src}:{sport} -> {dst}:{dport}, {flowId}, {time})")

            src_mapped = self.net.id_rmap[src]
            dst_mapped = self.net.id_rmap[dst]
            f_str = self.flow_info_to_str(src_mapped, dst_mapped, sport, dport)
            if f_str in self.path_allocation_states:
                fas : RestApiServer.flow_allocation_state = self.path_allocation_states[f_str]
                if fas is not None:
                    fas.allocated = False
                    if fas.path is not None:
                        self.release_path_for_flow(fas.path, fas.flow_size, self.link_forward_speed)
                        self.flow_log_file.write(f"{src_mapped},{dst_mapped},{sport},{dport},{'-'.join([str(nid) for nid in fas.path])},{fas.flow_size},{fas.flow_allocation_time},{time},{fas.layer_num}\n")
                    else:
                        self.flow_log_file.write(
                            f"{src_mapped},{dst_mapped},{sport},{dport},None,{fas.flow_size},{fas.flow_allocation_time},{time},{fas.layer_num}\n")
                    self.flow_log_file.flush()
        except (TypeError, ValueError):
            return "-1", 400

        return "0", 200

    def reportpath(self):
        try:
            src = int(request.args.get('src'))
            dst = int(request.args.get('dst'))
            sport = int(request.args.get('sport'))
            dport = int(request.args.get('dport'))
            time = int(request.args.get('time'))
            path_str = request.args.get('path')
            path = [int(nid) for nid in path_str.split(',')]
            print(f"[API Server] report path of ({src}:{sport} -> {dst}:{dport}, {time}): {path}")

            src_IP_str = self.int_to_ip(src)
            dst_IP_str = self.int_to_ip(dst)
            if src_IP_str in self.net.ip_addresses and dst_IP_str in self.net.ip_addresses:
                src_id = self.net.ip_addresses.index(src_IP_str)
                dst_id = self.net.ip_addresses.index(dst_IP_str)
                if src_id >= 0 and src_id < self.net.n_ep and dst_id >= 0 and dst_id < self.net.n_ep:
                    src_mapped = self.net.id_rmap[src_id]
                    dst_mapped = self.net.id_rmap[dst_id]
                    if (src_mapped in self.net.paths_all and dst_mapped in self.net.paths_all[src_mapped] and
                            len(self.net.paths_all[src_mapped][dst_mapped]) > 0):
                        path_mapped = [self.net.id_rmap[n] for n in path]

                        f = controller.flow_info()
                        f.node_src_id = src_mapped
                        f.node_dst_id = dst_mapped
                        f.port_src = sport
                        f.port_dst = dport
                        path_ecmp, ports_on_path, probe_delay = self.ctl.trace_ecmp_path(f)
                        if path_mapped != path_ecmp:
                            print(f"path inconsistency found: path_mapped from SimAI: {path_mapped}, path in HPS: {path_ecmp}")

                        f_str = self.flow_info_to_str(src_mapped, dst_mapped, sport, dport)
                        if f_str in self.path_allocation_states:
                            p_designated = self.path_allocation_states[f_str].path
                            if p_designated != path_mapped:
                                print(f"path designation failed from SimAI: {path_mapped}, while designated path: {p_designated}")
                            self.path_allocation_states[f_str].path = path_mapped

            return "0"
        except (TypeError, ValueError):
            return "-1", 400

    def reportqbb(self):
        data = request.get_json()
        if not data or 'timestamp' not in data:
            return jsonify({'error': 'Invalid data, missing timestamp'}), 400

        with self.lock:
            self.history.append(data)

            if len(self.history) >= 2:
                latest = self.history[-1]
                prev = self.history[-2]

                t1 = latest['timestamp']
                t0 = prev['timestamp']
                dt_sec = (t1 - t0) / 1e9
                if dt_sec <= 0:
                    return jsonify({'message': 'Data received, time invalid for throughput calc'}), 200

                instant_thrpt = {}
                for link, count1 in latest.items():
                    if link == 'timestamp':
                        continue
                    count0 = prev.get(link, 0)
                    diff = count1 - count0
                    if diff < 0:
                        diff = 0  # 忽略回绕
                    instant_thrpt[link] = diff / dt_sec

                self.throughput = instant_thrpt
                if len(instant_thrpt) > 0:
                    self.analyze_hotspot_links_in_diff_layers(instant_thrpt)

        return jsonify({
            "message": "OK"
        }), 200

    def analyze_hotspot_links_in_diff_layers(self, instant_thrpt):
        down_thrpt = defaultdict(lambda: defaultdict(int))
        up_thrpt = defaultdict(lambda: defaultdict(int))

        for link, thrpt in instant_thrpt.items():
            pair = [self.net.id_rmap[int(nid)] for nid in link.split('-')]
            node_src = self.net.get_node_obj(pair[0])
            node_dst = self.net.get_node_obj(pair[1])
            if node_src is None or node_dst is None:
                continue

            ln_src = node_src.layer_num
            ln_dst = node_dst.layer_num
            key = tuple(pair)
            if ln_src > ln_dst:
                down_thrpt[ln_src][key] = thrpt
            elif ln_dst > ln_src:
                up_thrpt[ln_dst][key] = thrpt

        for thrpt_all, direction in zip([down_thrpt, up_thrpt], ['down', 'up']):
            ln_all = sorted(thrpt_all.keys())
            gini_all = []
            s = direction + ","
            first = True
            for ln in ln_all:
                h = thrpt_all[ln]
                thrpt_vals = list(h.values())
                gini = self.gini_coefficient(thrpt_vals)
                gini_all.append(gini)
                s += "," if not first else ""
                s += f"{ln}:{gini}"
                first = False
            s += "\n"
            self.qbb_log_file.write(s)
            self.qbb_log_file.flush()


    def gini_coefficient(self, values):
        n = len(values)
        if n == 0:
            return 0.0
        sorted_vals = sorted(values)
        cum_diff = 0
        for i in range(n):
            for j in range(n):
                cum_diff += abs(sorted_vals[i] - sorted_vals[j])
        total = sum(sorted_vals)
        return cum_diff / (2 * n * total) if total > 0 else 0.0

    def allocate_path_for_flow(self, node_src_id, node_dst_id, flow_size, forwarding_speed):
        transmission_time = flow_size /  forwarding_speed
        flow_weight = transmission_time if transmission_time > 0 else 1.0

        paths = self.net.paths_all[node_src_id][node_dst_id]
        weighted_overlap_min = float('inf')
        best_path = None

        for p_can in paths:
            lh_p = []
            for i in range(0, len(p_can) - 1):
                lh = self.net.pair_hash(p_can[i], p_can[i + 1])
                lh_p.append(self.link_load[lh])

            lh_p.sort(reverse=True)
            ln_p_len = len(lh_p)
            weighted_overlap = sum([pow(self.alpha, ln_p_len - i) * lh_p[i] for i in range(ln_p_len)])

            if weighted_overlap < weighted_overlap_min:
                weighted_overlap_min = weighted_overlap
                best_path = p_can

        if best_path:
            for i in range(len(best_path) - 1):
                lh = self.net.pair_hash(best_path[i], best_path[i + 1])
                self.link_load[lh] += flow_weight

            ports_on_path = []
            for i in range(len(best_path) - 1):
                node_id_cur = best_path[i]
                node_id_next = best_path[i + 1]
                port_id = self.net.get_port_id(node_id_cur, node_id_next)
                ports_on_path.append(port_id)

            return best_path, ports_on_path, len(paths)
        else:
            return None, [], 0

    def release_path_for_flow(self, path, flow_size, forwarding_speed):
        transmission_time = flow_size / forwarding_speed
        flow_weight = transmission_time if transmission_time > 0 else 1.0

        for i in range(len(path) - 1):
            lh = self.net.pair_hash(path[i], path[i + 1])
            self.link_load[lh] -= flow_weight
            if self.link_load[lh] < 0:
                self.link_load[lh] = 0.0

    def int_to_ip(self, ip_int):
        return socket.inet_ntoa(struct.pack("!I", ip_int))

    def flow_info_to_str(self, src_id, dst_id, port_src, port_dst):
        return f"{src_id}:{port_src}-{dst_id}:{port_dst}"

    def run(self):
        self.app.run(host='127.0.0.1', port=5000, debug=False)

    def end(self):
        self.flow_log_file.close()
        self.qbb_log_file.close()

    def launch_gui(self):
        def plot_gini(gini_data, title, filename):
            plt.figure()
            for timestamp_idx, (layers, ginis) in enumerate(gini_data):
                plt.plot(layers, ginis, marker='o', label=f"t{timestamp_idx}")
            plt.xlabel("Layer Number")
            plt.ylabel("Gini Coefficient")
            plt.title(title)
            plt.legend()
            plt.grid(True)
            plt.savefig(filename)

        root = tk.Tk()
        root.title("DCN Gini Analyzer")

        ttk.Label(root, text="Click a button to plot Gini Coefficients").grid(row=0, column=0, columnspan=2, pady=10)

        down_btn = ttk.Button(root, text="Plot Down Gini",
                              command=lambda: plot_gini(self.down_thrpt_gini_across_times, "Downstream Gini Over Time",
                                                        "output-gini-down.png"))
        down_btn.grid(row=1, column=0, padx=10, pady=10)

        up_btn = ttk.Button(root, text="Plot Up Gini",
                            command=lambda: plot_gini(self.up_thrpt_gini_across_times, "Upstream Gini Over Time",
                                                      "output-gini-up.png"))
        up_btn.grid(row=1, column=1, padx=10, pady=10)

        root.mainloop()