import random

#import trafpy.generator as tpg
import csv
import argparse
import numpy as np

class flows_generator:
    def __init__(self, log_prefix):
        self.log_prefix = log_prefix
        self.params_flow_size_dist = {}
        self.params_iat_dist = {}

        self.params_flow_size_dist['commercial_cloud'] = {'dist': 'lognormal', 'params': {'_mu': 7, '_sigma': 2.5},
                                                     'min_val': 1, 'max_val': 2e7, 'round': 25}
        self.params_iat_dist['commercial_cloud'] = {'dist': 'multimodal', 'min_val': 1, 'max_val': 100000,
                                               'locations': [10, 20, 100, 1],
                                               'skews': [0, 0, 0, 100], 'scales': [1, 3, 4, 50],
                                               'num_skew_samples': [10000, 7000, 5000, 20000],
                                               'bg_factor': 0.01, 'round': 25}

    def extract_node_ids_from_flow_path_info_path(self, file_path):
        node_ids = set()
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line:
                    node_a = line.split(',')[0]
                    node_b = line.split(',')[1]
                    node_ids.add(int(node_a))
                    node_ids.add(int(node_b))
        return sorted(node_ids)

    def read_ep_ids_list_from_csv(self, filename):
        with open(filename, mode='r') as file:
            reader = csv.reader(file)
            i = 0
            ep_ids = []
            rack_map = {}
            for row in reader:
                if i == 0:
                    ep_ids = [int(item) for item in row]
                    ep_ids_s = sorted(ep_ids)
                else:
                    rack_sw_id = int(row[0])
                    ep_ids_in_rack = [int(id) for id in row[1].split('-')]
                    rack_map[rack_sw_id] = ep_ids_in_rack
                i += 1
            return ep_ids, rack_map

    def extract_endpoints_and_flows(self, filename):
        n_endpoints = None
        n_flows = None

        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('n_endpoints'):
                    parts = line.split(',')
                    n_endpoints = int(parts[1])
                elif line.startswith('n_flows'):
                    parts = line.split(',')
                    n_flows = int(parts[1])

        return n_endpoints, n_flows

    def save_flow_data_to_csv(self, flow_centric_demand_data, node_ids, filename):
        # Define desired column order
        columns = ['index', 'flow_id', 'event_time', 'sn', 'dn', 'flow_size']

        # Prepare the rows with mapped sn and dn
        rows = []
        num_flows = len(flow_centric_demand_data['index'])
        for i in range(num_flows):
            sn_idx = int(flow_centric_demand_data['sn'][i])
            dn_idx = int(flow_centric_demand_data['dn'][i])
            if sn_idx == dn_idx:
                continue
            row = [
                flow_centric_demand_data['index'][i],
                flow_centric_demand_data['flow_id'][i],
                flow_centric_demand_data['event_time'][i],
                node_ids[sn_idx],  # Map sn
                node_ids[dn_idx],  # Map dn
                flow_centric_demand_data['flow_size'][i]
            ]
            rows.append(row)

        # Write to CSV
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(columns)  # Write header
            writer.writerows(rows)    # Write mapped rows


    def get_val_dist(self, params, endpoints=None):
        if params['dist'] != 'multimodal':
            return tpg.gen_named_val_dist(dist=params['dist'],
                                   params=params['params'],
                                   round_to_nearest=params['round'],
                                    min_val=1, max_val=2e7,
                                   show_fig=False)
        else:
            return tpg.gen_multimodal_val_dist(min_val=params['min_val'], max_val=params['max_val'],
                                               locations=params['locations'],
                                               skews=params['skews'], scales=params['scales'],
                                               num_skew_samples=params['num_skew_samples'], bg_factor=params['bg_factor'],
                                               round_to_nearest=params['round'],
                                               show_fig=False)


    def gen_flows_for_ns3(self, filename, workload_type='commercial_cloud', num_eps=None, num_demands=100):
        #node_ids = extract_node_ids_from_flow_path_info_path(log_prefix + "flow-path-info")
        node_ids, rack_map = self.read_ep_ids_list_from_csv(self.log_prefix + "ep-ids")
        if num_eps is not None:
            num_eps = np.min([num_eps, len(node_ids)])
            n_endpoints = num_eps
            node_ids_sel = random.choices(node_ids, k=num_eps)
            rack_map_sel = {}
            for rack_sw_id in rack_map:
                if rack_sw_id not in node_ids_sel:
                    continue
                node_ids = []
                for node_id in rack_map[rack_sw_id]:
                    if node_id not in node_ids_sel:
                        continue
                    node_ids.append(node_id)
                if len(node_ids) == 0:
                    continue
                rack_map_sel[rack_sw_id] = node_ids

            node_ids = node_ids_sel
            rack_map = rack_map_sel

        else:
            n_endpoints = len(node_ids)

        node_ids_map = {}
        for i in range(len(node_ids)):
            node_ids_map[node_ids[i]] = i

        rack_map_adjusted = {}
        for rack_sw_id in rack_map:
            rack_nodes = []
            for node_id in rack_map[rack_sw_id]:
                rack_nodes.append(node_ids_map[node_id])
            rack_map_adjusted['rack_sw_' + str(rack_sw_id)] = rack_nodes

        ep_capacity = 1e10

        network = tpg.gen_arbitrary_network(num_eps=n_endpoints, ep_capacity=ep_capacity, num_channels=1)
                                            #racks_dict=rack_map_adjusted)
        print(f"generating flow_size_dist...")
        flow_size_dist = self.get_val_dist(self.params_flow_size_dist[workload_type])

        print(f"generating interarrival_time_dist...")
        interarrival_time_dist = self.get_val_dist(self.params_iat_dist[workload_type])

        print(f"generating node_dist...")
        endpoints = network.graph['endpoints']
        num_skewed_nodes = int(np.ceil(0.2 * len(endpoints)))
        skewed_node_probs = [0.2, 0.55]
        node_dist = tpg.gen_multimodal_node_dist(eps=endpoints, num_skewed_nodes=num_skewed_nodes,
                                                 skewed_node_probs=skewed_node_probs, show_fig=False)
        #node_dist = tpg.gen_multimodal_node_dist(eps=endpoints, num_skewed_nodes=1, show_fig=False)

        network_load_config = {'network_rate_capacity': network.graph['max_nw_capacity'],
                               'ep_link_capacity': network.graph['ep_link_capacity'],
                               'target_load_fraction': 0.1}

        print(f"creating demand data...")
        flow_centric_demand_data = tpg.create_demand_data(eps=endpoints, node_dist=node_dist,
                                                          flow_size_dist=flow_size_dist,
                                                          min_num_demands=num_demands,
                                                          jensen_shannon_distance_threshold=None,
                                                          interarrival_time_dist=interarrival_time_dist,
                                                          network_load_config=network_load_config,
                                                          auto_node_dist_correction=True)

        self.save_flow_data_to_csv(flow_centric_demand_data, node_ids, self.log_prefix + filename)
        print(f"CSV file is generated at {filename}")

    def gen_flows_for_ns3_incast(self, filename, num_eps=10,
                                 num_bytes_per_flow_min=1e6, num_bytes_per_flow_max=1e7):
        node_ids, rack_map = self.read_ep_ids_list_from_csv(self.log_prefix + "ep-ids")

        node_ids_sel = random.choices(node_ids, k=num_eps)
        receiver_idx = random.randint(0, num_eps - 1)
        receiver = node_ids_sel[receiver_idx]
        flows = []
        i = 0
        for sender in node_ids_sel:
            if sender == receiver:
                continue
            event_time = round(random.uniform(0, 0.0001), 10)
            flow_size = int(random.uniform(num_bytes_per_flow_min, num_bytes_per_flow_max))
            flows.append({
                "index": i,
                "flow_id": f"flow_{i}",
                "event_time": event_time,
                "sn": sender,
                "dn": receiver,
                "flow_size": flow_size
            })
            i += 1

        self._write_csv(self.log_prefix + filename, flows)

    def gen_flows_for_ns3_incast_rack2rack(self, filename, num_racks=4,
                                 num_bytes_per_flow_min=1e6, num_bytes_per_flow_max=1e7,
                                           fixed_event_time=False, fixed_flow_size=False):
        node_ids, rack_map = self.read_ep_ids_list_from_csv(self.log_prefix + "ep-ids")

        racks_ids = list(rack_map.keys())
        racks_ids_sel = random.sample(racks_ids, num_racks)
        rack_receiver_idx = random.randint(0, num_racks - 1)
        rack_receiver_id = racks_ids_sel[rack_receiver_idx]
        node_ids_in_racks_receiver = rack_map[rack_receiver_id]
        i = 0
        flows = []
        for rack_id in racks_ids_sel:
            if rack_id == rack_receiver_id:
                continue
            sender = rack_map[rack_id][0]
            receiver = random.choice(node_ids_in_racks_receiver)

            if fixed_event_time:
                event_time = 0
            else:
                event_time = round(random.uniform(0, 0.0001), 10)

            if fixed_flow_size:
                flow_size = int(num_bytes_per_flow_min)
            else:
                flow_size = int(random.uniform(num_bytes_per_flow_min, num_bytes_per_flow_max))
            flows.append({
                "index": i,
                "flow_id": f"flow_{i}",
                "event_time": event_time,
                "sn": sender,
                "dn": receiver,
                "flow_size": flow_size
            })

            i += 1

        self._write_csv(self.log_prefix + filename, flows)


    def gen_flows_for_ns3_alltoall(self, filename, num_eps=10,
                                   num_bytes_per_flow_min=1e6, num_bytes_per_flow_max=1e7):
        node_ids, rack_map = self.read_ep_ids_list_from_csv(self.log_prefix + "ep-ids")
        node_ids_sel = random.choices(node_ids, k=num_eps)

        flows = []
        index = 0
        for src in range(num_eps):
            for dst in range(num_eps):
                if src == dst:
                    continue
                event_time = round(random.uniform(0, 0.0001), 10)
                flow_size = int(random.uniform(num_bytes_per_flow_min, num_bytes_per_flow_max))

                flows.append({
                    "index": index,
                    "flow_id": f"flow_{index}",
                    "event_time": event_time,
                    "sn": node_ids_sel[src],
                    "dn": node_ids_sel[dst],
                    "flow_size": flow_size
                })
                index += 1

        self._write_csv(self.log_prefix + filename, flows)

    def _write_csv(self, filename, flows):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["index", "flow_id", "event_time", "sn", "dn", "flow_size"])
            writer.writeheader()
            for flow in flows:
                writer.writerow(flow)