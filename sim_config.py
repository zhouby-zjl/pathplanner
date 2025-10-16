from dcn_networks import dcn_network, switch
from controller import controller
import random
from rib import route_table
from utils import utils
from controller import controller
import itertools
import csv

class sim_config:
    @staticmethod
    def save_topo(net : dcn_network, file_path):
        fc_str = ""
        fc_str += f"nodes:\n{len(net.G.nodes())}\nep_ids:\n"
        ep_ids = []
        for ep in net.endpoints:
            ep_ids.append(str(ep.node_id))
        fc_str += ",".join(ep_ids)
        fc_str += "\nedges:\n"
        edge_lines = []
        for e in net.G.edges:
            edge_lines.append(f"{e[0]},{e[1]},{net.get_edge_addr(e[0], e[1])},{net.get_edge_addr(e[1], e[0])}")
        fc_str += "\n".join(edge_lines)

        with open(file_path, 'w') as file:
            file.write(fc_str)
            file.close()

    @staticmethod
    def generate_ip_addresses(subnet_a_addr, n_addr):
        ip_addresses = []
        subnet_base = subnet_a_addr << 24
        start_ip = subnet_base + 1

        current_ip = start_ip
        while len(ip_addresses) < n_addr:
            octets = [(current_ip >> (8 * i)) & 0xFF for i in range(4)][::-1]
            ip_str = '.'.join(map(str, octets))

            if current_ip & 0xFF != 0 and current_ip & 0xFF != 255:
                ip_addresses.append(ip_str)

            current_ip += 1
            if (current_ip & 0xFF000000) != subnet_base:
                break

        return ip_addresses

    @staticmethod
    def save_as_simai_topo(net: dcn_network, file_path: str, file_path_mapping: str, file_path_server_addresses: str,
                           gpus_per_server=8, gpu_type_str="H100", band_map={}):
        link_lines = []

        # 获取节点ID
        endpoint_ids = net.get_ep_ids()
        switch_ids = net.get_switch_ids()
        server_addresses = []

        # NVSwitch数量和分组
        n_eps = len(endpoint_ids)
        n_full_groups = n_eps // gpus_per_server
        remainder = n_eps % gpus_per_server
        nvswitch_count = n_full_groups + (1 if remainder > 0 else 0)

        # 新的 SimAI ID 分配（0开始递增）：ep -> nv -> sw
        simai_id_counter = 0
        id_map = {}

        # 分配 SimAI ID 给 endpoint
        ep_id_simai = []
        for ep_id in endpoint_ids:
            id_map[ep_id] = simai_id_counter
            ep_id_simai.append(simai_id_counter)
            simai_id_counter += 1

        # 分配 SimAI ID 给 NVSwitch（自动生成）
        nv_ids_net = list(range(max(net.G.nodes) + 1, max(net.G.nodes) + 1 + nvswitch_count))
        nv_id_simai = []
        for nv_id in nv_ids_net:
            id_map[nv_id] = simai_id_counter
            nv_id_simai.append(simai_id_counter)
            simai_id_counter += 1

        # 分配 SimAI ID 给 Switch
        sw_id_simai = []
        for sw_id in switch_ids:
            id_map[sw_id] = simai_id_counter
            sw_id_simai.append(simai_id_counter)
            simai_id_counter += 1

        # 构建节点间连接（只包含net拓扑中的边）
        for u, v in net.G.edges:
            if u in id_map and v in id_map:
                if (u, v) in band_map:
                    band = band_map[(u, v)]
                else:
                    band = 200
                line = f"{id_map[u]} {id_map[v]} {band}Gbps 0.0005ms 0"
                link_lines.append(line)
                if u in endpoint_ids:
                    server_addresses.append([id_map[u], net.get_edge_addr(u, v)])
                if v in endpoint_ids:
                    server_addresses.append([id_map[v], net.get_edge_addr(v, u)])

        #ip_addr_nvswitch_all = sim_config.generate_ip_addresses(9, nvswitch_count)

        # 构建节点内连接（每组ep连接一个NVSwitch）
        i = 0
        for group_id in range(nvswitch_count):
            nv_id_net = nv_ids_net[group_id]
            nv_id_new = id_map[nv_id_net]
            ep_start = group_id * gpus_per_server
            ep_end = min(ep_start + gpus_per_server, n_eps)
            for i in range(ep_start, ep_end):
                ep_id_net = endpoint_ids[i]
                ep_id_new = id_map[ep_id_net]
                line = f"{ep_id_new} {nv_id_new} 2880Gbps 0.000025ms 0"
                link_lines.append(line)
                i += 1

            server_addresses.append([nv_id_new, net.ip_addresses[net.n_ep + group_id]])


        # 构造文件头部
        n_nodes_total = len(id_map)
        header_line = f"{n_nodes_total} {gpus_per_server} {nvswitch_count} {len(switch_ids)} {len(link_lines)} {gpu_type_str}"
        id_order_line = " ".join(map(str, nv_id_simai + sw_id_simai))

        # 写拓扑文件
        with open(file_path, "w") as f:
            f.write(header_line + "\n")
            f.write(id_order_line + "\n")
            f.write("\n".join(link_lines) + "\n")

        id_rmap = {}
        # 写翻译表 CSV 文件：original_id,new_id
        with open(file_path_mapping, "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            for original_id, simai_id in sorted(id_map.items(), key=lambda x: x[1]):
                id_rmap[simai_id] = original_id
                writer.writerow([original_id, simai_id])

        with open(file_path_server_addresses, "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            for sa in server_addresses:
                writer.writerow(sa)

        return id_map, id_rmap, server_addresses

    @staticmethod
    def save_node_ecmp_seeds_and_permutations(net : dcn_network, file_path_seeds, file_path_permutations, id_map=None):
        all_seeds_str = ""
        all_permutations_str = ""
        for node_id in net.G.nodes:
            node = net.get_node_obj(node_id)
            if not isinstance(node, switch):
                continue
            sw : switch = node
            seed_str = ",".join([str(s) for s in sw.seed])
            _node_id = node_id if id_map is None else id_map[node_id]
            all_seeds_str += str(_node_id) + ":" + seed_str + "\n"
            permutation_str = ",".join([str(s) for s in sw.permutation])
            all_permutations_str += str(_node_id) + ":" + permutation_str + "\n"

        with open(file_path_seeds, 'w') as file:
            file.write(all_seeds_str)
            file.close()

        with open(file_path_permutations, 'w') as file:
            file.write(all_permutations_str)
            file.close()

    @staticmethod
    def load_node_ecmp_seeds_and_permutations(net: dcn_network, file_path_seeds, file_path_permutations, id_rmap=None):
        # 读取 seed 文件
        with open(file_path_seeds, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(":")
                mapped_id = int(parts[0])
                node_id = mapped_id if id_rmap is None else id_rmap[mapped_id]
                seed_list = list(map(int, parts[1].split(",")))
                node = net.get_node_obj(node_id)
                if isinstance(node, switch):
                    node.seed = seed_list

        # 读取 permutation 文件
        with open(file_path_permutations, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(":")
                mapped_id = int(parts[0])
                node_id = mapped_id if id_rmap is None else id_rmap[mapped_id]
                permutation_list = list(map(int, parts[1].split(",")))
                node = net.get_node_obj(node_id)
                if isinstance(node, switch):
                    node.permutation = permutation_list

    @staticmethod
    def save_routes(net : dcn_network, file_path):
        all_routes_str = ""
        for node_id in net.G.nodes:
            node = net.get_node_obj(node_id)
            routes : route_table = node.routes
            routes_node = []
            for dst in routes.routes_hash:
                dst_str = utils.getIpInStr(dst)
                ports = list(routes.routes_hash[dst].keys())
                routes_node.append(dst_str + "," + '-'.join([str(p) for p in ports]))
            routes_node_str = str(node_id) + ":" + "|".join([str(n) for n in routes_node])
            all_routes_str += routes_node_str + "\n"

        with open(file_path, 'w') as file:
            file.write(all_routes_str)
            file.close()

    @staticmethod
    def save_port_id_info(net : dcn_network, file_path):
        ports_str = ""
        for e in net.G.edges:
            port_a = net.get_port_id(e[0], e[1])
            port_b = net.get_port_id(e[1], e[0])
            ports_str += f"{e[0]},{e[1]},{port_a}\n{e[1]},{e[0]},{port_b}\n"

        with open(file_path, 'w') as file:
            file.write(ports_str)
            file.close()

    @staticmethod
    def save_paths_all(net : dcn_network, file_path):
        paths_all_str = ""
        for n_s in net.paths_all.keys():
            for n_d in net.paths_all[n_s].keys():
                paths_all_str += f"{n_s},{n_d},{'#'.join(['-'.join([str(x) for x in p]) for p in net.paths_all[n_s][n_d]])}\n"

        with open(file_path, 'w') as file:
            file.write(paths_all_str)
            file.close()

    @staticmethod
    def save_src_dst_flows(src_dst_flows, file_path):
        s = ""
        for flow in src_dst_flows:
            s += f"{flow.idx},{flow.node_src_id},{flow.node_dst_id},{flow.port_src},{flow.port_dst}\n"

        with open(file_path, 'w') as file:
            file.write(s)
            file.close()

    @staticmethod
    def gen_random_src_dst_flows_for_incasting(net : dcn_network, m):
        ep_ids = [ep.node_id for ep in net.endpoints]
        if m >= len(ep_ids):
            m = len(ep_ids) - 1
        ids_can = random.sample(ep_ids, m + 1)
        id_dst = random.sample(ids_can, 1)[0]
        ports_src = random.sample(range(0, 2**16), m)
        ports_dst = random.sample(range(0, 2**16), m)
        flows = []
        i = 0
        for id_src in ids_can:
            if id_src == id_dst:
                continue
            f = controller.flow_info()
            f.node_src_id = id_src
            f.node_dst_id = id_dst
            f.port_src = ports_src[i]
            f.port_dst = ports_dst[i]
            flows.append(f)
            i += 1
        return flows

    @staticmethod
    def pick_m_different_pairs(node_ids, m):
        pairs = set()
        n = len(node_ids)

        if m > (n * (n - 1)) // 2:
            m = (n * (n - 1)) // 2
            print(f"change the number of different pairs to pick to {m}")
            #raise ValueError("m is too large for the number of unique pairs possible.")

        while len(pairs) < m:
            i, j = random.sample(range(n), 2)
            pair = tuple(sorted((node_ids[i], node_ids[j])))
            pairs.add(pair)

        return list(pairs)

    @staticmethod
    def pick_m_different_pairs_from_viable_paths(paths_all, m):
        all_pairs = []
        for src in paths_all.keys():
            for dst in paths_all[src].keys():
                if (src, dst) not in all_pairs:
                    all_pairs.append((src, dst))

        idx_all = random.sample(range(0, len(all_pairs)), m)
        ports_src = random.sample(range(0, 2**16), m)
        ports_dst = random.sample(range(0, 2**16), m)

        i = 0
        flows = []
        for idx in idx_all:
            f = controller.flow_info()
            f.node_src_id = all_pairs[idx][0]
            f.node_dst_id = all_pairs[idx][1]
            f.port_src = ports_src[i]
            f.port_dst = ports_dst[i]
            flows.append(f)
            i += 1
        return flows

    @staticmethod
    def gen_random_src_dst_flows_for_one_to_one(net : dcn_network, m):
        ep_ids = [ep.node_id for ep in net.endpoints]
        pairs_selected = sim_config.pick_m_different_pairs(ep_ids, m)
        n_pairs = len(pairs_selected)

        ports_src = random.sample(range(0, 2**16), m)
        ports_dst = random.sample(range(0, 2**16), m)

        flows = []
        for i in range(n_pairs):
            f = controller.flow_info()
            f.node_src_id = pairs_selected[i][0]
            f.node_dst_id = pairs_selected[i][1]
            f.port_src = ports_src[i]
            f.port_dst = ports_dst[i]
            flows.append(f)

        return flows

    @staticmethod
    def gen_src_dst_flows_with_random_ports_for_one_to_one(sn, dn, n_pairs):
        ports_src = random.sample(range(0, 2 ** 16), n_pairs)
        ports_dst = random.sample(range(0, 2 ** 16), n_pairs)
        flows = []
        for i in range(n_pairs):
            f = controller.flow_info()
            f.node_src_id = sn[i]
            f.node_dst_id = dn[i]
            f.port_src = ports_src[i]
            f.port_dst = ports_dst[i]
            flows.append(f)

        return flows

    @staticmethod
    def gen_src_dst_flows_with_random_ports_allowing_duplicated_for_one_to_one(idxes, sn, dn, event_time, flow_size, n_pairs):
        ports_src = random.choices(range(0, 2 ** 16), k=n_pairs)
        ports_dst = random.choices(range(0, 2 ** 16), k=n_pairs)
        flows = []
        for i in range(n_pairs):
            f = controller.flow_info()
            f.idx = idxes[i]
            f.node_src_id = sn[i]
            f.node_dst_id = dn[i]
            f.port_src = ports_src[i]
            f.port_dst = ports_dst[i]
            f.event_time = event_time[i]
            f.flow_size = flow_size[i]
            flows.append(f)

        return flows


    @staticmethod
    def write_pairs_with_flow_schedules(src_dst_pairs, pg, packet_count, start_time, file_path):
        flows_str = "src,dst,sport,dport,pg,maxPacketCount,startTime\n"
        for f in src_dst_pairs:
            flows_str += f"{f.node_src_id},{f.node_dst_id},{f.port_src},{f.port_dst},{pg},{packet_count},{start_time}\n"

        with open(file_path, 'w') as file:
            file.write(flows_str)
            file.close()

    @staticmethod
    def save_flow_and_path_info(flow_and_path_info, file_path):
        s = ""
        for [idx, node_src_id, node_dst_id, port_src, port_dst,
             valid_port_src, event_time, flow_size,
             path_ecmp, designated_path, path_with_valid_port_src] in flow_and_path_info:
            s += (f"{str(idx)},{str(node_src_id)},{str(node_dst_id)},{str(port_src)},{str(port_dst)},{str(valid_port_src)},"
                  f"{str(event_time)},{str(flow_size)},"
                  f"{'-'.join([str(x) for x in path_ecmp])},{'-'.join([str(x) for x in designated_path])},"
                  f"{'-'.join([str(x) for x in path_with_valid_port_src])}\n")

        with open(file_path, 'w') as file:
            file.write(s)
            file.close()

    @staticmethod
    def save_flow_and_path_info_by_idxes(flow_and_path_info, idxes, file_path):
        s = ""
        for idx in idxes:
            [node_src_id, node_dst_id, port_src, port_dst, valid_port_src,
             path_ecmp, designated_path, path_with_valid_port_src] = flow_and_path_info[idx]
            s += (f"{str(node_src_id)},{str(node_dst_id)},{str(port_src)},{str(port_dst)},{str(valid_port_src)},"
                  f"{'-'.join([str(x) for x in path_ecmp])},{'-'.join([str(x) for x in designated_path])},"
                  f"{'-'.join([str(x) for x in path_with_valid_port_src])}\n")

        with open(file_path, 'w') as file:
            file.write(s)
            file.close()


    @staticmethod
    def load_flow_and_path_info(file_path):
        flow_and_path_info = []

        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(',')
                node_src_id = int(parts[0])
                node_dst_id = int(parts[1])
                port_src = int(parts[2])
                port_dst = int(parts[3])
                valid_port_src = int(parts[4])

                path_ecmp = [int(x) for x in parts[5].split('-')] if parts[5] else []
                designated_path = [int(x) for x in parts[6].split('-')] if parts[6] else []
                path_with_valid_port_src = [int(x) for x in parts[7].split('-')] if parts[7] else []

                flow_and_path_info.append([
                    node_src_id, node_dst_id, port_src, port_dst,
                    valid_port_src,
                    path_ecmp, designated_path, path_with_valid_port_src
                ])

        return flow_and_path_info

    @staticmethod
    def save_sim_config(net : dcn_network, src_dst_flows, prefix):
        sim_config.save_topo(net, prefix + 'topo')
        sim_config.save_port_id_info(net, prefix + "ports")
        sim_config.save_node_ecmp_seeds_and_permutations(net, prefix + "seeds", prefix + "permutations")
        sim_config.save_routes(net, prefix + "routes")
        sim_config.save_paths_all(net, prefix + "paths")
        sim_config.save_src_dst_flows(src_dst_flows, prefix + "flows")

    @staticmethod
    def save_topo_config(net : dcn_network, prefix):
        sim_config.save_topo(net, prefix + 'topo')
        sim_config.save_port_id_info(net, prefix + "ports")
        sim_config.save_node_ecmp_seeds_and_permutations(net, prefix + "seeds", prefix + "permutations")
