#!/usr/bin/python3

import random
import time

from dcn_networks import dcn_network, sl_network, fattree_network, aspen_trees_network, switch, endpoint, packet, EcmpHashAlgorithm
from controller import controller
from sim_config import sim_config
from typing import List
from tracer import tracer
import os
import config
from tqdm import tqdm
import argparse
import psutil
import subprocess
import csv
from flows_generator import flows_generator
from data_plane_ns3_sim import data_plane_ns3_sim
from path_generator_with_ecmp_hash_linearity import path_generator_with_ecmp_hash_linearity
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from api_server import RestApiServer
import threading

random.seed(10)
simai_dir = '/home/zby/dcqcn-net-control/exp/'
flows_generator_for_ns3 = '/home/zby/dcqcn-net-control/traffic-gen/dcn-traffic-generator/flows-generator-for-ns3.py'

tracer.enable_trace_time = True
tracer.enable_trace_space = False
tracer.enable_trace_iterations = True
tracer.enable_trace_path_trace_delay = True

SIM_TYPE_INCAST = 0
SIM_TYPE_ONE_TO_ONE = 1
SIM_TYPE_PICK_FROM_VIALBLE_PATHS = 2

src_dst_flows_prev = None
net_prev = None
net_desc_prev = None
port_srcs_to_retry_all_prev = None

def get_phy_cpu_cores():
    return psutil.cpu_count(logical=False)

def reset_prev_flows_and_net():
    global src_dst_flows_prev, net_prev, net_desc_prev, port_srcs_to_retry_all_prev
    src_dst_flows_prev = None
    net_prev = None
    net_desc_prev = None
    port_srcs_to_retry_all_prev = None

def gen_port_srcs_to_retry_for_flows(max_retrying_time, n_flows):
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

def run_sim_for_a_net(log_prefix, net, sim_type, n_pairs,
                      max_retrying_time=3,
                      max_paths=-1, use_rand_path=False,
                      use_progress_bar=False, keep_silience=False,
                      batch_size=100,
                      num_cpus=None, net_desc=None,
                      only_use_sht=False,
                      reuse_prev_flows=False,
                      reuse_prev_port_srcs_to_retry=False,
                      use_linearity_routes=False, m_paths=None,
                      paths_installed=False):
    global src_dst_flows_prev, port_srcs_to_retry_all_prev
    orig_keep_silience = config.KEEP_SILENCE
    config.KEEP_SILENCE = keep_silience
    tracer.clear()

    if not reuse_prev_flows or src_dst_flows_prev is None:
        src_dst_flows : List[controller.flow_info] = None
        if sim_type == SIM_TYPE_INCAST:
            src_dst_flows = sim_config.gen_random_src_dst_flows_for_incasting(net, n_pairs)
        elif sim_type == SIM_TYPE_ONE_TO_ONE:
            src_dst_flows = sim_config.gen_random_src_dst_flows_for_one_to_one(net, n_pairs)
        elif sim_type == SIM_TYPE_PICK_FROM_VIALBLE_PATHS:
            src_dst_flows = sim_config.pick_m_different_pairs_from_viable_paths(net.paths_all, n_pairs)
        else:
            return

        if not paths_installed:
            if not use_linearity_routes:
                net.generate_and_install_paths(src_dst_flows, max_paths, use_rand_path)
            else:
                net.generate_and_install_paths_ensuring_ecmp_linearity(src_dst_flows, m_paths)
        src_dst_flows_prev = src_dst_flows
    else:
        src_dst_flows = src_dst_flows_prev

    #print(f"src_dst_flows: \n{src_dst_flows}\n")

    port_srcs_to_retry_all = []
    if reuse_prev_port_srcs_to_retry:
        if port_srcs_to_retry_all_prev is None:
            port_srcs_to_retry_all = gen_port_srcs_to_retry_for_flows(max_retrying_time, len(src_dst_flows))
            port_srcs_to_retry_all_prev = port_srcs_to_retry_all
        else:
            port_srcs_to_retry_all = port_srcs_to_retry_all_prev

    #print(f"port_srcs_to_retry_all: {port_srcs_to_retry_all}\n")

    ctl = controller(net, n_cpus=num_cpus, batch_size=batch_size)
    epr_cht, epr_sht = ctl.evaluate_o_delta_comb()

    orig_ecmp_paths_info = ctl.gen_orig_ecmp_paths(src_dst_flows)
    paths = [paths_info[1] for paths_info in orig_ecmp_paths_info]
    jr_orig = ctl.compute_link_jointness_ratio(paths)
    #print("============ orig_ecmp_paths_info =============")
    #print(orig_ecmp_paths_info)
    #print("===============================================")
    max_disjoint_paths = ctl.gen_max_disjoint_paths(src_dst_flows)
    md_path_vectors = []
    for src in max_disjoint_paths:
        for dst in max_disjoint_paths[src]:
            md_path_vectors.append(max_disjoint_paths[src][dst][0])
    jr_expected = ctl.compute_link_jointness_ratio(md_path_vectors)
    num_flows = len(src_dst_flows)
    valid_selection = 0
    paths_selected = []
    n_flows = len(src_dst_flows)
    i = 0
    progress_bar = None
    if use_progress_bar:
        config.KEEP_SILENCE = True
        progress_bar = tqdm(total=n_flows, desc="Processing", unit="flows")

    for flow in src_dst_flows:
        tracer.log(f"finding valid EP src port for flow {flow}")
        port_srcs_to_retry = port_srcs_to_retry_all[i] if reuse_prev_port_srcs_to_retry else []
        ep_port_src, succ, retrying_times = ctl.find_ep_port_src_uha(flow,
                                                                    max_disjoint_paths, max_retrying_time,
                                                                     only_use_sht,
                                                                     port_srcs_to_retry)

        i += 1
        custom_info = {'flows processed': f'{i + 1} / {n_flows}', 'max # CPUs': num_cpus}
        progress_bar.set_postfix(custom_info)

        if not succ:
            tracer.log(f"obtaining valid delta is failed for the flow {flow}")
        valid, path_fd, designated_path = ctl.validate_designated_flow(flow, ep_port_src, max_disjoint_paths)
        paths_selected.append(path_fd)
        tracer.log(f"validation for the flow {flow}: {valid}")
        if valid:
            valid_selection += 1

        tracer.retrying_times(retrying_times)

        if net_desc is not None:
            net_desc_str = ', '.join([k + ": " + str(net_desc[k]) for k in net_desc])
        else:
            net_desc_str = 'no desc'

        tracer.log(f"====> progression for ({net_desc_str}): {i} / {n_flows}")
        if use_progress_bar:
            progress_bar.update(1)

    pss = valid_selection / num_flows
    jr_selected = ctl.compute_link_jointness_ratio(paths_selected)
    n_endpoints = net.get_num_endpoints()
    n_switches = len(net.get_switch_ids())

    #tracer.log(f"dump exe time: \n{tracer.dump_time_csv()}")
    #tracer.log(f"dump exe space: \n{tracer.dump_space_csv()}")
    #tracer.log(f"dump iterations: \n{tracer.dump_iterations_csv()}")
    #tracer.log(f"path trace delays: \n{tracer.dump_path_trace_csv()}")
    tracer.dump_all_to_files(log_prefix)
    tracer.write_sim_stats(log_prefix, epr_cht, epr_sht, pss,
                           n_endpoints, n_switches, n_flows,
                           jr_orig, jr_expected, jr_selected,
                           net_desc)
    tracer.clear()

    config.KEEP_SILENCE = orig_keep_silience
    #clean_stop_flag_shared()


def run_test_for_fattree(log_dir, sim_type=SIM_TYPE_INCAST, n_pairs=10, batch_size=100,
                            keep_silience=True, num_cpus=None, 
                            k_port_per_switch=32,
                         max_retrying_time=3,
                         reuse_prev_flows=False,
                         reuse_prev_net=False,
                         reuse_prev_port_srcs_to_retry=False,
                         use_linearity_routes=False,
                         m_paths=None):
    global net_prev, net_desc_prev
    if not reuse_prev_net or net_prev is None or net_desc_prev is None:
        net = fattree_network(k=k_port_per_switch)
        net.build_topo()
        net_desc = {'topo': 'fattree', 'k': k_port_per_switch}
        if reuse_prev_net:
            net_prev = net
            net_desc_prev = net_desc
    else:
        net = net_prev
        net_desc = net_desc_prev

    run_sim_for_a_net(log_prefix=log_dir, net=net,
                      sim_type=sim_type, n_pairs=n_pairs,
                      max_retrying_time=max_retrying_time, max_paths=-1, use_rand_path=False,
                      use_progress_bar=True, keep_silience=keep_silience,
                      batch_size=batch_size,
                      num_cpus=num_cpus, net_desc=net_desc,
                      reuse_prev_flows=reuse_prev_flows,
                      reuse_prev_port_srcs_to_retry=reuse_prev_port_srcs_to_retry,
                      use_linearity_routes=use_linearity_routes,
                      m_paths=m_paths)


def run_test_for_spineleaf(log_dir, sim_type=SIM_TYPE_INCAST, n_pairs=10, batch_size=100,
                              keep_silience=True, num_cpus=None, only_use_sht=False,
                           n_spine_sw=4, n_leaf_sw=2, n_ep=4, n_ep_per_leaf_sw=2,
                           k_port_spine_sw=2, k_port_leaf_sw=2,
                           max_retrying_time=3,
                           reuse_prev_flows=False,
                           reuse_prev_net=False,
                           reuse_prev_port_srcs_to_retry=False,
                           use_linearity_routes=False,
                           m_paths=None):
    global net_prev, net_desc_prev
    if not reuse_prev_net or net_prev is None or net_desc_prev is None:
        net = sl_network(n_spine_sw=n_spine_sw, n_leaf_sw=n_leaf_sw, n_ep=n_ep,
                         n_ep_per_leaf_sw=n_ep_per_leaf_sw)
        net.build_topo()
        net_desc = {'topo': 'fattree', 'ks': k_port_spine_sw, 'kl': k_port_leaf_sw}
        if reuse_prev_net:
            net_prev = net
            net_desc_prev = net_desc
    else:
        net = net_prev
        net_desc = net_desc_prev

    run_sim_for_a_net(log_prefix=log_dir, net=net,
                      sim_type=sim_type, n_pairs=n_pairs,
                      max_retrying_time=max_retrying_time, max_paths=-1, use_rand_path=False,
                      use_progress_bar=True, keep_silience=keep_silience,
                      batch_size=batch_size,
                      num_cpus=num_cpus, net_desc=net_desc,
                      only_use_sht=only_use_sht,
                      reuse_prev_flows=reuse_prev_flows,
                      reuse_prev_port_srcs_to_retry=reuse_prev_port_srcs_to_retry,
                      use_linearity_routes=use_linearity_routes,
                      m_paths=m_paths)

def run_test_for_aspen(log_dir, sim_type=SIM_TYPE_INCAST, n_pairs=10, batch_size=100,
                       keep_silience=True, num_cpus=None, 
                       n_levels=3, k_port_per_switch=4,
                       max_retrying_time=3,
                       reuse_prev_flows=False,
                       reuse_prev_net=False,
                       reuse_prev_port_srcs_to_retry=False,
                       use_linearity_routes=False,
                       m_paths=None):
    global net_prev, net_desc_prev
    if not reuse_prev_net or net_prev is None or net_desc_prev is None:
        c_factored = [1] * n_levels
        net = aspen_trees_network(n_levels=n_levels, k_port_per_switch=k_port_per_switch, c_factored=c_factored)
        net.build_topo()
        net_desc = {'topo': 'aspen', 'n': n_levels, 'k': k_port_per_switch,
                    'c_factored': ','.join([str(c_i) for c_i in c_factored])}
        if reuse_prev_net:
            net_prev = net
            net_desc_prev = net_desc
    else:
        net = net_prev
        net_desc = net_desc_prev

    run_sim_for_a_net(log_prefix=log_dir, net=net,
                      sim_type=sim_type, n_pairs=n_pairs,
                      max_retrying_time=max_retrying_time, max_paths=-1, use_rand_path=False,
                      use_progress_bar=True, keep_silience=keep_silience,
                      batch_size=batch_size,
                      num_cpus=num_cpus, net_desc=net_desc,
                      reuse_prev_flows=reuse_prev_flows,
                      reuse_prev_port_srcs_to_retry=reuse_prev_port_srcs_to_retry,
                      use_linearity_routes=use_linearity_routes,
                      m_paths=m_paths)

# ========================= VARYING TOPOLOGIES ===================================
def run_batch_tests_for_aspen(log_dir_prefix, sim_type=SIM_TYPE_INCAST, n_pairs=10, batch_size=100,
                              renew=False, keep_silience=True, num_cpus=None, use_linearity_routes=False,
                              m_paths=None):
    n_levels_all = [3]
    k_port_per_switch_all = [4]
    #n_levels_all = [4]
    #k_port_per_switch_all = [16]
    total_runs = len(n_levels_all) * len(k_port_per_switch_all)

    i = 0
    for n_levels in n_levels_all:
        for k_port_per_switch in k_port_per_switch_all:
            i += 1
            print(f"====> progression: Computing for {i} / {total_runs} with n: {n_levels}, k: {k_port_per_switch}....")
            log_dir = log_dir_prefix + 'aspen-' + ('incast' if sim_type == SIM_TYPE_INCAST else 'unicast') + \
                      '-n-' + str(n_levels) + '-k-' + str(k_port_per_switch) + '/'
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
            if not renew and os.path.exists(log_dir):
                if os.path.exists(log_dir + 'stats') and os.path.getsize(log_dir + 'stats') > 0:
                    print("---> automatically jump the old files")
                    continue

            run_test_for_aspen(log_dir, sim_type=sim_type, n_pairs=n_pairs, batch_size=batch_size,
                               keep_silience=keep_silience, num_cpus=num_cpus,
                               n_levels=n_levels, k_port_per_switch=k_port_per_switch,
                               reuse_prev_flows=False,
                               reuse_prev_net=False,
                               reuse_prev_port_srcs_to_retry=False,
                               max_retrying_time=5,
                               use_linearity_routes=use_linearity_routes,
                               m_paths=m_paths)
            print(f"====> progression: Done for {i} / {total_runs}")

def run_batch_tests_for_aspen_varying_retries(log_dir_prefix, sim_type=SIM_TYPE_INCAST, n_pairs=10, batch_size=100,
                              renew=False, keep_silience=True, num_cpus=None):
    n_levels = 5
    k_port_per_switch = 8
    max_retrying_time_all = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    total_runs = len(max_retrying_time_all)

    i = 0
    for max_retrying_time in max_retrying_time_all:
        i += 1
        print(f"====> progression: Computing for {i} / {total_runs} with n: {n_levels}, k: {k_port_per_switch}....")
        log_dir = log_dir_prefix + 'aspen-' + ('incast' if sim_type == SIM_TYPE_INCAST else 'unicast') + \
                  '-retries-' + str(max_retrying_time) + '/'
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        if not renew and os.path.exists(log_dir):
            if os.path.exists(log_dir + 'stats') and os.path.getsize(log_dir + 'stats') > 0:
                print("---> automatically jump the old files")
                continue

        run_test_for_aspen(log_dir, sim_type=sim_type, n_pairs=n_pairs, batch_size=batch_size,
                           keep_silience=keep_silience, num_cpus=num_cpus,
                           n_levels=n_levels, k_port_per_switch=k_port_per_switch,
                           reuse_prev_flows=False,
                           reuse_prev_net=False,
                           reuse_prev_port_srcs_to_retry=False,
                           max_retrying_time=max_retrying_time,
                           m_paths=m_paths)
        print(f"====> progression: Done for {i} / {total_runs}")

def run_batch_tests_for_aspen_varying_hashing(log_dir_prefix, sim_type=SIM_TYPE_INCAST, n_pairs=10, batch_size=100,
                                                renew=False, keep_silience=True, num_cpus=None,
                                                use_linearity_routes=False, m_paths=None):
    MIXED = -1
    hashing_policies = [[EcmpHashAlgorithm.CRC, 'crc'], [EcmpHashAlgorithm.XOR, 'xor'],
                        [EcmpHashAlgorithm.CRC_32LO, 'crc_32lo'], [EcmpHashAlgorithm.CRC_32HI, 'crc_32hi'],
                        [EcmpHashAlgorithm.CRC_CCITT, 'crc_ccitt'], [EcmpHashAlgorithm.CRC_XOR, 'crc_xor'],
                        [MIXED, 'mixed']]
    n_levels = 5
    c_factored = [1] * n_levels
    k_port_per_switch = 8
    n_hp = len(hashing_policies)

    max_retrying_time_all = [5, 10]
    total = n_hp * len(max_retrying_time_all)

    i = 0
    net_prev = None
    net_desc_prev = None
    for [hashing_type, hashing_name] in hashing_policies:
        for max_retrying_time in max_retrying_time_all:
            i += 1
            print(f"====> progression: Computing for {i} / {total} with hashing policy: {hashing_name}....")

            if net_prev is None or net_desc_prev is None:
                net = aspen_trees_network(n_levels=n_levels, k_port_per_switch=k_port_per_switch, c_factored=c_factored)
                net.build_topo()
                net_desc = {'topo': 'aspen', 'n': n_levels, 'k': k_port_per_switch,
                            'c_factored': ','.join([str(c_i) for c_i in c_factored])}

                net_prev = net
                net_desc_prev = net_desc
            else:
                net = net_prev
                net_desc = net_desc_prev

            sw_ids = net.get_switch_ids()
            for sw_id in sw_ids:
                sw = net.get_node_obj(sw_id)
                if hashing_type != MIXED:
                    sw.hashing_alg = hashing_type
                else:
                    ht_sel = random.randint(0, n_hp - 2)
                    sw.hashing_alg = hashing_policies[ht_sel][0]

            log_dir = log_dir_prefix + 'aspen-' + ('incast' if sim_type == SIM_TYPE_INCAST else 'unicast') + \
                      '-hashing-' + hashing_name + '-retrying-' + str(max_retrying_time) + '/'

            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
            if not renew and os.path.exists(log_dir):
                if os.path.exists(log_dir + 'stats') and os.path.getsize(log_dir + 'stats') > 0:
                    print("---> automatically jump the old files")
                    continue

            run_sim_for_a_net(log_prefix=log_dir, net=net,
                              sim_type=sim_type, n_pairs=n_pairs,
                              max_retrying_time=max_retrying_time, max_paths=-1, use_rand_path=False,
                              use_progress_bar=True, keep_silience=keep_silience,
                              batch_size=batch_size,
                              num_cpus=num_cpus, net_desc=net_desc,
                              reuse_prev_flows=True,
                              reuse_prev_port_srcs_to_retry=True,
                              use_linearity_routes=use_linearity_routes,
                              m_paths=m_paths)

            print(f"====> progression: Done for {i} / {total}")

def run_batch_tests_for_fattree(log_dir_prefix, sim_type=SIM_TYPE_INCAST, n_pairs=10, batch_size=100,
                              renew=False, keep_silience=True, num_cpus=None,
                                use_linearity_routes=False, m_paths=None):
    #k_port_per_switch_all = [4, 8, 16, 32, 64]
    k_port_per_switch_all = [32]
    total_runs = len(k_port_per_switch_all)
    i = 0
    for k_port_per_switch in k_port_per_switch_all:
        i += 1
        print(f"====> progression: Computing for {i} / {total_runs} with k: {k_port_per_switch}....")
        log_dir = log_dir_prefix + 'fattree-' + ('incast' if sim_type == SIM_TYPE_INCAST else 'unicast') + \
                  '-k-' + str(k_port_per_switch) + '/'
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        if not renew and os.path.exists(log_dir):
            if os.path.exists(log_dir + 'stats') and os.path.getsize(log_dir + 'stats') > 0:
                print("---> automatically jump the old files")
                continue
        run_test_for_fattree(log_dir, sim_type=sim_type, n_pairs=n_pairs,
                             batch_size=batch_size,
                             keep_silience=keep_silience, num_cpus=num_cpus,
                             k_port_per_switch=k_port_per_switch,
                             reuse_prev_flows=False,
                             reuse_prev_net=False,
                             reuse_prev_port_srcs_to_retry=False,
                             use_linearity_routes=use_linearity_routes,
                             m_paths=m_paths)

        print(f"====> progression: Done for {i} / {total_runs}")


def run_batch_tests_for_spineleaf(log_dir_prefix, sim_type=SIM_TYPE_INCAST, n_pairs=10, batch_size=100,
                              renew=False, keep_silience=True, num_cpus=None, only_use_sht=False):
    k_port_spine_sw_all = [4, 8, 16, 32, 64, 128]
    k_port_leaf_sw_all = [k * 2 for k in k_port_spine_sw_all]
    band_ratio_spine_to_leaf = 2
    num_spine_sw_all = [int(k_ls / (2 * band_ratio_spine_to_leaf)) for k_ls in k_port_leaf_sw_all]
    num_leaf_sw_all = k_port_spine_sw_all
    n_ep_per_leaf_sw_all = [int(k_ls / 2) for k_ls in k_port_leaf_sw_all]
    n_ep_all = [int(k_port_spine_sw_all[i] * k_port_leaf_sw_all[i] / 2) for i in range(len(k_port_spine_sw_all))]
    total_runs = len(k_port_spine_sw_all)

    for i in range(len(k_port_spine_sw_all)):
        print(f"====> progression: Computing for {i} / {total_runs} with k spine: {k_port_spine_sw_all[i]}, "
              f"k leaf: {k_port_leaf_sw_all[i]}....")
        case_name = 'spineleaf' if not only_use_sht else 'spineleafrecap-'
        log_dir = log_dir_prefix + case_name + '-' + ('incast' if sim_type == SIM_TYPE_INCAST else 'unicast') + \
                  '-ks-' + str(k_port_spine_sw_all[i]) + '-kl-' + str(k_port_leaf_sw_all[i]) + '/'
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        if not renew and os.path.exists(log_dir):
            if os.path.exists(log_dir + 'stats') and os.path.getsize(log_dir + 'stats') > 0:
                print("---> automatically jump the old files")
                continue

        run_test_for_spineleaf(log_dir, sim_type=sim_type, n_pairs=n_pairs, batch_size=batch_size,
                               keep_silience=keep_silience, num_cpus=num_cpus, only_use_sht=only_use_sht,
                               n_spine_sw=num_spine_sw_all[i], n_leaf_sw=num_leaf_sw_all[i], n_ep=n_ep_all[i],
                               n_ep_per_leaf_sw=n_ep_per_leaf_sw_all[i],
                               k_port_spine_sw=k_port_spine_sw_all[i], k_port_leaf_sw=k_port_leaf_sw_all[i],
                               reuse_prev_flows=False,
                               reuse_prev_net=False,
                               reuse_prev_port_srcs_to_retry=False)

        print(f"====> progression: Done for {i} / {total_runs}")

# ===================== VARYING PARALLELISM =========================
def run_parallel_batch_tests_for_fattree(log_dir_prefix, sim_type=SIM_TYPE_INCAST, n_pairs=10,
                                         batch_size=100,
                                         renew=False, keep_silience=True, 
                                         k_port_per_switch=32):
    reset_prev_flows_and_net()
    tracer.enable_trace_cpus = True
    parallel_configs = [i for i in range(1, get_phy_cpu_cores() + 1)]
    total_runs = len(parallel_configs)
    i = 0
    for parallel in parallel_configs:
        i += 1
        print(f"====> progression: Computing for {i} / {total_runs} with k: {k_port_per_switch}....")
        log_dir = (log_dir_prefix + 'fattree-parallel-' +
                   ('incast' if sim_type == SIM_TYPE_INCAST else 'unicast') + \
                  '-k-' + str(k_port_per_switch) + '-p-' + str(parallel) + '/')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        if not renew and os.path.exists(log_dir):
            if os.path.exists(log_dir + 'stats') and os.path.getsize(log_dir + 'stats') > 0:
                print("---> automatically jump the old files")
                continue
        run_test_for_fattree(log_dir, sim_type=sim_type, n_pairs=n_pairs,
                             keep_silience=keep_silience, num_cpus=parallel,
                             k_port_per_switch=k_port_per_switch,
                             reuse_prev_flows=True,
                             reuse_prev_net=True,
                             reuse_prev_port_srcs_to_retry=True,
                             batch_size=batch_size)

        print(f"====> progression: Done for {i} / {total_runs}")

# ================== VARYING RETRYING TIMES =======================
def run_varying_retrying_batch_tests_for_fattree(log_dir_prefix, sim_type=SIM_TYPE_INCAST, n_pairs=10,
                                         renew=False, keep_silience=True, 
                                         k_port_per_switch=32, batch_size=100):
    reset_prev_flows_and_net()
    retrying_configs = [i for i in range(1, 11)]
    parallel = get_phy_cpu_cores()
    total_runs = len(retrying_configs)
    i = 0
    for retrying in retrying_configs:
        i += 1
        print(f"====> progression: Computing for {i} / {total_runs} with k: {k_port_per_switch}....")
        log_dir = (log_dir_prefix + 'fattree-retrying-' +
                   ('incast' if sim_type == SIM_TYPE_INCAST else 'unicast') + \
                  '-k-' + str(k_port_per_switch) + '-r-' + str(retrying) + '/')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        if not renew and os.path.exists(log_dir):
            if os.path.exists(log_dir + 'stats') and os.path.getsize(log_dir + 'stats') > 0:
                print("---> automatically jump the old files")
                continue
        run_test_for_fattree(log_dir, sim_type=sim_type, n_pairs=n_pairs,
                             keep_silience=keep_silience, num_cpus=parallel,
                             k_port_per_switch=k_port_per_switch,
                             max_retrying_time=retrying,
                             batch_size=batch_size,
                             reuse_prev_flows=True,
                             reuse_prev_net=True,
                             reuse_prev_port_srcs_to_retry=True)

        print(f"====> progression: Done for {i} / {total_runs}")

# ===================== VARYING BATCH_SIZE =========================
def run_varying_batch_size_tests_for_fattree(log_dir_prefix, sim_type=SIM_TYPE_INCAST, n_pairs=10,
                                         renew=False, keep_silience=True,
                                         k_port_per_switch=32):
    reset_prev_flows_and_net()
    tracer.enable_trace_cpus = True
    batch_sizes = [b for b in range(10, 210, 10)]
    parallel = get_phy_cpu_cores()
    total_runs = len(batch_sizes)
    i = 0
    for batch_size in batch_sizes:
        i += 1
        print(f"====> progression: Computing for {i} / {total_runs} with k: {k_port_per_switch}....")
        log_dir = (log_dir_prefix + 'fattree-batch-size-' +
                   ('incast' if sim_type == SIM_TYPE_INCAST else 'unicast') + \
                  '-k-' + str(k_port_per_switch) + '-b-' + str(batch_size) + '/')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        if not renew and os.path.exists(log_dir):
            if os.path.exists(log_dir + 'stats') and os.path.getsize(log_dir + 'stats') > 0:
                print("---> automatically jump the old files")
                continue
        run_test_for_fattree(log_dir, sim_type=sim_type, n_pairs=n_pairs,
                             keep_silience=keep_silience, num_cpus=parallel,
                             batch_size=batch_size,
                             k_port_per_switch=k_port_per_switch,
                             reuse_prev_flows=True,
                             reuse_prev_net=True,
                             reuse_prev_port_srcs_to_retry=True)

        print(f"====> progression: Done for {i} / {total_runs}")

# ===================== Fattree for NS-3 =========================
def read_sn_dn(filename):
    idxes = []
    sn_list = []
    dn_list = []
    flow_size_list = []
    event_time_list = []

    with open(filename, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            idx = int(row['index'])
            sn = int(row['sn'])
            dn = int(row['dn'])
            event_time = float(row['event_time'])
            flow_size = int(row['flow_size'])

            if sn == dn:
                continue
            idxes.append(idx)
            sn_list.append(sn)
            dn_list.append(dn)
            event_time_list.append(event_time)
            flow_size_list.append(flow_size)

    return idxes, sn_list, dn_list, event_time_list, flow_size_list, len(sn_list)



def ns3_gen_topo(topo_type='fattree', aspen_layers=5, k_port_per_switch=32, topo_output_dir='/tmp/'):
    if topo_type == 'fattree':
        net = fattree_network(k=k_port_per_switch)
        net_desc = {'topo': 'fattree', 'k': k_port_per_switch}
    elif topo_type == 'aspen':
        net = aspen_trees_network(n_levels=aspen_layers, k_port_per_switch=k_port_per_switch, c_factored=[1] * aspen_layers)
        net_desc = {'topo': 'aspen', 'k': k_port_per_switch, 'layers': aspen_layers}
    net.build_topo()

    ep_ids = net.get_ep_ids()
    rack_map = {}
    for ep_id in ep_ids:
        rack_sw_id = list(net.G.neighbors(ep_id))[0]
        if rack_sw_id not in rack_map:
            rack_map[rack_sw_id] = [ep_id]
        else:
            rack_map[rack_sw_id].append(ep_id)

    with open(topo_output_dir + '/ep-ids', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(ep_ids)
        for rack_sw_id in rack_map:
            writer.writerow([rack_sw_id, '-'.join([str(id) for id in rack_map[rack_sw_id]])])

    return net, net_desc

def exe_update_flows_to_schdule(topo_output_dir='/tmp/', n_demands=1000):
    print("Executing flows-generator-for-ns3...")
    command = ["/home/zby/dcqcn-net-control/traffic-gen/dcn-traffic-generator/flows-generator-for-ns3.sh",
               topo_output_dir, str(n_demands)]

    result = subprocess.run(command, capture_output=True, text=True, stdout=None, stderr=None)
    if result.returncode != 0:
        print(f"Error to execute command {command}")
        return

def ns3_gen_src_dst_flows(net, topo_output_dir='/tmp/', flow_to_schedule_file_name='flows-to-schedule.csv',
                          use_routing_linearity=False, m_paths=None):
    idxes, sn_list, dn_list, event_time_list, flow_size_list, n_pairs = read_sn_dn(topo_output_dir + flow_to_schedule_file_name)

    src_dst_flows: List[controller.flow_info] = sim_config.gen_src_dst_flows_with_random_ports_allowing_duplicated_for_one_to_one(
                                                        idxes, sn_list, dn_list, event_time_list, flow_size_list, n_pairs)
    if not use_routing_linearity:
        net.generate_and_install_paths(src_dst_flows, -1, False)
    else:
        net.generate_and_install_paths_ensuring_ecmp_linearity(src_dst_flows, m_paths)
    #print(f"src_dst_flows: \n{src_dst_flows}\n")

    sim_config.save_sim_config(net, src_dst_flows, topo_output_dir)

    return src_dst_flows

def remove_duplicate_lists(list_of_lists):
    seen = set()
    unique = []
    for sublist in list_of_lists:
        t = tuple(sublist)
        if t not in seen:
            seen.add(t)
            unique.append(sublist)
    return unique

def run_path_selection_for_ns3(log_prefix, net, net_desc, src_dst_flows,
                               max_retrying_time=3, only_use_sht=False,
                               flow_path_to_schedule_file_name='flow-path-to-schedule-info.csv',
                               useGen=True):
    config.KEEP_SILENCE = True
    tracer.clear()

    ctl = controller(net, n_cpus=num_cpus, batch_size=batch_size)
    epr_cht, epr_sht = ctl.evaluate_o_delta_comb()

    orig_ecmp_paths_info = ctl.gen_orig_ecmp_paths(src_dst_flows)
    paths = [paths_info[1] for paths_info in orig_ecmp_paths_info]
    jr_orig = ctl.compute_link_jointness_ratio(paths)
    #print("============ orig_ecmp_paths_info =============")
    #print(orig_ecmp_paths_info)
    #print("===============================================")
    if useGen:
        max_disjoint_paths = ctl.gen_max_disjoint_paths_for_flows(src_dst_flows)
    else:
        max_disjoint_paths = ctl.assign_max_disjoint_paths_for_flows(src_dst_flows)

    num_flows = len(src_dst_flows)
    valid_selection = 0
    paths_selected = []
    paths_designated = []
    n_flows = len(src_dst_flows)
    i = 0

    flow_and_path_info = [] # a list of lists: [flow, valid_source_port, path_ecmp, path_designated, path_with_valid_source_port]

    port_srcs_to_retry_all = gen_port_srcs_to_retry_for_flows(max_retrying_time, len(src_dst_flows))
    print(f"port_srcs_to_retry_all: {port_srcs_to_retry_all}\n")

    for flow in src_dst_flows:
        tracer.log(f"finding valid EP src port for flow {flow}")
        port_srcs_to_retry = port_srcs_to_retry_all[i]
        ep_port_src, succ, retrying_times = ctl.find_ep_port_src_uha(flow,
                                                                     max_disjoint_paths, max_retrying_time,
                                                                     only_use_sht,
                                                                     port_srcs_to_retry, True)

        if not succ:
            tracer.log(f"obtaining valid delta is failed for the flow {flow}")
        valid, path_fd, designated_path = ctl.validate_designated_flow(flow, ep_port_src, max_disjoint_paths, True)
        paths_selected.append(path_fd)
        paths_designated.append(designated_path[0])
        tracer.log(f"validation for the flow {flow}: {valid}")
        if valid:
            valid_selection += 1

        tracer.retrying_times(retrying_times)

        if net_desc is not None:
            net_desc_str = ', '.join([k + ": " + str(net_desc[k]) for k in net_desc])
        else:
            net_desc_str = 'no desc'

        flow_and_path_info.append([flow.idx, flow.node_src_id, flow.node_dst_id, flow.port_src, flow.port_dst,
                                   ep_port_src, flow.event_time, flow.flow_size,
                                   paths[i], designated_path[0], path_fd])

        i += 1
        print(f"====> progression for ({net_desc_str}): {i} / {n_flows}")

    pss = valid_selection / num_flows
    jr_selected = ctl.compute_link_jointness_ratio(remove_duplicate_lists(paths_selected))
    jr_expected = ctl.compute_link_jointness_ratio(remove_duplicate_lists(paths_designated))
    n_endpoints = net.get_num_endpoints()
    n_switches = len(net.get_switch_ids())

    tracer.dump_all_to_files(log_prefix)
    tracer.write_sim_stats(log_prefix, epr_cht, epr_sht, pss,
                           n_endpoints, n_switches, n_flows,
                           jr_orig, jr_expected, jr_selected,
                           net_desc)
    tracer.clear()

    sim_config.save_flow_and_path_info(flow_and_path_info, log_prefix + flow_path_to_schedule_file_name)

# col_idx: 5 for path ecmp, 6 for path designated, and 7 for path with valid source port
def getInfluencedPaths(sampled_flow_idxes, flow_and_path_info_all, col_idx, n_flows):
    links_all = set()
    sfi_all = set(sampled_flow_idxes)
    for idx in sampled_flow_idxes:
        p = flow_and_path_info_all[idx][col_idx]
        for j in range(0, len(p) - 1):
            links_all.add((p[j] << 16) + p[j + 1])

    influencedIdxes = sampled_flow_idxes.copy()
    for i in range(n_flows):
        if i in sfi_all:
            continue
        p = flow_and_path_info_all[i][col_idx]
        for j in range(0, len(p) - 1):
            l_hash = (p[j] << 16) + p[j + 1]
            if l_hash in links_all:
                influencedIdxes.append(i)
                break

    influencedIdxes.sort()
    return influencedIdxes


def computeInfluencedPathsInVariousTypes(log_prefix, n_nodes_samples):
    flow_path_info_file_path = log_prefix + 'flow-path-info.csv'
    flow_and_path_info_all = sim_config.load_flow_and_path_info(flow_path_info_file_path)
    n_flows = len(flow_and_path_info_all)
    sampled_flow_idxes = random.sample(range(n_flows), k=n_nodes_samples)

    path_ecmp_idxes = getInfluencedPaths(sampled_flow_idxes, flow_and_path_info_all, 5, n_flows)
    path_designated_idxes = getInfluencedPaths(sampled_flow_idxes, flow_and_path_info_all, 6, n_flows)
    path_valid_idxes = getInfluencedPaths(sampled_flow_idxes, flow_and_path_info_all, 7, n_flows)

    sim_config.save_flow_and_path_info_by_idxes(flow_and_path_info_all, path_ecmp_idxes,
                                                log_prefix + 'flow-path-info-ecmp.csv')
    sim_config.save_flow_and_path_info_by_idxes(flow_and_path_info_all, path_designated_idxes,
                                                log_prefix + 'flow-path-info-designated.csv')
    sim_config.save_flow_and_path_info_by_idxes(flow_and_path_info_all, path_valid_idxes,
                                                log_prefix + 'flow-path-info-valid.csv')


def run_path_selection_for_ns3_in_batch(ctl, log_prefix, src_dst_flows,
                               max_retrying_time=3, only_use_sht=False,
                               flow_path_to_schedule_file_name='flow-path-to-schedule-info.csv'):

    orig_ecmp_paths_info = ctl.gen_orig_ecmp_paths(src_dst_flows)
    paths = [paths_info[1] for paths_info in orig_ecmp_paths_info]
    max_disjoint_paths = ctl.gen_max_disjoint_paths_for_flows(src_dst_flows)
    valid_selection = 0
    paths_selected = []
    paths_designated = []
    n_flows = len(src_dst_flows)
    i = 0

    flow_and_path_info = [] # a list of lists: [flow, valid_source_port, path_ecmp, path_designated, path_with_valid_source_port]

    port_srcs_to_retry_all = gen_port_srcs_to_retry_for_flows(max_retrying_time, len(src_dst_flows))
    print(f"port_srcs_to_retry_all: {port_srcs_to_retry_all}\n")

    for flow in src_dst_flows:
        tracer.log(f"finding valid EP src port for flow {flow}")
        port_srcs_to_retry = port_srcs_to_retry_all[i]
        ep_port_src, succ, retrying_times = ctl.find_ep_port_src_uha(flow,
                                                                     max_disjoint_paths, max_retrying_time,
                                                                     only_use_sht,
                                                                     port_srcs_to_retry, True)

        if not succ:
            tracer.log(f"obtaining valid delta is failed for the flow {flow}")
        valid, path_fd, designated_path = ctl.validate_designated_flow(flow, ep_port_src, max_disjoint_paths, True)
        paths_selected.append(path_fd)
        paths_designated.append(designated_path[0])
        tracer.log(f"validation for the flow {flow}: {valid}")
        if valid:
            valid_selection += 1

        tracer.retrying_times(retrying_times)

        flow_and_path_info.append([flow.idx, flow.node_src_id, flow.node_dst_id, flow.port_src, flow.port_dst,
                                   ep_port_src, flow.event_time, flow.flow_size,
                                   paths[i], designated_path[0], path_fd])

        i += 1
        print(f"====> progression: {i} / {n_flows}")


    sim_config.save_flow_and_path_info(flow_and_path_info, log_prefix + flow_path_to_schedule_file_name)

def gen_flow_path_info_for_fattree_in_batch_for_incast_r2r(log_path, n_times=10, num_cpus=4):
    net, net_desc = ns3_gen_topo(k_port_per_switch=16, topo_output_dir=log_path)
    gen = flows_generator(log_path)

    num_bytes_per_flow_min = 1e7
    num_bytes_per_flow_max = 1e8

    num_racks_all = [x for x in range(10, 40, 10)]
    template_flow_to_schedule_file_name = "flows-to-schedule-rack-{num_racks}-times-{times}.csv"
    template_flow_path_to_schedule_file_name = "flow-path-to-schedule-info-rack-{num_racks}-times-{times}.csv"

    src_dst_flows_aggregated = []

    src_dst_flows_all = {}
    for num_racks in num_racks_all:
        src_dst_flows_all[num_racks] = {}
        for times in range(n_times):
            flow_to_schedule_file_name = template_flow_to_schedule_file_name.format(num_racks=num_racks, times=times)
            gen.gen_flows_for_ns3_incast_rack2rack(flow_to_schedule_file_name, num_racks=num_racks,
                                                   num_bytes_per_flow_min=num_bytes_per_flow_min,
                                                   num_bytes_per_flow_max=num_bytes_per_flow_max)

            idxes, sn_list, dn_list, event_time_list, flow_size_list, n_pairs = read_sn_dn(
                                            log_path + flow_to_schedule_file_name)

            src_dst_flows: List[controller.flow_info] = sim_config.gen_src_dst_flows_with_random_ports_allowing_duplicated_for_one_to_one(
                idxes, sn_list, dn_list, event_time_list, flow_size_list, n_pairs)

            src_dst_flows_all[num_racks][times] = src_dst_flows

            for f in src_dst_flows:
                matched = False
                for fa in src_dst_flows_aggregated:
                    if f.node_src_id == fa.node_src_id and f.node_dst_id == fa.node_dst_id:
                        matched = True
                        break
                if not matched:
                    src_dst_flows_aggregated.append(f)

    net.generate_and_install_paths(src_dst_flows_aggregated, -1, False)
    sim_config.save_sim_config(net, src_dst_flows_aggregated, log_path)

    config.KEEP_SILENCE = True
    tracer.clear()
    ctl = controller(net, n_cpus=num_cpus, batch_size=batch_size)

    for num_racks in num_racks_all:
        for times in range(n_times):
            src_dst_flows = src_dst_flows_all[num_racks][times]
            flow_path_to_schedule_file_name = template_flow_path_to_schedule_file_name.format(num_racks=num_racks, times=times)

            run_path_selection_for_ns3_in_batch(ctl=ctl, log_prefix=log_path, src_dst_flows=src_dst_flows,
                                   max_retrying_time=3, only_use_sht=False,
                                   flow_path_to_schedule_file_name=flow_path_to_schedule_file_name)

def write_csv_with_routing_tests_perf(pu_all, time_all, n_levels_all, k_port_per_switch_all, n_hosts, n_switches, log_file_path):
    s = ""
    for n_levels in n_levels_all:
        for k_port_per_switch in k_port_per_switch_all:
            s += (f"{n_levels},{k_port_per_switch},{n_switches},{n_hosts},{'#'.join([str(x) for x in pu_all[n_levels][k_port_per_switch]])},"
                  f"{'#'.join([str(x) for x in time_all[n_levels][k_port_per_switch]])}\n")
    tracer.write_to_file(s, log_file_path)

def evaluate_routing_trace_results_all(trace_results):
    mean_path_densities = []
    std_path_densities = []
    link_usage_ratios_all = []
    path_density_map = {}
    for l_id in trace_results['link_density_map_paths_all']:
        if l_id in trace_results['link_density_map_paths_pow2']:
            path_density_map[l_id] = trace_results['link_density_map_paths_pow2'][l_id]
        else:
            path_density_map[l_id] = 0
    path_densities = [path_density_map[l_id] for l_id in path_density_map]
    mean_path_densities.append(np.mean(path_densities))
    std_path_densities.append(np.std(path_densities))
    link_usage_ratios_all.append(len(trace_results['link_density_map_paths_pow2']) / len(trace_results['link_density_map_paths_all']))

    return mean_path_densities, std_path_densities, link_usage_ratios_all

def write_csv_with_routing_tests_detail(trace_results_gathered, log_file_path):
    s = ""
    for item in trace_results_gathered:
        s += (f"{item[0]},{item[1]},{item[2]},{item[3]},"
              f"{'#'.join([str(x) for x in item[4]])},"
              f"{'#'.join([str(x) for x in item[5]])},"
              f"{'#'.join([str(x) for x in item[6]])},"
              f"{item[7]}\n")
    tracer.write_to_file(s, log_file_path)

def run_single_routing_case(params):
    n_levels, k_port_per_switch, _m_paths, n_pairs, c_factored = params

    net = aspen_trees_network(n_levels=n_levels, k_port_per_switch=k_port_per_switch, c_factored=c_factored)
    net.build_topo()

    n_hosts = len(net.get_ep_ids())
    max_n_pairs = n_hosts * (n_hosts - 1)
    n_pairs_to_pick = np.min([n_pairs, max_n_pairs])

    src_dst_flows = sim_config.gen_random_src_dst_flows_for_one_to_one(net, n_pairs_to_pick)
    gen = path_generator_with_ecmp_hash_linearity()
    paths_all, trace_results_all = gen.generate_paths(net.G, src_dst_flows=src_dst_flows, m_paths=_m_paths,
                                                      time_records=None, trace=True)

    mean_path_densities, std_path_densities, link_usage_ratios_all = (
        evaluate_routing_trace_results_all(trace_results_all))

    num_paths_max = -1
    for node_src in paths_all:
        for node_dst in paths_all[node_src]:
            num_paths = len(paths_all[node_src][node_dst])
            if num_paths > num_paths_max:
                num_paths_max = num_paths

    return [n_levels, k_port_per_switch, _m_paths, n_pairs, mean_path_densities,
            std_path_densities, link_usage_ratios_all, num_paths_max]

def run_powerpath_in_parallel():
    log_file_path = log_path + 'log_routing_detail'
    n_levels_all = [2, 3, 4, 5]
    k_port_per_switch_all = [i for i in range(8, 16 + 2, 2)]
    c_factored = [1, 1, 1, 1, 1]
    n_pairs_all = [n for n in range(100, 1000 + 10, 100)]
    m_paths_all = [m_paths for m_paths in range(50, 1000 + 50, 50)] + [-1]

    param_list = []
    for n_levels in n_levels_all:
        for k_port_per_switch in k_port_per_switch_all:
            for _m_paths in m_paths_all:
                for n_pairs in n_pairs_all:
                    param_list.append((n_levels, k_port_per_switch, _m_paths, n_pairs, c_factored))

    total_run = len(param_list)
    trace_results_gathered = []

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = {executor.submit(run_single_routing_case, params): idx for idx, params in enumerate(param_list)}
        for idx, future in enumerate(as_completed(futures)):
            result = future.result()
            trace_results_gathered.append(result)
            print(f"!!!===> total progression: {idx + 1} / {total_run}")

    write_csv_with_routing_tests_detail(trace_results_gathered, log_file_path)


def run_single_routing_case_for_routing_tables(params):
    n_levels, k_port_per_switch, _m_paths, n_pairs, c_factored = params

    net = aspen_trees_network(n_levels=n_levels, k_port_per_switch=k_port_per_switch, c_factored=c_factored)
    net.build_topo()

    n_hosts = len(net.get_ep_ids())
    max_n_pairs = n_hosts * (n_hosts - 1)
    n_pairs_to_pick = np.min([n_pairs, max_n_pairs])

    src_dst_flows = sim_config.gen_random_src_dst_flows_for_one_to_one(net, n_pairs_to_pick)
    net.generate_and_install_paths_ensuring_ecmp_linearity(src_dst_flows, _m_paths)
    routing_table_sizes = []
    for sw_id in net.get_switch_ids():
        node_obj = net.get_node_obj(sw_id)
        size = node_obj.routes.get_size()
        routing_table_sizes.append(size)

    return [n_levels, k_port_per_switch, _m_paths, n_pairs, routing_table_sizes]

def write_csv_with_routing_tables(trace_results_gathered, log_file_path):
    s = ""
    for item in trace_results_gathered:
        s += (f"{item[0]},{item[1]},{item[2]},{item[3]},"
              f"{'#'.join([str(x) for x in item[4]])}\n")
    tracer.write_to_file(s, log_file_path)

def run_powerpath_in_parallel_routing_tables():
    log_file_path = log_path + 'log_routing_tables'
    n_levels_all = [2, 3, 4, 5]
    k_port_per_switch_all = [i for i in range(8, 16 + 2, 2)]
    c_factored = [1, 1, 1, 1, 1]
    n_pairs_all = [n for n in range(100, 1000 + 10, 100)]
    m_paths_all = [m_paths for m_paths in range(50, 1000 + 50, 50)] + [-1]

    param_list = []
    for n_levels in n_levels_all:
        for k_port_per_switch in k_port_per_switch_all:
            for _m_paths in m_paths_all:
                for n_pairs in n_pairs_all:
                    param_list.append((n_levels, k_port_per_switch, _m_paths, n_pairs, c_factored))

    total_run = len(param_list)
    trace_results_gathered = []

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = {executor.submit(run_single_routing_case_for_routing_tables, params): idx for idx, params in enumerate(param_list)}
        for idx, future in enumerate(as_completed(futures)):
            result = future.result()
            trace_results_gathered.append(result)
            print(f"!!!===> total progression: {idx + 1} / {total_run}")

    write_csv_with_routing_tables(trace_results_gathered, log_file_path)

def get_max_nranks_in_simai_topo(k_port_per_switch):
    int(k_port_per_switch / 2)**2

def build_simai_net(k_port_per_switch, nranks):
    random.seed(5)
    n_ep_per_leaf_sw = int(k_port_per_switch / 2)
    n_spine_sw = int(nranks / n_ep_per_leaf_sw)
    n_leaf_sw = int(nranks / n_ep_per_leaf_sw)
    n_ep = n_leaf_sw * n_ep_per_leaf_sw
    net = sl_network(n_spine_sw=n_spine_sw, n_leaf_sw=n_leaf_sw, n_ep=n_ep, n_ep_per_leaf_sw=n_ep_per_leaf_sw)
    net.build_topo_for_simai()
    trunk_per_spine_leaf_conn = int(k_port_per_switch / (2 * n_spine_sw))
    per_link_band = 200
    band_map = {}
    for sw_spine in net.switches_spine:
        for sw_leaf in net.switches_leaf:
            band_map[(sw_spine.node_id, sw_leaf.node_id)] = per_link_band * trunk_per_spine_leaf_conn
            band_map[(sw_leaf.node_id, sw_spine.node_id)] = per_link_band * trunk_per_spine_leaf_conn
    ep_ids = net.get_ep_ids()
    for sw_leaf in net.switches_leaf:
        for port_id, node_id in net.next_node_id[sw_leaf.node_id].items():
            if node_id in ep_ids:
                band_map[(sw_leaf.node_id, node_id)] = per_link_band
                band_map[(node_id, sw_leaf.node_id)] = per_link_band

    net_desc = {'topo': 'spineleaf', 'n_spine_sw': n_spine_sw, 'n_leaf_sw': n_leaf_sw,
                'n_ep': n_ep, 'n_ep_per_leaf_sw': n_ep_per_leaf_sw}
    return net, net_desc, band_map

if __name__ == '__main__':
    #profiler = cProfile.Profile()
    #profiler.enable()

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-t", "--topo", required=True,
                            help="Topologies including aspen, fattree, spineleaf, "
                                 "spineleafrecap (ReCap, only use SHT), "
                                 "parallel_fattree, retrying_fattree, batch_size_fattree,"
                                 "fattree_ns3")
    arg_parser.add_argument("-n", "--npairs", required=False,
                            help="The number of communication pairs")
    arg_parser.add_argument("-c", "--cast", required=False,
                            help="The type of flows either unicast or incast")
    arg_parser.add_argument("-l", "--logpath", required=True,
                            help="Log path")
    arg_parser.add_argument("-p", "--parallel", required=False,
                            help="The maximum number of CPUs to use for computing")
    arg_parser.add_argument("-b", "--batchsize", required=False,
                            help="The threshold of batch size for parallel executions, in the number of combinations")
    arg_parser.add_argument("-r", "--renew", required=False, action='store_true',
                            help="Renew logs by overwriting")
    arg_parser.add_argument("-o", "--porttype", required=False,
                            help="port type for NS-3 simulation, either ecmp or valid")
    arg_parser.add_argument("-a", "--nracks", required=False,
                            help="the number of racks")
    arg_parser.add_argument("-s", "--timesstart", required=False,
                            help="times to start in testing")
    arg_parser.add_argument("-d", "--timesend", required=False,
                            help="times to end in testing")
    arg_parser.add_argument("-x", "--linearityroutes", required=False, action='store_false',
                            help="Use routes ensuring ECMP hash linearity, where mpaths parameter must be set")
    arg_parser.add_argument("-m", "--mpaths", required=False,
                            help="Use mpaths to allow the abitrary number of paths for each communication pair, "
                                 "while maintaining ECMP hash linearity")
    arg_parser.add_argument("-k", "--kports", required=False,
                            help="the number of switch ports, only for aspen_ns3, simai")
    arg_parser.add_argument("-i", "--policy", required=False,
                            help="the policy of the source port assignment for SimAI, including random, incremental, or hps")
    arg_parser.add_argument("-y", "--simai", required=False,
                            help="the root directory of SIMAI")
    arg_parser.add_argument("-z", "--nranks", required=False,
                            help="the number of ranks in SIMAI")
    args = arg_parser.parse_args()

    if args.simai is not None:
        simai_dir = args.simai
    else:
        simai_dir = '/home/zby/ext/SimAI/'

    num_cpus = get_phy_cpu_cores()
    if args.parallel is not None:
        num_cpus = int(args.parallel)

    print(f"num CPUs: {num_cpus}")

    if args.cast == 'unicast':
        sim_type = SIM_TYPE_ONE_TO_ONE
    else:
        sim_type = SIM_TYPE_INCAST

    batch_size = 100
    if args.batchsize is not None:
        batch_size = int(args.batchsize)

    renew = False
    if args.renew is not None:
        renew = args.renew

    use_linearity_routes = False
    if args.linearityroutes is not None:
        use_linearity_routes = True

    n_pairs = int(args.npairs)

    log_path = args.logpath

    m_paths = None
    if args.mpaths is not None:
        m_paths = int(args.mpaths)

    if args.topo == 'aspen':
        run_batch_tests_for_aspen(log_path, sim_type=sim_type,
                                  batch_size=batch_size,
                                  n_pairs=n_pairs, renew=renew, num_cpus=num_cpus,
                                  use_linearity_routes=use_linearity_routes,
                                  m_paths=m_paths)

    if args.topo == 'aspen_varying_retries':
        run_batch_tests_for_aspen_varying_retries(log_path, sim_type=sim_type,
                                  batch_size=batch_size,
                                  n_pairs=n_pairs, renew=renew, num_cpus=num_cpus,
                                  use_linearity_routes=use_linearity_routes,
                                  m_paths=m_paths)

    if args.topo == 'aspen_varying_hashing':
        run_batch_tests_for_aspen_varying_hashing(log_path, sim_type=sim_type,
                                  batch_size=batch_size,
                                  n_pairs=n_pairs, renew=renew, num_cpus=num_cpus,
                                  use_linearity_routes=use_linearity_routes,
                                  m_paths=m_paths)

    elif args.topo == 'fattree':
        run_batch_tests_for_fattree(log_path, sim_type=sim_type,
                                  batch_size=batch_size,
                                  n_pairs=n_pairs, renew=renew, num_cpus=num_cpus,
                                  use_linearity_routes=use_linearity_routes,
                                  m_paths=m_paths)
    elif args.topo == 'spineleaf':
        run_batch_tests_for_spineleaf(log_path, sim_type=sim_type,
                                  batch_size=batch_size,
                                  n_pairs=n_pairs, renew=renew, num_cpus=num_cpus,
                                  use_linearity_routes=use_linearity_routes,
                                  m_paths=m_paths)
    elif args.topo == 'spineleafrecap':
        run_batch_tests_for_spineleaf(log_path, sim_type=sim_type,
                                    batch_size=batch_size,
                                    n_pairs=n_pairs, renew=renew, num_cpus=num_cpus,
                                    only_use_sht=True,
                                    use_linearity_routes=use_linearity_routes,
                                    m_paths=m_paths)
    elif args.topo == 'parallel_fattree':
        run_parallel_batch_tests_for_fattree(log_path, sim_type=sim_type, n_pairs=n_pairs,
                                            batch_size=batch_size,
                                            renew=renew,
                                            k_port_per_switch=32,
                                            use_linearity_routes=use_linearity_routes,
                                            m_paths=m_paths)
    elif args.topo == 'retrying_fattree':
        run_varying_retrying_batch_tests_for_fattree(log_path, sim_type=sim_type, n_pairs=n_pairs,
                                                    batch_size=batch_size,
                                                    renew=renew,
                                                    k_port_per_switch=32,
                                                    use_linearity_routes=use_linearity_routes,
                                                    m_paths=m_paths)
    elif args.topo == 'batch_size_fattree':
        run_varying_batch_size_tests_for_fattree(log_path, sim_type=sim_type, n_pairs=n_pairs,
                                                renew=renew,
                                                k_port_per_switch=32,
                                                use_linearity_routes=use_linearity_routes,
                                                m_paths=m_paths)

    elif args.topo == 'fattree_ns3':
        random.seed(time.time())
        net, net_desc = ns3_gen_topo(topo_type='fattree', k_port_per_switch=16, topo_output_dir=log_path)
        flow_type = "incast_r2r"
        num_eps = 16
        num_racks = 20
        num_bytes_per_flow_fixed = 1e7
        num_bytes_per_flow_min = 1e7
        num_bytes_per_flow_max = 1e8
        flow_to_schedule_file_name = "flows-to-schedule.csv"
        flow_path_to_schedule_file_name = "flow-path-to-schedule-info.csv"
        gen = flows_generator(log_path)

        if flow_type == "trafpy":
            num_demands = 10
            gen.gen_flows_for_ns3(flow_to_schedule_file_name, num_eps=num_eps, num_demands=num_demands)
        elif flow_type == "incast":
            gen.gen_flows_for_ns3_incast(flow_to_schedule_file_name, num_eps=num_eps,
                                         num_bytes_per_flow_min=num_bytes_per_flow_min,
                                         num_bytes_per_flow_max=num_bytes_per_flow_max)
        elif flow_type == "incast_r2r":
            gen.gen_flows_for_ns3_incast_rack2rack(flow_to_schedule_file_name, num_racks=num_racks,
                                                    num_bytes_per_flow_min=num_bytes_per_flow_fixed,
                                                    fixed_flow_size=True, fixed_event_time=True)
            #gen.gen_flows_for_ns3_incast_rack2rack(flow_to_schedule_file_name, num_racks=num_racks,
            #                             num_bytes_per_flow_min=num_bytes_per_flow_min,
            #                             num_bytes_per_flow_max=num_bytes_per_flow_max)
        elif flow_type == "alltoall":
            gen.gen_flows_for_ns3_alltoall(flow_to_schedule_file_name, num_eps=num_eps,
                                           num_bytes_per_flow_min=num_bytes_per_flow_min,
                                           num_bytes_per_flow_max=num_bytes_per_flow_max)

        src_dst_flows = ns3_gen_src_dst_flows(net, topo_output_dir=log_path,
                                              flow_to_schedule_file_name=flow_to_schedule_file_name)
        run_path_selection_for_ns3(log_prefix=log_path, net=net, net_desc=net_desc, src_dst_flows=src_dst_flows,
                                   max_retrying_time=3, only_use_sht=False,
                                   flow_path_to_schedule_file_name=flow_path_to_schedule_file_name)
        #computeInfluencedPathsInVariousTypes(log_prefix=log_path, n_nodes_samples=1)

    elif args.topo == 'aspen_ns3':
        random.seed(time.time())
        if args.kports is None:
            k_port_per_switch = 8
        else:
            k_port_per_switch = int(args.kports)

        net, net_desc = ns3_gen_topo(topo_type='aspen', aspen_layers=4,  k_port_per_switch=k_port_per_switch, topo_output_dir=log_path)
        num_racks = 20
        num_bytes_per_flow_fixed = 1e7
        flow_to_schedule_file_name = "flows-to-schedule.csv"
        flow_path_to_schedule_file_name = "flow-path-to-schedule-info.csv"
        gen = flows_generator(log_path)
        gen.gen_flows_for_ns3_incast_rack2rack(flow_to_schedule_file_name, num_racks=num_racks,
                                               num_bytes_per_flow_min=num_bytes_per_flow_fixed,
                                               fixed_flow_size=True, fixed_event_time=True)
        if m_paths is None:
            m_paths = -1
        src_dst_flows = ns3_gen_src_dst_flows(net, topo_output_dir=log_path,
                                              flow_to_schedule_file_name=flow_to_schedule_file_name,
                                              use_routing_linearity=True, m_paths=m_paths)
        run_path_selection_for_ns3(log_prefix=log_path, net=net, net_desc=net_desc, src_dst_flows=src_dst_flows,
                                   max_retrying_time=3, only_use_sht=False,
                                   flow_path_to_schedule_file_name=flow_path_to_schedule_file_name,
                                   useGen=True)


    elif args.topo == 'run_fattree_ns3':
        flow_path_to_schedule_file_name = "flow-path-to-schedule-info.csv"

        data_plane_ns3_sim.run_dcn_sim(log_prefix=log_path, port_type=args.porttype,
                                       schedule_file_name=flow_path_to_schedule_file_name,
                                       output_file_path=log_path + 'flow-perf-port-' + args.porttype,
                                       routes_file_name="routes", renew=True)

    elif args.topo == 'fattree_ns3_incast_batch':
        gen_flow_path_info_for_fattree_in_batch_for_incast_r2r(log_path=log_path, n_times=10, num_cpus=4)
    elif args.topo == 'run_data_plane_fattree_ns3_incast_batch':
        data_plane_ns3_sim.run_batch_dcn_sim_incast(log_prefix=log_path, port_type="valid",
                                                    n_times=10, n_threads=8, renew=True)

        data_plane_ns3_sim.run_batch_dcn_sim_incast(log_prefix=log_path, port_type="ecmp",
                                                    n_times=10, n_threads=4, renew=True)
    elif args.topo == 'run_incast_batch_folders':
        num_racks = 20
        times_start = 0
        times_end = 20
        if args.nracks is not None:
            num_racks = int(args.nracks)
        if args.timesstart is not None:
            times_start = int(args.timesstart)
        if args.timesend is not None:
            times_end = int(args.timesend)

        if args.porttype == "valid":
            data_plane_ns3_sim.run_batch_dcn_sim_incast_for_folders(log_prefix=log_path, num_rack=num_racks,
                                                                    port_type="valid",
                                                                    times_start=times_start, times_end=times_end,
                                                                    n_threads=4, renew=True)
        elif args.porttype == "ecmp":
            data_plane_ns3_sim.run_batch_dcn_sim_incast_for_folders(log_prefix=log_path, num_rack=num_racks,
                                                                    port_type="ecmp",
                                                                    times_start=times_start, times_end=times_end,
                                                                    n_threads=4, renew=True)
    elif args.topo == 'run_aspen_incast_batch_folders':
        times_start = 0
        times_end = 19
        if args.nracks is not None:
            num_racks = int(args.nracks)
        if args.timesstart is not None:
            times_start = int(args.timesstart)
        if args.timesend is not None:
            times_end = int(args.timesend)

        if args.kports is None:
            k_port_per_switch = 8
        else:
            k_port_per_switch = int(args.kports)

        if m_paths is None:
            m_paths = -1

        subfolder_template = f"aspen-topo-k-{k_port_per_switch}-rack-" + "{num_rack}-m-{m_paths}-{i}"
        if args.porttype == "valid":
            data_plane_ns3_sim.run_batch_dcn_sim_incast_for_folders(log_prefix=log_path, num_rack=num_racks,
                                                                    port_type="valid",
                                                                    times_start=times_start, times_end=times_end,
                                                                    m_paths=m_paths,
                                                                    n_threads=8, renew=False,
                                                                    subfolder_template=subfolder_template)
        elif args.porttype == "ecmp":
            data_plane_ns3_sim.run_batch_dcn_sim_incast_for_folders(log_prefix=log_path, num_rack=num_racks,
                                                                    port_type="ecmp",
                                                                    times_start=times_start, times_end=times_end,
                                                                    m_paths=m_paths,
                                                                    n_threads=8, renew=False,
                                                                    subfolder_template=subfolder_template)

    elif args.topo == 'linearity_route_tests':
        net = aspen_trees_network(n_levels=4, k_port_per_switch=10, c_factored=[1, 1, 1, 1])
        net.build_topo()
        src_dst_flows = []
        f = controller.flow_info()
        f.node_src_id = 875
        f.node_dst_id = 2123
        src_dst_flows.append(f)
        #src_dst_flows = sim_config.gen_random_src_dst_flows_for_one_to_one(net, n_pairs)
        gen = path_generator_with_ecmp_hash_linearity()
        paths, _ = gen.generate_paths(net.G, src_dst_flows=src_dst_flows, m_paths=m_paths)
        print(paths)

    elif args.topo == 'routing_tests_power_of_two':
        log_file_path_perf = log_path + 'log_routing_perf'
        n_levels_all = [2, 3, 4, 5]
        k_port_per_switch_all = [i for i in range(2, 16 + 2, 2)]
        c_factored = [1, 1, 1, 1, 1]
        n_pairs = 100
        pu_all = {}
        time_all = {}

        total_run = len(n_levels_all) * len(k_port_per_switch_all)
        i = 0
        for n_levels in n_levels_all:
            pu_all[n_levels] = {}
            time_all[n_levels] = {}
            for k_port_per_switch in k_port_per_switch_all:
                pu_all[n_levels][k_port_per_switch] = []
                time_all[n_levels][k_port_per_switch] = []
                net = aspen_trees_network(n_levels=n_levels, k_port_per_switch=k_port_per_switch, c_factored=c_factored)
                net.build_topo()
                n_hosts = len(net.get_ep_ids())
                max_n_pairs = n_hosts * (n_hosts - 1)
                n_pairs_to_pick = np.min([n_pairs, max_n_pairs])
                n_switches = len(net.get_switch_ids())

                src_dst_flows = sim_config.gen_random_src_dst_flows_for_one_to_one(net, n_pairs_to_pick)
                gen = path_generator_with_ecmp_hash_linearity()
                paths_all, _ = gen.generate_paths(net.G, src_dst_flows=src_dst_flows, m_paths=m_paths,
                                               time_records=time_all[n_levels][k_port_per_switch])

                for src in paths_all.keys():
                    for dst in paths_all[src].keys():
                        paths = paths_all[src][dst]
                        results, pu = gen.check_ecmp_hash_linearity(paths)
                        pu_all[n_levels][k_port_per_switch] += [pu]

                print(f"!!!===> total progression: {i} / {total_run}")
                i += 1

        write_csv_with_routing_tests_perf(pu_all, time_all, n_levels_all, k_port_per_switch_all, n_hosts, n_switches, log_file_path_perf)
        print(f"log file has been written to {log_file_path_perf}\n")

    elif args.topo == 'routing_tests_power_of_two_detail':
        run_powerpath_in_parallel()

    elif args.topo == 'routing_tests_power_of_two_routing_tables':
        run_powerpath_in_parallel_routing_tables()

    elif args.topo == 'simai_topo':
        if args.kports is None:
            k_port_per_switch = 16
        else:
            k_port_per_switch = int(args.kports)
        if args.nranks is None:
            nranks = 32
        else:
            nranks = int(args.nranks)
        net, net_desc, band_map = build_simai_net(k_port_per_switch, nranks)
        id_map, id_rmap, server_addresses = sim_config.save_as_simai_topo(net=net,
                                                                          file_path=simai_dir + 'simai-topo',
                                                                          file_path_mapping=simai_dir + 'simai-topo-id-map',
                                                                          file_path_server_addresses=simai_dir + 'simai-server-addresses',
                                                                          gpus_per_server=4, gpu_type_str="H100", band_map=band_map)
        sim_config.save_node_ecmp_seeds_and_permutations(net=net,
                                                         file_path_seeds=simai_dir + 'seeds',
                                                         file_path_permutations=simai_dir + 'permutations',
                                                         id_map=id_map)


    elif args.topo == 'simai_run':
        if args.kports is None:
            k_port_per_switch = 16
        else:
            k_port_per_switch = int(args.kports)
        if args.nranks is None:
            nranks = 32
        else:
            nranks = int(args.nranks)
        net, net_desc, band_map = build_simai_net(k_port_per_switch, nranks)
        net.load_id_mapping_for_simai(simai_dir)
        net.adjust_port_ids_to_align_with_simai(simai_dir)
        sim_config.load_node_ecmp_seeds_and_permutations(net=net,
                                                         file_path_seeds=simai_dir + 'seeds',
                                                         file_path_permutations=simai_dir + 'permutations',
                                                         id_rmap=net.id_rmap)
        net.install_routes_from_csv(simai_dir)
        switch.target_simai = True
        config.KEEP_SILENCE = True

        ctl = controller(net, n_cpus=num_cpus, batch_size=batch_size)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        server = RestApiServer(net, ctl, log_path)
        # gui_thread = threading.Thread(target=server.launch_gui, daemon=True)
        #gui_thread.start()
        print("Server has been started...")
        server.run()
        server.end()
        print("Server has ended...")

    elif args.topo == 'simai_run_ps':
        if args.kports is None:
            k_port_per_switch = 16
        else:
            k_port_per_switch = int(args.kports)
        if args.nranks is None:
            nranks = 32
        else:
            nranks = int(args.nranks)
        net, net_desc, band_map = build_simai_net(k_port_per_switch, nranks)
        net.load_id_mapping_for_simai(simai_dir)
        net.adjust_port_ids_to_align_with_simai(simai_dir)
        net.install_routes_from_csv(simai_dir)
        switch.target_simai = True

        log_dir = '/home/zby/ext/log-simai-tests/'
        run_sim_for_a_net(log_prefix=log_dir, net=net,
                          sim_type=SIM_TYPE_PICK_FROM_VIALBLE_PATHS, n_pairs=n_pairs,
                          max_retrying_time=3, max_paths=-1, use_rand_path=False,
                          use_progress_bar=True, keep_silience=False,
                          batch_size=batch_size,
                          num_cpus=num_cpus, net_desc=net_desc,
                          only_use_sht=False,
                          reuse_prev_flows=True,
                          reuse_prev_port_srcs_to_retry=True,
                          paths_installed=True,
                          m_paths=m_paths)

    # elif args.topo == 'routing_tests_power_of_two_link_failures':
    #     n_levels_all = [2, 3, 4, 5]
    #     k_port_per_switch_all = [i for i in range(2, 16 + 2, 2)]
    #     c_factored = [1, 1, 1, 1, 1]
    #     n_pairs = 100
    #     pu_all = {}
    #     time_all = {}
    #
    #     for n_levels in n_levels_all:
    #         for k_port_per_switch in k_port_per_switch_all:
    #             net = aspen_trees_network(n_levels=n_levels, k_port_per_switch=k_port_per_switch, c_factored=c_factored)
    #             net.build_topo()
    else:
        print('unknown topology')

    #profiler.disable()
    #stats = pstats.Stats(profiler).sort_stats("cumtime")
    #stats.print_stats()

#net = aspen_trees_network(n_levels=3, k_port_per_switch=16, c_factored=[1, 1, 1, 1, 1])
#net = sl_network(n_spine_sw=4, n_leaf_sw=8, n_ep=128, n_ep_per_leaf_sw=16)
#net = fattree_network(k=64)
#run_sim_for_a_net(log_prefix='/tmp/', net,
#                      sim_type=SIM_TYPE_INCAST, n_pairs=10,
#                      max_retrying_time=3, max_paths=-1, use_rand_path=False,
#                      use_progress_bar=False, keep_silience=False, batch_size=10, num_cpus=8)

#net.build_topo()
#net.visualize()
#test_hash_collision(net, 100)

#ctl.run_path_tests(ep_src_idx=5, ep_dst_idx=30)

#sys.exit(0)


#net.visualize()

#for i in range(100):
#    test_ecmp_hash(net, k=5, nnh=8)

#sys.exit(0)
#ep_src : endpoint = net.endpoints[0]
#ep_dst : endpoint = net.endpoints[net.n_ep - 1]
#ep_src.send_test_packet(8090, ep_dst.node_id, 8800)


#net.visualize()
#net.dump_ports()
#net.dump_routes()
