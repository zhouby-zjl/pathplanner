#include "astra-sim/system/AstraNetworkAPI.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/RecvPacketEventHadndlerData.hh"
#include "astra-sim/system/Common.hh"
#include "astra-sim/system/MockNcclLog.h"
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/rdma-client.h"
#include "entry.h"
#include <execinfo.h>
#include <fstream>
#include <iostream>
#include <queue>
#include <stdio.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
#include <jsoncpp/json/json.h>
#include "ns3/crc32.h"
#ifdef NS3_MTP
#include "ns3/mtp-interface.h"
#endif
#ifdef NS3_MPI
#include "ns3/mpi-interface.h"
#include <mpi.h>
#endif
#include <curl/curl.h>

#define RESULT_PATH "./ncclFlowModel_"

using namespace std;
using namespace ns3;

extern std::map<std::pair<std::pair<int, int>,int>, AstraSim::ncclFlowTag> receiver_pending_queue;
extern uint32_t node_num, switch_num, link_num, trace_num, nvswitch_num, gpus_per_server;
extern GPUType gpu_type;
extern std::vector<int>NVswitchs;

// string logPrefix;
map<uint32_t, vector<uint8_t>> seedsMap;
map<uint32_t, vector<uint32_t>> permMap;

struct sim_event {
	void *buffer;
	uint64_t count;
	int type;
	int dst;
	int tag;
	string fnType;
};


class ASTRASimNetwork : public AstraSim::AstraNetworkAPI {
private:
	int npu_offset;

public:
	queue<sim_event> sim_event_queue;
	ASTRASimNetwork(int rank, int npu_offset) : AstraNetworkAPI(rank) {
		this->npu_offset = npu_offset;
	}
	~ASTRASimNetwork() {}
	int sim_comm_size(AstraSim::sim_comm comm, int *size) { return 0; }
	int sim_finish() {
		for (auto it = nodeHash.begin(); it != nodeHash.end(); it++) {
			pair<int, int> p = it->first;
			if (p.second == 0) {
				std::cout << "sim_finish on sent, " << " Thread id: " << pthread_self() << std::endl;
				cout << "All data sent from node " << p.first << " is " << it->second
						<< "\n";
			} else {
				std::cout << "sim_finish on received, " << " Thread id: " << pthread_self() << std::endl;
				cout << "All data received by node " << p.first << " is " << it->second
						<< "\n";
			}
		}
		exit(0);
		return 0;
	}
	double sim_time_resolution() { return 0; }
	int sim_init(AstraSim::AstraMemoryAPI *MEM) { return 0; }
	AstraSim::timespec_t sim_get_time() {
		AstraSim::timespec_t timeSpec;
		timeSpec.time_val = Simulator::Now().GetNanoSeconds();
		return timeSpec;
	}
	virtual void sim_schedule(AstraSim::timespec_t delta,
			void (*fun_ptr)(void *fun_arg), void *fun_arg) {
		task1 t;
		t.type = 2;
		t.fun_arg = fun_arg;
		t.msg_handler = fun_ptr;
		t.schTime = delta.time_val;
		Simulator::Schedule(NanoSeconds(t.schTime), t.msg_handler, t.fun_arg);
		return;
	}
	virtual int sim_send(void *buffer,
			uint64_t count,
			int type,
			int dst,
			int tag,
			AstraSim::sim_request *request,
			void (*msg_handler)(void *fun_arg), void *fun_arg) {
		dst += npu_offset;
		task1 t;
		t.src = rank;
		t.dest = dst;
		t.count = count;
		t.type = 0;
		t.fun_arg = fun_arg;
		t.msg_handler = msg_handler;
		{
#ifdef NS3_MTP
			MtpInterface::explicitCriticalSection cs;
#endif
			sentHash[make_pair(tag, make_pair(t.src, t.dest))] = t;
#ifdef NS3_MTP
			cs.ExitSection();
#endif
		}
		cout << "--> SendFlow: " << rank << ", " << dst << ", " << count << endl;
		SendFlow(rank, dst, count, msg_handler, fun_arg, tag, request);
		return 0;
	}

	virtual int sim_recv(void *buffer, uint64_t count, int type, int src, int tag,
			AstraSim::sim_request *request,
			void (*msg_handler)(void *fun_arg), void *fun_arg) {
#ifdef NS3_MTP
		MtpInterface::explicitCriticalSection cs;
#endif
		MockNcclLog* NcclLog = MockNcclLog::getInstance();
		AstraSim::ncclFlowTag flowTag = request->flowTag;
		src += npu_offset;
		task1 t;
		t.src = src;
		t.dest = rank;
		t.count = count;
		t.type = 1;
		t.fun_arg = fun_arg;
		t.msg_handler = msg_handler;
		AstraSim::RecvPacketEventHadndlerData* ehd = (AstraSim::RecvPacketEventHadndlerData*) t.fun_arg;
		AstraSim::EventType event = ehd->event;
		tag = ehd->flowTag.tag_id;
		NcclLog->writeLog(NcclLogLevel::DEBUG,"接收事件注册 src %d sim_recv on rank %d tag_id %d channdl id %d",src,rank,tag,ehd->flowTag.channel_id);

		if (recvHash.find(make_pair(tag, make_pair(t.src, t.dest))) !=
				recvHash.end()) {
			uint64_t count = recvHash[make_pair(tag, make_pair(t.src, t.dest))];
			if (count == t.count) {
				recvHash.erase(make_pair(tag, make_pair(t.src, t.dest)));
				assert(ehd->flowTag.child_flow_id == -1 && ehd->flowTag.current_flow_id == -1);
				if(receiver_pending_queue.count(std::make_pair(std::make_pair(rank, src),tag))!= 0) {
					AstraSim::ncclFlowTag pending_tag = receiver_pending_queue[std::make_pair(std::make_pair(rank, src),tag)];
					receiver_pending_queue.erase(std::make_pair(std::make_pair(rank,src),tag));
					ehd->flowTag = pending_tag;
				}
#ifdef NS3_MTP
				cs.ExitSection();
#endif
				t.msg_handler(t.fun_arg);
				goto sim_recv_end_section;
			} else if (count > t.count) {
				recvHash[make_pair(tag, make_pair(t.src, t.dest))] = count - t.count;
				assert(ehd->flowTag.child_flow_id == -1 && ehd->flowTag.current_flow_id == -1);
				if(receiver_pending_queue.count(std::make_pair(std::make_pair(rank, src),tag))!= 0) {
					AstraSim::ncclFlowTag pending_tag = receiver_pending_queue[std::make_pair(std::make_pair(rank, src),tag)];
					receiver_pending_queue.erase(std::make_pair(std::make_pair(rank,src),tag));
					ehd->flowTag = pending_tag;
				}
#ifdef NS3_MTP
				cs.ExitSection();
#endif
				t.msg_handler(t.fun_arg);
				goto sim_recv_end_section;
			} else {
				recvHash.erase(make_pair(tag, make_pair(t.src, t.dest)));
				t.count -= count;
				expeRecvHash[make_pair(tag, make_pair(t.src, t.dest))] = t;
			}
		} else {
			if (expeRecvHash.find(make_pair(tag, make_pair(t.src, t.dest))) ==
					expeRecvHash.end()) {
				expeRecvHash[make_pair(tag, make_pair(t.src, t.dest))] = t;
				NcclLog->writeLog(NcclLogLevel::DEBUG," 网络包后到，先进行注册 recvHash do not find expeRecvHash.new make src  %d dest  %d t.count:  %d channel_id  %d current_flow_id  %d",t.src,t.dest,t.count,tag,flowTag.current_flow_id);

			} else {
				uint64_t expecount =
						expeRecvHash[make_pair(tag, make_pair(t.src, t.dest))].count;
				NcclLog->writeLog(NcclLogLevel::DEBUG," 网络包后到，重复注册 recvHash do not find expeRecvHash.add make src  %d dest  %d expecount:  %d t.count:  %d tag_id  %d current_flow_id  %d",t.src,t.dest,expecount,t.count,tag,flowTag.current_flow_id);

			}
		}
#ifdef NS3_MTP
		cs.ExitSection();
#endif

		sim_recv_end_section:
		return 0;
	}
	void handleEvent(int dst, int cnt) {
	}
};

struct user_param {
	int thread;
	string workload;
	string network_topo;
	string network_conf;
	bool only_gen_routes_and_ports;
	user_param() {
		thread = 1;
		workload = "";
		network_topo = "";
		network_conf = "";
		only_gen_routes_and_ports = false;
	};
	~user_param(){};
};

bool ONLY_PLAN_PATHS = false;

static int user_param_prase(int argc,char * argv[],struct user_param* user_param){
	int opt;
	while ((opt = getopt(argc,argv,"hrt:w:g:sn:c:l:y"))!=-1){
		switch (opt)
		{
		case 'h':
			/* code */
			std::cout << "-t    number of threads,default 1"<<std::endl;
			std::cout << "-w    workloads default none "<<std::endl;
			std::cout << "-n    network topo"<<std::endl;
			std::cout << "-c    network_conf"<<std::endl;
			std::cout << "-l	log path prefix of HPS" << std::endl;
			std::cout << "-r    only generate routes and ports output files under the log path" << std::endl;
			std::cout << "-s    whether to use HPS to perform path selection" << std::endl;
			std::cout << "-y    only plan paths" << std::endl;
			return 1;
			break;
		case 't':
			user_param->thread = stoi(optarg);
			break;
		case 'w':
			user_param->workload = optarg;
			break;
		case 'n':
			user_param->network_topo = optarg;
			break;
		case 'c':
			user_param->network_conf = optarg;
			break;
		case 'l':
			AstraSim::logPrefix = optarg;
			break;
		case 'r':
			user_param->only_gen_routes_and_ports = true;
			break;
		case 's':
			USE_HPS_FLOW = true;
			break;
		case 'y':
			ONLY_PLAN_PATHS = true;
			break;
		default:
			std::cerr<<"-h    help message"<<std::endl;
			return 1;
		}
	}
	return 0 ;
}

int MyLinearityEcmpRouting(Ptr<SwitchNode> node, Ptr<const Packet> pkt, const CustomHeader& ch) {
	auto entry = node->m_rtTable.find(ch.dip);
	if (entry == node->m_rtTable.end()) {
		return -1;
	}

	const auto& nexthops = entry->second;
	vector<int> nexthops_ = nexthops;
	std::sort(nexthops_.begin(), nexthops_.end());

	// 构造12字节哈希输入: [sip][dip][sport][dport]
	uint8_t buf[12];
	buf[0] = (ch.sip >> 24) & 0xff;
	buf[1] = (ch.sip >> 16) & 0xff;
	buf[2] = (ch.sip >> 8) & 0xff;
	buf[3] = ch.sip & 0xff;

	buf[4] = (ch.dip >> 24) & 0xff;
	buf[5] = (ch.dip >> 16) & 0xff;
	buf[6] = (ch.dip >> 8) & 0xff;
	buf[7] = ch.dip & 0xff;

	uint16_t sport = 0, dport = 0;
	if (ch.l3Prot == 0x6) {
		sport = ch.tcp.sport;
		dport = ch.tcp.dport;
	} else if (ch.l3Prot == 0x11) {
		sport = ch.udp.sport;
		dport = ch.udp.dport;
		TracePath(node->GetId(), ch.sip, ch.dip, sport, dport);
	} else if (ch.l3Prot == 0xFC || ch.l3Prot == 0xFD) {
		sport = ch.ack.sport;
		dport = ch.ack.dport;
	}

	buf[8]  = (sport >> 8) & 0xff;
	buf[9]  = sport & 0xff;
	buf[10] = (dport >> 8) & 0xff;
	buf[11] = dport & 0xff;

	// Apply permutation
	vector<uint32_t>& m_permutation = permMap[node->GetId()];
	vector<uint8_t> ba_perm (m_permutation.size ());
	for (size_t i = 0; i < m_permutation.size (); ++i)
	{
		ba_perm[i] = buf[m_permutation[i]];
	}

	// XOR with seed
	vector<uint8_t>& m_seed = seedsMap[node->GetId()];
	vector<uint8_t> xored;
	for (size_t i = 0; i < ba_perm.size (); ++i)
	{
		xored.push_back (ba_perm[i] ^ m_seed[i]);
	}

	// CRC32
	uint32_t h = CRC32Calculate (xored.data (), xored.size ());
	int32_t hv = h % nexthops_.size();

	return nexthops_[hv];
}


void LoadSeedsAndPermutations(const string& seedsFile,
		const string& permFile) {
	ifstream seedsIn(seedsFile);
	ifstream permIn(permFile);
	string line;

	// Parse seeds
	while (getline(seedsIn, line)) {
		stringstream ss(line);
		string idStr, valuesStr;
		if (getline(ss, idStr, ':') && getline(ss, valuesStr)) {
			uint32_t nodeId = stoul(idStr);
			vector<uint8_t> seed;
			stringstream vs(valuesStr);
			string num;
			while (getline(vs, num, ',')) {
				seed.push_back(static_cast<uint8_t>(stoi(num)));
			}
			seedsMap[nodeId] = seed;
		}
	}

	// Parse permutations
	while (getline(permIn, line)) {
		stringstream ss(line);
		string idStr, valuesStr;
		if (getline(ss, idStr, ':') && getline(ss, valuesStr)) {
			uint32_t nodeId = stoul(idStr);
			vector<uint32_t> perm;
			stringstream vs(valuesStr);
			string num;
			while (getline(vs, num, ',')) {
				perm.push_back(static_cast<uint32_t>(stoi(num)));
			}
			permMap[nodeId] = perm;
		}
	}

}

// CURL 写回调函数
size_t path_plan_write_callback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

bool PlanPathsv2() {
    auto map_ps_to_group_to_pairs = AstraSim::Sys::get_nccl_comms_topology_str_v2();

    // root_json 顶层应为 object（每个 ps -> object）
    Json::Value root_json(Json::objectValue);
    std::map<std::string, std::map<int, std::vector<std::string>>> ret_data;

    for (const auto& kv : map_ps_to_group_to_pairs) {
        AstraSim::ParallelStrategy ps_enum = kv.first;
        const auto& group_map = kv.second;

        std::string ps_str;
        switch (ps_enum) {
            case AstraSim::ParallelStrategy::TP:    ps_str = "TP"; break;
            case AstraSim::ParallelStrategy::DP:    ps_str = "DP"; break;
            case AstraSim::ParallelStrategy::EP:    ps_str = "EP"; break;
            case AstraSim::ParallelStrategy::DP_EP: ps_str = "DP_EP"; break;
            case AstraSim::ParallelStrategy::PP:    ps_str = "PP"; break;
            case AstraSim::ParallelStrategy::NONE:  ps_str = "NONE"; break;
            default: ps_str = "UNKNOWN"; break;
        }

        // ps 对应一个 object： { "groupId": [ ... ] }
        if (!root_json.isMember(ps_str)) {
            root_json[ps_str] = Json::Value(Json::objectValue);
        }

        // 遍历每个 groupId -> set<pair<int,int>>
        for (const auto& gm : group_map) {
            int groupId = gm.first;
            const auto& pair_set = gm.second;

            Json::Value pairs_json(Json::arrayValue);
            std::vector<std::string> pairs_str;

            for (const auto& pr : pair_set) {
                int node_a = pr.first;
                int node_b = pr.second;
                const auto& nextHops = nextHop[n.Get(node_a)][n.Get(node_b)];
                bool has_node_type_1 = false;
                for (const auto& nh : nextHops) {
                    if (nh && nh->GetNodeType() == 1) {
                        has_node_type_1 = true;
                        break;
                    }
                }
                if (!has_node_type_1) continue;

                std::string s = std::to_string(node_a) + "-" + std::to_string(node_b);
                pairs_json.append(s);
                pairs_str.push_back(s);
            }

            if (!pairs_str.empty()) {
                // JSON 的 key 是字符串，所以把 groupId 转为 string
                root_json[ps_str][std::to_string(groupId)] = pairs_json;
                ret_data[ps_str][groupId] = std::move(pairs_str);
            }
        }
    }

    // 转为字符串
    Json::StreamWriterBuilder writer_builder;
    std::string postData = Json::writeString(writer_builder, root_json);

    // 2. 发起HTTP请求
    std::string response_data;
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Failed to init curl" << std::endl;
        return false;
    }

    curl_easy_setopt(curl, CURLOPT_URL, "http://127.0.0.1:5000/api/planpaths");
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, postData.c_str());
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, path_plan_write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);

    CURLcode res = curl_easy_perform(curl);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        std::cerr << "CURL POST failed: " << curl_easy_strerror(res) << std::endl;
        return false;
    }

    // 3. 解析返回JSON
    Json::CharReaderBuilder reader_builder;
    Json::Value response_json;
    std::string errs;
    std::istringstream iss(response_data);
    if (!Json::parseFromStream(reader_builder, iss, &response_json, &errs)) {
        std::cerr << "Failed to parse response JSON: " << errs << std::endl;
        return false;
    }

    // 4. 处理JSON，格式：
    // key = "PS|src-dst", value = [port list]
    // 解析填充 ps_to_pair_to_sport_all
    ps_to_pair_to_sport_all.clear();

    for (const auto& ps : response_json.getMemberNames()) {
        if (ps == "dport") continue;

        const Json::Value& pair_map = response_json[ps];
        if (!pair_map.isObject()) {
            std::cerr << "Expected object for ps " << ps << std::endl;
            continue;
        }

        for (const auto& pair_key : pair_map.getMemberNames()) {
            const Json::Value& port_array = pair_map[pair_key];
            // pair_key 格式是 "src-dst"
            size_t dash_pos = pair_key.find('-');
            if (dash_pos == std::string::npos) {
                std::cerr << "Invalid pair format in key: " << pair_key << std::endl;
                continue;
            }
            int src = std::stoi(pair_key.substr(0, dash_pos));
            int dst = std::stoi(pair_key.substr(dash_pos + 1));
            std::pair<int, int> pair_ = {src, dst};

            if (!port_array.isArray()) {
                std::cerr << "Expected array for pair key " << pair_key << std::endl;
                continue;
            }

            std::vector<int> port_list;
            for (const auto& port_val : port_array) {
                if (!port_val.isInt()) {
                    std::cerr << "Non-int port value for pair key " << pair_key << std::endl;
                    continue;
                }
                port_list.push_back(port_val.asInt());
            }
            if (port_list.empty()) {
                std::cerr << "No valid ports for pair key " << pair_key << std::endl;
                continue;
            }

            //std::vector<bool> is_sport_allocated(port_list.size(), false);
            auto sai = std::make_shared<sport_allocation_info>(port_list);
            sport_allocated[src][dst];
            ps_to_pair_to_sport_all[ps][pair_] = sai;
        }
    }


    // 5. （可选）打印检查
    for (const auto& [ps, pair_map] : ps_to_pair_to_sport_all) {
        std::cout << "PS: " << ps << std::endl;
        for (const auto& [p, sai_ptr] : pair_map) {
            std::cout << "  Pair (" << p.first << "," << p.second << ") ports: ";
            for (int port : sai_ptr->sport_all) std::cout << port << " ";
            std::cout << std::endl;
        }
    }

    return true;
}


int main(int argc, char *argv[]) {
	curl_global_init(CURL_GLOBAL_ALL);
	outfile_sport_allocation.open("/tmp/sport_allocation", std::ios::out);

	RdmaClient::REPORT_TO_API_SERVER = true;
	USE_HPS_FLOW = false;
	REPORT_FLOW_INIT = true;
	ONLY_PLAN_PATHS = false;

	struct user_param user_param;
	MockNcclLog::set_log_name("SimAI.log");
	MockNcclLog* NcclLog = MockNcclLog::getInstance();
	NcclLog->writeLog(NcclLogLevel::INFO," init SimAI.log ");

	if(user_param_prase(argc,argv,&user_param)){
		curl_global_cleanup();
		return 0;
	}

	string seedsFile = AstraSim::logPrefix + "seeds";
	string permutationsFile = AstraSim::logPrefix + "permutations";
	string filePathServerAddresses = AstraSim::logPrefix + "simai-server-addresses";
	string writeFilePathRouting = AstraSim::logPrefix + "simai-routing";
	string writeFilePathPorts = AstraSim::logPrefix + "simai-ports";

#ifdef NS3_MTP
	MtpInterface::Enable(user_param.thread);
#endif

	main1(user_param.network_topo, user_param.network_conf, true, true, true,
			filePathServerAddresses, writeFilePathRouting, writeFilePathPorts);

	if (user_param.only_gen_routes_and_ports) {
		curl_global_cleanup();
		return 0;
	}

	for (uint32_t i = 0; i < n.GetN(); ++i) {
		Ptr<Node> node = n.Get(i);
		if (node->GetNodeType() != 1) {
			continue;
		}
		Ptr<SwitchNode> sw = DynamicCast<SwitchNode>(n.Get(i));
		sw->SetOutDevCallback(&MyLinearityEcmpRouting);
	}
	LoadSeedsAndPermutations(seedsFile, permutationsFile);

	int nodes_num = node_num - switch_num;
	int gpu_num = node_num - nvswitch_num - switch_num;

	std::map<int, int> node2nvswitch;
	for(int i = 0; i < gpu_num; ++ i) {
		node2nvswitch[i] = gpu_num + i / gpus_per_server;
	}
	for(int i = gpu_num; i < gpu_num + nvswitch_num; ++ i){
		node2nvswitch[i] = i;
		NVswitchs.push_back(i);
	}

	LogComponentEnable("OnOffApplication", LOG_LEVEL_INFO);
	LogComponentEnable("PacketSink", LOG_LEVEL_INFO);
	LogComponentEnable("GENERIC_SIMULATION", LOG_LEVEL_INFO);

	std::vector<ASTRASimNetwork *> networks(nodes_num, nullptr);
	std::vector<AstraSim::Sys *> systems(nodes_num, nullptr);

	for (int j = 0; j < nodes_num; j++) {
		networks[j] =
				new ASTRASimNetwork(j ,0);
		systems[j ] = new AstraSim::Sys(
				networks[j],
				nullptr,
				j,
				0,
				1,
				{nodes_num},
				{1},
				"",
				user_param.workload,
				1,
				1,
				1,
				1,
				0,
				RESULT_PATH,
				"test1",
				true,
				false,
				gpu_type,
				{gpu_num},
				NVswitchs,
				gpus_per_server
		);
		systems[j ]->nvswitch_id = node2nvswitch[j];
		systems[j ]->num_gpus = nodes_num - nvswitch_num;
	}

	// ================ added by zby ========================
	//PlanPaths();
	PlanPathsv2();
	if (ONLY_PLAN_PATHS) {
		curl_global_cleanup();
		return 0;
	}
	// ======================================================

	for (int i = 0; i < nodes_num; i++) {
		systems[i]->workload->fire();
	}

	std::cout << "simulator run " << std::endl;
	reportQbbMaxTxTimeEnd = ns3::Seconds(1);
	//Simulator::Schedule(ns3::Seconds(0), &PeriodicQbbMacTxReport);
	Simulator::Run();
	Simulator::Stop(Seconds(2000000000));
	 std::chrono::high_resolution_clock::time_point end_sim_time = std::chrono::high_resolution_clock::now();

	 // ====================== added by zby =================
	 	 AstraSim::Tick iteration_time = AstraSim::Sys::boostedTick();
	 	 std::cout << "--> iteration_time_in_ns: " <<  iteration_time << std::endl;
	 	 string writeFilePathIterationTime = AstraSim::logPrefix + "iteration_time";
	 	 stringstream ss;
	 	 ss << iteration_time;
	 	 std::ofstream file(writeFilePathIterationTime);
	 	 file << ss.str();
	 	 file.close();
	 // ======================================================

	Simulator::Destroy();

#ifdef NS3_MPI
	MpiInterface::Disable ();
#endif
	outfile_sport_allocation.close();
	curl_global_cleanup();
	return 0;
}
