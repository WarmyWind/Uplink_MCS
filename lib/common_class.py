import numpy as np
from config.settings import *

class UE:
    def __init__(self, no, tra):
        self.no = no  # UE编号
        # self.record_len = record_len
        self.tra = tra
        self.posi = None
        self.Ptmax_dBm = User.Ptmax_dBm
        self.Ptmax = User.Ptmax
        self.history_arrival_power = []
        # self.posi_record = [posi for _ in range(record_len)]
        # self.future_posi = [posi for _ in range(record_len)]
        # self.type = type
        # self.type_no = type_no  # 对应类型中的UE编号

        # self.GBR_flag = GBR_flag
        # self.min_rate = min_rate

        # self.active = active
        # self.Rreq = 0
        # self.state = 'unserved'  # 'served', 'unserved'
        self.serv_BS = -1
        self.HO_type = 'none'  # 'none' or 'ideal HO'
        self.arrival_power = 0
        self.SINR_dB = -np.Inf
        # self.serv_BS_L3_h = None  # 服务基站的信道功率L3测量值
        # self.ToS = -1  # 在当前服务小区的停留时间
        # self.MTS = 100  # 最小停留时间参数
        # self.RB_Nt_ocp = []  # 占用的RB_Nt,列表内的元素是元组（RB，Nt）
        #
        # self.neighbour_BS = []  # 邻基站列表,信道由好到差排序
        # self.neighbour_BS_L3_h = []  # 邻基站的信道功率L3测量值,由好到差
        # self.all_BS_L3_h_record = []
        # self.posi_type = None
        # self.RB_type = None

class ResourceMap:
    def __init__(self, nRB, nNt, center_RB_idx, edge_RB_idx):
        self.map = np.zeros((nRB, nNt)) - 1  # 记录各个资源块上服务的UE序号
        self.RB_ocp = [np.array([]) for _ in range(nRB)]  # 记录各个RB在哪些天线上服务
        self.RB_idle_antenna = [np.array(range(nNt)) for _ in range(nRB)]  # 记录各个RB在哪些天线上空闲
        self.RB_ocp_num = np.zeros((nRB,))  # 记录各个RB在多少天线上服务
        self.RB_sorted_idx = np.array(range(nRB))  # 由少到多排列占用最少的RB，以序号表示对应的RB
        self.serv_UE_list = []
        # self.extra_edge_RB_serv_list = np.array([])
        # self.extra_center_RB_serv_list = np.array([])
        #
        # self.center_RB_sorted_idx = center_RB_idx
        # self.edge_RB_sorted_idx = edge_RB_idx


class ServingMap:
    def __init__(self, nBS, nUE):
        self.map = np.zeros((nBS, nUE))

    def update(self, new_map):
        self.map = new_map

    def change_by_entry(self, BS_no, UE_no, state):
        self.map[BS_no, UE_no] = state

    def query_by_UE(self, UE_no):
        col = self.map[:, UE_no]
        return np.where(col > 0)

    def query_by_BS(self, BS_no):
        row = self.map[BS_no, :]
        return np.where(row > 0)


class BS:
    def __init__(self, no, posi):
        self.no = no
        self.nRB = nRB
        self.posi = posi
        self.nNt = Macro.nNt
        self.Ptmax_dBm = Macro.Ptmax_dBm
        self.Ptmax = Macro.Ptmax
        self.history_interference = []

        # self.RB_per_UE = RB_per_UE
        # self.opt_UE_per_RB = opt_UE_per_RB
        # self.MaxUE_per_RB = MaxUE_per_RB
        # self.active = active
        # self.resource_map = ResourceMap(nRB, nNt, center_RB_idx, edge_RB_idx)
        # self.max_edge_UE_num = 0  # 小区的最大边缘UE数
        # self.low_SINR_nUE_in_range = 0  # 小区内SINR低于一定门限的UE数
        # self.nUE_in_range = 0  # 小区范围内的UE数（包括未服务的）
        # self.UE_in_range = []
        # self.edge_UE_in_range = []  # 小区范围内的边缘UE（包括未服务的）,以SINR由小到大排序
        # self.center_UE_in_range = []  # 小区范围内的中心UE（包括未服务的）,以SINR由小到大排序
        # self.center_RB_idx = center_RB_idx
        # self.edge_RB_idx = edge_RB_idx
        # self.serv_UE_list_record = []
        # self.RB_ocp_num_record = []
        # self.ICIC_group = ICIC_gruop