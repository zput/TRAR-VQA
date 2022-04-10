# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import numpy as np
import glob, json, torch, random
import torch.utils.data as Data
import torch.nn as nn
from openvqa.utils.feat_filter import feat_filter

class BaseDataSet(Data.Dataset):
    def __init__(self):
        self.token_to_ix = None
        self.pretrained_emb = None
        self.ans_to_ix = None
        self.ix_to_ans = None

        self.data_size = None
        self.token_size = None
        self.ans_size = None
        #self.valid_idx = [ 9, 25, 30, 34, 36, 49, 61, 64, 71, 72, 77, 78, 81, 86, 89, 92, 94, 109, 110, 113, 127, 138, 142, 144, 149, 154, 165, 194, 201, 247, 250, 260, 307, 308, 309, 312, 315, 322, 332, 368, 370, 382, 389, 394, 404, 419, 431, 438, 443, 450, 471, 508, 510, 514, 529, 531, 532, 540, 542, 562, 572, 575, 4376, 4377, 8747, 21826, 21895, 21926, 34933, 37186, 39288, 43697, 43773, 43780, 101431, 109277, 109986, 112497, 114353, 131074, 131075, 131084, 131087, 131093, 131101, 131113, 131118, 131126, 131127, 131128, 131133, 131160, 131172, 131174, 131190, 131197, 131208, 131215, 131225, 131245, 131277, 131279, 131299, 131300, 131312, 131315, 131323, 131330, 131339, 131342, 131351, 131352, 131366, 131373, 131374, 131376, 131388, 131400, 131415, 131419, 131427, 131434, 131450, 131465, 131470, 131486, 131487, 131498, 131509, 131511, 131524, 131564, 131565, 131579, 131589, 131595, 131613, 131621, 137045, 137918, 139105, 152974, 157125, 165859, 170235, 178957, 181566, 192053, 196663, 196688, 200782, 213863, 218508, 231338, 239873, 240304, 248779, 257513, 262146, 262159, 262171, 262172, 262180, 262184, 262187, 262191, 262201, 262204, 262207, 262221, 262260, 262261, 262273, 262283, 262285, 262299, 262308, 262329, 262336, 262359, 262389, 262393, 262399, 262415, 262442, 262454, 262463, 262465, 262477, 262492, 262495, 262508, 262519, 262521, 262529, 262541, 262544, 262545, 262549, 262550, 262552, 262554, 262561, 262588, 262599, 262603, 262619, 262623, 262662, 262670, 262683, 262688, 262690, 262691, 262692, 262704, 262705, 262707, 262710, 262715, 262718, 265364, 270612, 283700, 283704, 284012, 284084, 305527, 305853, 305901, 327754, 336077, 344814, 349607, 370986, 393221, 393223, 393224, 393227, 393228, 393230, 393242, 393251, 393268, 393286, 393290, 393291, 393292, 393294, 393297, 393306, 393311, 393317, 393362, 393375, 393379, 393384, 393386, 393394, 393396, 393403, 393412, 393418, 393419, 393422, 393428, 393432, 393438, 393442, 393445, 393464, 393480, 393488, 393489, 393493, 393503, 393508, 393534, 393542, 393546, 393575, 393592, 393602, 393608, 393611, 393629, 393634, 393641, 393649, 393656, 393664, 393677, 393680, 393686, 393696, 393699, 393705, 393714, 393719, 393721, 393735, 393738, 393744, 393781, 397186, 398917, 409739, 414639, 415089, 436929, 436975, 436990, 438196, 448671, 458752, 458763, 458785, 471373, 474858, 480683, 497565, 505583, 510665, 524291, 524297, 524311, 524314, 524320, 524325, 524338, 524340, 524375, 524377, 524386, 524420, 524428, 524470, 524471, 524476, 524486, 524508, 524518, 524520, 524522, 524525, 524547, 524551, 524557, 524572, 524594, 524613, 524623, 524625, 524628, 524645, 524648, 524649, 524651, 524661, 524662, 524672, 524676, 524679, 524690, 524695, 524709, 524710, 524718, 524723, 524724, 524730, 524766, 524788, 524790, 524838, 524866, 536855, 546151, 546179, 546218, 554301, 554854, 567990, 568052576128 ]

    def load_ques_ans(self, idx):
        raise NotImplementedError()


    def load_img_feats(self, idx, iid):
        raise NotImplementedError()


    def __getitem__(self, idx):
        #idx = self.valid_idx[idx]

        ques_ix_iter, ans_iter, iid = self.load_ques_ans(idx)

        value, frcn_feat_iter, grid_feat_iter, bbox_feat_iter = self.load_img_feats(idx, iid)
        if ( value == False ):
            return \
                torch.from_numpy(np.zeros(1,dtype=int)),\
                torch.from_numpy(frcn_feat_iter),\
                torch.from_numpy(grid_feat_iter),\
                torch.from_numpy(bbox_feat_iter),\
                torch.from_numpy(ques_ix_iter),\
                torch.from_numpy(ans_iter)
        else:
            return \
                torch.from_numpy(np.ones(1,dtype=int)),\
                torch.from_numpy(frcn_feat_iter),\
                torch.from_numpy(grid_feat_iter),\
                torch.from_numpy(bbox_feat_iter),\
                torch.from_numpy(ques_ix_iter),\
                torch.from_numpy(ans_iter)


    def __len__(self):
        # return len(self.valid_idx)
        return self.data_size

    def shuffle_list(self, list):
        random.shuffle(list)


class BaseAdapter(nn.Module):
    def __init__(self, __C):
        super(BaseAdapter, self).__init__()
        self.__C = __C
        if self.__C.DATASET in ['vqa']:
            self.vqa_init(__C)

        elif self.__C.DATASET in ['clevr']:
            self.clevr_init(__C)

        else:
            exit(-1)

        # eval('self.' + __C.DATASET + '_init()')

    def vqa_init(self, __C):
        raise NotImplementedError()

    def clevr_init(self, __C):
        raise NotImplementedError()

    def forward(self, frcn_feat, grid_feat, bbox_feat):
        feat_dict = feat_filter(self.__C.DATASET, frcn_feat, grid_feat, bbox_feat)

        if self.__C.DATASET in ['vqa']:
            return self.vqa_forward(feat_dict)

        elif self.__C.DATASET in ['clevr']:
            return self.clevr_forward(feat_dict)

        else:
            exit(-1)

    def vqa_forward(self, feat_dict):
        raise NotImplementedError()

    def clevr_forward(self, feat_dict):
        raise NotImplementedError()

