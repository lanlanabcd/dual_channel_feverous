# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2020/11/30 21:12
# Description:
from .base_graph_data_generator import BaseGraphGenerator
from .roberta_pair_generator import RobertaPairGenerator
from .new_roberta_generator import NewRobertaGenerator
from .multi_task_generator import MultiTaskGenerator
from .global_distill_generator import GlobalDistillGenerator
from .evi_masker_generator import EviMaskerGenerator
from .subevis_generator import SubevisGenerator
from .subevis_sep_generator import SubevisSepGenerator

from .base_data_generator import BaseGenerator
from .subevis_gnn_generator import SubevisGnnGenerator
from .subevis_gnn_w_label_generator import SubevisGnnWLabelGenerator
from .subevis_gnn_w_global_label_generator import SubevisGnnWGlobalLabelGenerator
