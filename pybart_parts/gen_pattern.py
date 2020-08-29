import networkx as nx
from typing import List, Optional
import os
from .sample_types import AnnotatedSample
from .pattern_selectors import EdgeSelector, NodeSelector

BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class PatternGenerator(object):
    def __init__(
        self,
        trigger_node_selectors: List[NodeSelector],
        path_edge_selector: EdgeSelector,
        path_node_selectors: List[NodeSelector],
    ):
        self.trigger_node_selectors = trigger_node_selectors
        self.path_edge_selector = path_edge_selector
        self.path_node_selectors = path_node_selectors

    @staticmethod
    def _gen_node_pattern(node_idx, node_selectors: List[NodeSelector], ann_sample_: AnnotatedSample) -> List[str]:
        pattern_ = []
        for idx, node_selector in enumerate(node_selectors):
            pattern_.append("[" if idx == 0 else " & ")
            pattern_.append(node_selector.gen_pattern_elem(node_idx, ann_sample_))
            if idx == len(node_selectors) - 1:
                pattern_.append("]")

        return pattern_

    @staticmethod
    def _pattern_from_path(
        path: List[int], ann_sample_: AnnotatedSample, edge_selector: EdgeSelector, node_selectors: List[NodeSelector]
    ) -> str:
        pattern_ = []
        for idx, (s, e) in enumerate(zip(path, path[1:])):
            pattern_.append(edge_selector.gen_pattern_elem(s, e, ann_sample_))

            if idx < len(path) - 2:  # don't add a node restriction to the last path element
                pattern_.extend(PatternGenerator._gen_node_pattern(e, node_selectors, ann_sample_))
        return " ".join(pattern_)

    def generate_trigger_pattern(self, ann_sample: AnnotatedSample) -> str:
        subj_head = PatternGenerator._decide_head(ann_sample.digraph, ann_sample.subject_tok_range())
        if subj_head is None:
            raise ValueError("Specified subject is not a graph component")
        obj_head = PatternGenerator._decide_head(ann_sample.digraph, ann_sample.object_tok_range())
        if obj_head is None:
            raise ValueError("Specified object is not a graph component")
        trigger_head = PatternGenerator._decide_head(ann_sample.digraph, ann_sample.trigger_tok_range())
        if trigger_head is None:
            raise ValueError("Specified trigger is not a graph component")
        trigger_to_subj = nx.shortest_path(ann_sample.graph, source=trigger_head, target=subj_head)
        trigger_to_obj = nx.shortest_path(ann_sample.graph, source=trigger_head, target=obj_head)
        trigger_pattern = " ".join(
            PatternGenerator._gen_node_pattern(trigger_head, self.trigger_node_selectors, ann_sample)
        )
        subj_pattern = PatternGenerator._pattern_from_path(
            trigger_to_subj, ann_sample, self.path_edge_selector, self.path_node_selectors
        )
        obj_pattern = PatternGenerator._pattern_from_path(
            trigger_to_obj, ann_sample, self.path_edge_selector, self.path_node_selectors
        )
        pattern_ = "trigger={}\nsubject:{}={}\nobject:{}={}".format(
            trigger_pattern, ann_sample.subj_entity_type, subj_pattern, ann_sample.obj_entity_type, obj_pattern
        )
        return pattern_

    def generate_subject_object_pattern(self, ann_sample_: AnnotatedSample) -> str:
        subj_head = PatternGenerator._decide_head(ann_sample_.digraph, ann_sample_.subject_tok_range())
        if subj_head is None:
            raise ValueError("Specified subject is not a graph component")
        obj_head = PatternGenerator._decide_head(ann_sample_.digraph, ann_sample_.object_tok_range())
        if obj_head is None:
            raise ValueError("Specified object is not a graph component")
        subj_to_obj = nx.shortest_path(ann_sample_.graph, source=subj_head, target=obj_head)
        obj_pattern = PatternGenerator._pattern_from_path(
            subj_to_obj, ann_sample_, self.path_edge_selector, self.path_node_selectors
        )
        pattern_ = "subject:{}\nobject:{}={}".format(
            ann_sample_.subj_entity_type, ann_sample_.obj_entity_type, obj_pattern
        )
        return pattern_

    @staticmethod
    def _decide_head(g: nx.DiGraph, token_range: List[int]) -> Optional[int]:
        token_indices = set(token_range)
        for tok_idx in token_indices:
            other_tokens = token_indices - {tok_idx}
            if nx.descendants(g, tok_idx).issuperset(other_tokens):
                return tok_idx

        return None
