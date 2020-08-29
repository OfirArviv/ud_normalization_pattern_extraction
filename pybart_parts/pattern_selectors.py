"""Pattern Element Selectors

This module contains the class hierarchy of pattern element selectors. To create a pattern, the dep tree edges and nodes
are traversed from subject to object (or from trigger to subject and trigger to object). When a node is traversed, node
selectors will determine how a corresponding node pattern element will be created (e.g. if the node selectors
WordNodeSelector and PosTagNodeSelector are used, then a node for "John" will result in the pattern element
[ word = John && tag = NNP ]). Likewise, when an edge is traversed and edge selector will determine how to create the
corresponding edge pattern element.

"""
from typing import List
from .sample_types import AnnotatedSample


class ElemSelector(object):
    def __init__(self, name, desc):
        self.name = name
        self.desc = desc


class NodeSelector(ElemSelector):
    def __init__(self, name, desc):
        super().__init__(name, desc)

    def gen_pattern_elem(self, elem_idx: int, ann_sample: AnnotatedSample):
        raise NotImplementedError()


class EdgeSelector(ElemSelector):
    def __init__(self, name, desc):
        super().__init__(name, desc)

    def gen_pattern_elem(self, head_idx: int, dep_idx: int, ann_sample: AnnotatedSample):
        raise NotImplementedError()


class LabelEdgeSelector(EdgeSelector):
    def __init__(self):
        super().__init__("label", "adds edge labels to the pattern")

    def gen_pattern_elem(self, start_idx: int, end_idx: int, ann_sample: AnnotatedSample):
        digraph = ann_sample.digraph
        if digraph.has_edge(start_idx, end_idx):
            # TODO: I needed to add [0], why?
            return ">" + digraph[start_idx][end_idx][0]["label"].split(":")[0]
        elif digraph.has_edge(end_idx, start_idx):
            return "<" + digraph[end_idx][start_idx][0]["label"].split(":")[0]
        else:
            raise ValueError("invalid input path.")


class DirectionEdgeSelector(EdgeSelector):
    def __init__(self):
        super().__init__("direction", "adds edge direction to the pattern")

    def gen_pattern_elem(self, head_idx: int, dep_idx: int, ann_sample: AnnotatedSample):
        digraph = ann_sample.digraph
        if digraph.has_edge(head_idx, dep_idx):
            return ">>"
        elif digraph.has_edge(dep_idx, head_idx):
            return "<<"
        else:
            raise ValueError("invalid input path.")


class PosTagNodeSelector(NodeSelector):
    def __init__(self):
        super().__init__("pos", "adds node pos tags to the pattern")

    def gen_pattern_elem(self, elem_idx: int, ann_sample: AnnotatedSample):
        return f"tag={ann_sample.tags[elem_idx]}"


class LemmaNodeSelector(NodeSelector):
    def __init__(self):
        super().__init__("lemma", "adds node lemma to the pattern")

    def gen_pattern_elem(self, elem_idx: int, ann_sample: AnnotatedSample):
        return f'lemma="{ann_sample.lemmas[elem_idx]}"'


class EntityNodeSelector(NodeSelector):
    def __init__(self):
        super().__init__("entity", "adds node entity to the pattern")

    def gen_pattern_elem(self, elem_idx: int, ann_sample: AnnotatedSample):
        return f"entity={ann_sample.entities[elem_idx]}"


class WordNodeSelector(NodeSelector):
    def __init__(self):
        super().__init__("word", "adds node word to the pattern")

    def gen_pattern_elem(self, elem_idx: int, ann_sample: AnnotatedSample):
        return f'word="{ann_sample.tokens[elem_idx]}"'


class TriggerVarNodeSelector(NodeSelector):
    def __init__(self, triggers: List[str]):
        super().__init__("trigger", "adds node word to the pattern using a trigger variable")
        self.triggers = triggers

    def gen_pattern_elem(self, elem_idx: int, ann_sample: AnnotatedSample):
        triggers_regex_str = "/" + "|".join(self.triggers) + "/"
        return f"word={triggers_regex_str}"


def get_pattern_node_selectors():
    return [WordNodeSelector(), LemmaNodeSelector(), PosTagNodeSelector(), EntityNodeSelector()]


def get_pattern_edge_selectors():
    return [LabelEdgeSelector(), DirectionEdgeSelector()]


def get_selectors_by_name(selector_names: List[str]):
    all_selectors = dict(
        [
            (selector.name, selector)
            for selector in set(get_pattern_node_selectors()) | set(get_pattern_edge_selectors())
        ]
    )

    return [all_selectors[name] for name in selector_names]
