from typing import Tuple, List
import networkx as nx

from .definitions import Relation


class Sample(object):
    def __init__(
        self,
        raw: str,
        text: str,
        subj_char_offsets: Tuple[int, int],
        obj_char_offsets: Tuple[int, int],
        trigger_char_offsets: Tuple[int, int],
        relation="${label}",
        subj_entity_type="${subject-type}",
        obj_entity_type="${object-type}",
        negative=False,
    ):
        self.raw = raw
        self.text = text
        self.subj_char_offsets = subj_char_offsets
        self.obj_char_offsets = obj_char_offsets
        self.trigger_char_offsets = trigger_char_offsets
        self.relation = relation
        self.subj_entity_type = subj_entity_type
        self.obj_entity_type = obj_entity_type
        self.negative = negative

    @classmethod
    def from_empty(cls):
        return cls("", "", None, None, None)


class AnnotatedSample(object):
    def __init__(
        self,
        raw: str,
        text: str,
        relation: str,
        subj_entity_type: str,
        obj_entity_type: str,
        tokens: List[str],
        tags: List[str],
        entities: List[str],
        chunks: List[str],
        lemmas: List[str],
        subj_tok_offsets: Tuple[int, int],
        obj_tok_offsets: Tuple[int, int],
        trigger_tok_offsets: Tuple[int, int],
        graph: nx.Graph,
        digraph: nx.DiGraph,
    ):
        self.raw = raw
        self.text = text
        self.relation = relation
        self.subj_entity_type = subj_entity_type
        self.obj_entity_type = obj_entity_type
        self.tokens = tokens
        self.tags = tags
        self.entities = entities
        self.chunks = chunks
        self.lemmas = lemmas
        self.subj_tok_offsets = subj_tok_offsets
        self.obj_tok_offsets = obj_tok_offsets
        self.trigger_tok_offsets = trigger_tok_offsets
        self.graph = graph
        self.digraph = digraph

    def subject_tok_range(self):
        return range(self.subj_tok_offsets[0], self.subj_tok_offsets[1])

    def object_tok_range(self):
        return range(self.obj_tok_offsets[0], self.obj_tok_offsets[1])

    def trigger_tok_range(self):
        return range(self.trigger_tok_offsets[0], self.trigger_tok_offsets[1])

    def update_relation_data(self, relation: Relation):
        self.relation = relation.label
        self.subj_entity_type = relation.subject_entity_type
        self.obj_entity_type = relation.object_entity_type
