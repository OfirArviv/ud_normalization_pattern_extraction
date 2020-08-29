from __future__ import annotations
from typing import List, Dict, Optional, Iterator, cast, Tuple
from itertools import chain
from collections import defaultdict
from dataclasses import dataclass, replace, field

class Relation:
    label: str = field(metadata={"description": "The label of the relation, such as org:founded"})
    name: str = field(metadata={"description": "A human readable name of the relation"})
    subject_entity_type: str = field(metadata={"description": "The type of entity for the subject of this relation"})
    object_entity_type: str = field(metadata={"description": "The type of entity for the object of this relation"})



# --------------------------------------------------------------------------------- #
# Immutable value objects that represent the annotated document and it's components #
# --------------------------------------------------------------------------------- #

TokenIndex = int


@dataclass(frozen=True)
class Span:
    """
    A span of tokens
    """

    first: TokenIndex
    last: TokenIndex

    @staticmethod
    def single(index: TokenIndex) -> Span:
        """
        A builder method to create a span over a single token
        Args:
            index: the index of a token

        Returns: a span

        """
        return Span(index, index)

    def overlaps(self, other: Span) -> bool:
        """
        Check if this span overlaps the provided span
        Args:
            other: the other span to check against

        Returns:
            'true' if the spans overlap, false otherwise
        """
        if self.first == other.first or self.last == other.last:
            return True
        elif self.first < other.first:
            return self.last >= other.first
        elif self.first > other.first:
            return self.first <= other.last
        else:
            return False

    def __len__(self):
        return self.last - self.first + 1

    def __contains__(self, item: int) -> bool:
        return self.first <= item <= self.last

    def indices(self) -> Iterator[int]:
        return range(self.first, self.last + 1)


@dataclass(frozen=True)
class LabeledSpan(Span):
    """
    A span of tokens with a label
    """

    label: str

    @property
    def unlabeled(self) -> Span:
        """
        Convert to an unlabeled span representation, throwing away the label information
        Returns: an unlabeled span

        """
        return Span(self.first, self.last)


@dataclass(frozen=True)
class BinaryRelation:
    """
    A binary relation

    NOTE: although the relation subject/object arguments should not have labels, they are used because of
    how we visualize the relations in the Brat UI
    """

    subject: LabeledSpan
    object: LabeledSpan
    label: str

    def normalize(self) -> BinaryRelation:
        """
        Normalize the relation by ignoring the subject/object arguments meaning,
        simply assigning the subject argument to be the first occurring argument in the sentence, and the object
        as the second occurring argument in the sentence
        Returns: a normalized binary relation

        """
        if self.subject.first > self.object.first:
            return BinaryRelation(self.object, self.subject, self.label)
        else:
            return self


@dataclass(frozen=True)
class Entity(LabeledSpan):
    """
    An entity
    """

    source: Optional[str] = None


@dataclass(frozen=True)
class Event(LabeledSpan):
    """
    A generic event, can a trigger and any number of arguments
    """

    args: List[LabeledSpan]
    trigger: Span

    def __post_init__(self):
        # sort the arguments by start offset
        sorted_args = sorted(self.args, key=lambda a: a.first)
        object.__setattr__(self, "args", sorted_args)


@dataclass(frozen=True)
class Edge:
    """
    An edge representation in a graph
    """

    source: Span
    target: Span
    label: str


@dataclass(frozen=True)
class Graph:
    roots: List[Span]
    edges: List[Edge]


@dataclass(frozen=True)
class SentenceSource:
    """
    A unique identifier of a sentence inside a data set
    """

    filename: str
    offset: int  # the sentence offset inside the files (as a single file can contain a list of sentences)


@dataclass(frozen=True)
class Sentence:
    """
    An annotated sentence in a data set
    """

    # The word tokens of a sentence
    words: List[str]
    # character level offsets for the start of each word, based on offset in the entire document
    document_start_char_offsets: List[int]
    # character level offsets for the end of each word, based on offset in the entire document
    document_end_char_offsets: List[int]
    # Annotations that provide a label to each token in the sentence (such as lemmas, and tags)
    labels: Dict[str, List[str]]
    # Annotations that provide spans of tokens (such as entities and chunks)
    spans: Dict[str, List[LabeledSpan]]
    # Annotations that have a graph representation
    graphs: Dict[str, Graph]
    # Annotations of binary relations
    relations: Dict[str, List[BinaryRelation]]
    # Annotation of generic events
    events: List[Event]
    # An optional unique identifier for the sentence inside a data set
    source: Optional[SentenceSource] = field(default=None, compare=False, hash=False)

    def with_relations(self, relations: Dict[str, List[BinaryRelation]]) -> Sentence:
        """
        A sentence with it's events replaced by the provided events

        Args:
            relations: the new events

        Returns: a nee sentence with the relations replaced

        """
        return replace(self, relations=relations)

    def with_events(self, events: List[Event]) -> Sentence:
        """
        A sentence with it's relations replaced by the provided relations

        Args:
            events: the new relations

        Returns: a nee sentence with the relations replaced

        """
        return replace(self, events=events)

    def default_graph(self) -> Graph:
        if "universal-enhanced" in self.graphs:
            return self.graphs["universal-enhanced"]
        elif "universal-basic" in self.graphs:
            return self.graphs["universal-basic"]
        else:
            raise AssertionError("No graph representation found")


@dataclass(frozen=True)
class CorefMention:
    sentence_idx: int  # The sentence index within the document
    head_idx: int  # the index of the mention head within the sentence
    first: int  # the index of the first mention token.
    last: int  # the index of the last mention token.
    chain_id: int  # the chain id.


DocumentSource = str


@dataclass(frozen=True)
class Document:
    """
    An annotated  document inside a data set
    """

    id: str = field(compare=False, hash=False)  # unique identifier in the data set this document belongs to
    text: str  # The full unprocessed text of the document
    sentences: List[Sentence]
    source: Optional[DocumentSource] = field(default=None, compare=False, hash=False)  # don't include in eq, hash .
    coref_chains: List[CorefMention] = field(default_factory=list)

    def get_coref_mention_for_entity(self, sent_idx, ent_first_idx, ent_last_idx) -> Optional[CorefMention]:
        """

        Args:
            sent_idx: the sentence index of the entity
            ent_first_idx: the entity first token index
            ent_last_idx: the entity last token index

        Returns: The coref mention which corresponds to the entity if one exists, or None otherwise.

        """
        return next(
            (
                cm
                for cm in self.coref_chains
                if cm.sentence_idx == sent_idx and ent_first_idx <= cm.head_idx <= ent_last_idx
            ),
            None,
        )

    def get_coref_chain_by_chain_id(self, chain_id: int) -> List[CorefMention]:
        """

        Args:
            chain_id:

        Returns: The coref chain identified by the given id

        """
        return [cm for cm in self.coref_chains if cm.chain_id == chain_id]

    def get_coref_chain_for_entity(self, sent_idx, ent_first_idx, ent_last_idx) -> List[CorefMention]:
        """

        Args:
            sent_idx: sentence index within the document
            ent_first_idx: index of the first entity token
            ent_last_idx: index of the last entity token

        Returns: The CorefMention of the chain the entity belongs to, or an empty list if no chain exists.

        """
        ent_coref_mention = self.get_coref_mention_for_entity(sent_idx, ent_first_idx, ent_last_idx)
        if ent_coref_mention:
            return self.get_coref_chain_by_chain_id(ent_coref_mention.chain_id)
        else:
            return []

    def get_representative_string_for_chain(self, chain_id: int, ent_type=None) -> Optional[str]:
        """ Returns the longest entity string among the named entities as the representative string for the chain

        Args:
            chain_id: an id of the input chain
            ent_type: if specified, we'll return the longest entity string among the named entities whose type matches
                      this given type.

        """
        chain_id_to_ent_srings = defaultdict(list)
        for cm in self.get_coref_chain_by_chain_id(chain_id):
            for ent in self.sentences[cm.sentence_idx].spans["entities"]:
                entc = cast(Entity, ent)
                # we ignore entities added by coref propagation during decoding.
                if ent.first <= cm.head_idx <= ent.last and entc.source != "coref":
                    if not ent_type or ent_type.lower() == ent.label.lower():
                        ent_string = " ".join(self.sentences[cm.sentence_idx].words[ent.first : ent.last + 1])
                        chain_id_to_ent_srings[chain_id].append(ent_string)

        if chain_id_to_ent_srings:
            return max([ent_string for ent_string in chain_id_to_ent_srings[chain_id]], key=len)
        else:
            return None


# ------------------------------------------------------------------------------------------- #
# Immutable value objects that represent the relations and entities extracted from a sentence #
# ------------------------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ExtractedSpan:
    """
    A labeled span extracted by some rule
    """

    span: LabeledSpan
    found_by: str  # name of rule that extracted this labeled span


@dataclass(frozen=True)
class ExtractedRelation:
    """
    A binary relation extracted by some rule
    """

    relation: BinaryRelation
    found_by: str  # name of rule that extracted this binary relation


@dataclass(frozen=True)
class ExtractedEvent:
    """
    An event (binary relation with a trigger) extracted by some rule
    """

    relation: BinaryRelation
    trigger: Span  # The span of tokens that identifies the trigger of the even
    found_by: str  # name of rule that extracted this event


@dataclass(frozen=True)
class SentenceExtractions:
    """
    All the extracted information for a single sentence
    """

    spans: List[ExtractedSpan]
    relations: List[ExtractedRelation]
    events: List[ExtractedEvent]

    @staticmethod
    def empty() -> SentenceExtractions:
        """
        Builder method for an empty extraction
        Returns: empty extraction

        """
        return SentenceExtractions([], [], [])

    def contains_any_relation(self) -> bool:
        """
        Are there any binary relations in this extraction

        Returns: 'True' if at least one binary or event extraction

        """
        return len(self.relations) + len(self.events) > 0

    def contains_relation(self, relation_name: str) -> bool:
        """
        Is the specified relation exists in wither the the extracted binary relations or event

        Args:
            relation_name: name of the relation

        Returns: 'True' if the provided relation exists, 'False' otherwise

        """
        return relation_name in (r.label for r in self.all_binary_relations())

    def all_binary_relations(self) -> Iterator[BinaryRelation]:
        """
        A generator for all the contained binary relations, those that were extracted directly or as events

        Returns: a generator of binary relations

        """
        return chain((r.relation for r in self.relations), (r.relation for r in self.events))


@dataclass(frozen=True)
class Extraction:
    """
    The extracted information for a sentence, along with the sentence and document
    """

    doc: Document
    sent: Sentence
    sent_extractions: SentenceExtractions

    @staticmethod
    def empty(doc: Document, sent: Sentence) -> Extraction:
        return Extraction(doc, sent, SentenceExtractions.empty())

    @classmethod
    def to_empty_extractions(cls, it: Iterator[Tuple[Document, Sentence]]) -> Iterator[Extraction]:
        return map(lambda p: cls.empty(*p), it)
