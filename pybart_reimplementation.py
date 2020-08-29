import json
import pickle
from collections import defaultdict
from functools import reduce
from typing import List, Dict
import conllu
import networkx as nx
import spacy_udpipe
from pybart.conllu_wrapper import parse_conllu
from spacy_conll import ConllFormatter
from tqdm import tqdm
from os import path

from pybart_parts.gen_pattern import PatternGenerator
from pybart_parts.pattern_selectors import WordNodeSelector, LemmaNodeSelector, TriggerVarNodeSelector, \
    LabelEdgeSelector
from pybart_parts.sample_types import AnnotatedSample
from pybart_parts.utils import GenerationFromAnnotatedSamples


# region Prepare Data
def download_english_tacred_data():
    print("Download the TACRED dataset from: https://catalog.ldc.upenn.edu/LDC2018T24 and place the 'data' folder"
          "in in dataset/tacred/")


def generate_english_tacred_data():
    tacred_files = ["dataset/tacred/data/json/train.json",
                    "dataset/tacred/data/json/dev.json",
                    "dataset/tacred/data/json/test.json"
                    ]
    for file_path in tacred_files:
        if not path.exists(file_path):
            print(f'{file_path} cannot be found. Please download TACRED and follow step 1.')

    for split in ["train", "dev", "test"]:
        SampleBARTAnnotator.parse_tacred_json_to_ud(f'dataset/tacred/data/json/{split}.json',
                                                    f'data/annotated_tacred/en/{split}.conllu',
                                                    "en")
    for split in ["train", "dev", "test"]:
        main_generate(f'data/annotated_tacred/en/{split}.conllu',
                      "data/pattern_dicts/en/",
                      "en",
                      skip_no_relation=split == "train")


# endregion

# region Constants

spike_relations = ['org:alternate_names', 'org:city_of_headquarters', 'org:country_of_headquarters', 'org:dissolved',
                   'org:founded', 'org:founded_by', 'org:member_of', 'org:members', 'org:number_of_employees_members',
                   'org:parents', 'org:political_religious_affiliation', 'org:shareholders',
                   'org:stateorprovince_of_headquarters', 'org:subsidiaries', 'org:top_members_employees',
                   'org:website', 'per:age', 'per:alternate_names', 'per:cause_of_death', 'per:charges', 'per:children',
                   'per:cities_of_residence', 'per:city_of_birth', 'per:city_of_death', 'per:countries_of_residence',
                   'per:country_of_birth', 'per:country_of_death', 'per:date_of_birth', 'per:date_of_death',
                   'per:employee_of', 'per:origin', 'per:other_family', 'per:parents', 'per:religion',
                   'per:schools_attended', 'per:siblings', 'per:spouse', 'per:stateorprovince_of_birth',
                   'per:stateorprovince_of_death', 'per:stateorprovinces_of_residence', 'per:title']
relation_type_to_trigger_type = {
    "org:alternate_names": "org_alternate_names_dict",
    "org:city_of_headquarters": "org_headquarters_dict",
    "org:country_of_headquarters": "org_headquarters_dict",
    "org:dissolved": "org_date_dissolved_dict",
    "org:founded": "org_founded_dict",
    "org:founded_by": "org_founded_dict",
    "org:member_of": "org_members_dict",
    "org:members": "org_members_dict",
    "org:number_of_employees/members": "org_number_of_employees_members",
    "org:parents": "org_parents_dict",
    "org:political/religious_affiliation": None,
    "org:shareholders": "org_number_of_employees_members",
    "org:stateorprovince_of_headquarters": "org_headquarters_dict",
    "org:subsidiaries": None,
    "org:top_members/employees": "org_top_members_employees_dict",
    "org:website": None,
    "per:age": "per_age_dict",
    "per:alternate_names": "per_alternate_names",
    "per:cause_of_death": "per_death_dict",
    "per:charges": "per_charges_dict",
    "per:children": "per_children_dict",
    "per:cities_of_residence": "per_residence",
    "per:city_of_birth": "per_birth_dict",
    "per:city_of_death": "per_death_dict",
    "per:countries_of_residence": "per_residence",
    "per:country_of_birth": "per_birth_dict",
    "per:country_of_death": "per_death_dict",
    "per:date_of_birth": "per_birth_dict",
    "per:date_of_death": "per_death_dict",
    "per:employee_of": "per_employee_or_member_of_dict",
    "per:origin": None,
    "per:other_family": "per_other_family_dict",
    "per:parents": "per_parents_dict",
    "per:religion": None,
    "per:schools_attended": "per_schools_attended_dict",
    "per:siblings": "per_siblings_dict",
    "per:spouse": "per_spouse_dict",
    "per:stateorprovince_of_birth": "per_birth_dict",
    "per:stateorprovince_of_death": "per_death_dict",
    "per:stateorprovinces_of_residence": "per_residence",
    "per:title": None,
}


# endregion

# region Utils

def change_pattern_dict_pivot_to_example(pattern_dict: Dict) -> Dict:
    # change the dev_pattern_dict pivot from "pattern" to "example"
    _pattern_dict_by_example = dict()
    for relation in pattern_dict.keys():
        _pattern_dict_by_example[relation] = dict()
        for pattern in pattern_dict[relation].keys():
            for example in pattern_dict[relation][pattern]:
                example_key = example.raw
                if example_key not in _pattern_dict_by_example[relation]:
                    _pattern_dict_by_example[relation][example_key] = set()
                _pattern_dict_by_example[relation][example_key].add(pattern)

    return _pattern_dict_by_example


def change_pattern_dict_pivot_to_pattern(pattern_dict: Dict) -> Dict:
    _pattern_dict = dict()
    for relation in pattern_dict.keys():
        for pattern in pattern_dict[relation].keys():
            if pattern not in _pattern_dict:
                _pattern_dict[pattern] = dict()
            if relation not in _pattern_dict[pattern]:
                _pattern_dict[pattern][relation] = 0
            _pattern_dict[pattern][relation] = _pattern_dict[pattern][relation] + 1

    multi_relation_patterns = len(list(filter(lambda p: len(list(_pattern_dict[p])) > 1, _pattern_dict)))
    print(
        f'{multi_relation_patterns} patterns that match multiple patterns and being filtered out of {len(_pattern_dict)} patterns.')

    return _pattern_dict


def get_patterns_in_text_format(patterns_pkl_file_path: str, pattern_output_file: str):
    with open(
            f'{patterns_pkl_file_path}', "rb") as f:
        pkl_pattern_dict = pickle.load(f)

    text_pattern_dict = dict()
    for relation in pkl_pattern_dict:
        if relation not in text_pattern_dict:
            text_pattern_dict[relation] = dict()
        for pattern in pkl_pattern_dict[relation]:
            if pattern not in text_pattern_dict[relation]:
                text_pattern_dict[relation][pattern] = list()
            for sample in pkl_pattern_dict[relation][pattern]:
                text_pattern_dict[relation][pattern].append(sample.raw)

    with open(f'{pattern_output_file}', "w", encoding="utf-8") as f:
        f.write(json.dumps(text_pattern_dict, indent=4, ensure_ascii=False))


def get_trigger_words_stats(patterns_pkl_file_path: str):
    stats = dict()
    with open(f'{patterns_pkl_file_path}', "rb") as f:
        pattern_dict = pickle.load(f)

    for relation in pattern_dict:
        if relation not in stats:
            stats[relation] = {
                "with_trigger_word_count": 0,
                "without_trigger_word_count": 0,
            }
        for pattern in pattern_dict[relation]:
            for example in pattern_dict[relation][pattern]:
                stats[relation] = {
                    "with_trigger_word_count": stats[relation]["with_trigger_word_count"] + 1,
                    "without_trigger_word_count": stats[relation]["without_trigger_word_count"],
                } if example.trigger_tok_offsets is None else {
                    "with_trigger_word_count": stats[relation]["with_trigger_word_count"],
                    "without_trigger_word_count": stats[relation]["without_trigger_word_count"] + 1,
                }
    print(json.dumps(stats, indent=4))


# endregion

# region Annotation

def g_triggers(lang: str): return f'dataset/trigger_dict_{lang}/'


def get_entities(example_conllu: conllu.TokenList):
    entities = ["O"] * len(example_conllu)

    # The start and end starts from 1 and are inclusive so we substract 1
    for i in range(int(example_conllu.metadata['subj_start']) - 1, int(example_conllu.metadata['subj_end'])):
        entities[i] = example_conllu.metadata['subj_type']

    for i in range(int(example_conllu.metadata['obj_start']) - 1, int(example_conllu.metadata['obj_end'])):
        entities[i] = example_conllu.metadata['obj_type']

    return entities


def get_triggers(rel: str, lang: str):
    if rel not in relation_type_to_trigger_type:
        return []
    rel = relation_type_to_trigger_type[rel]

    if rel is None:
        return []
    try:
        # NOTE: not sure its the best choice of encoding, but it worked for me
        encoding = "windows-1252" if lang == "en" else "utf-8"
        with open(g_triggers(lang) + rel + ".xml", "r", encoding=encoding) as f:
            triggers = [l.strip() for l in f.readlines() if l.strip() != '']
        # This is done cause some trigger words list have duplicate values
        return list(set(triggers))
    except FileNotFoundError:
        return []


# looks for triggers in the sentence (and not as part of the two entities of the relation - which doesnt make sense)
def search_triggers(subj_start, subj_end, obj_start, obj_end, rel, tokens, lang: str):
    # The subject/obj range starts from 1 per the conllu format and are inclusive
    subj_start = subj_start - 1
    subj_end = subj_end - 1
    obj_start = obj_start - 1
    obj_end = obj_end - 1
    trigger_toks = []
    for trigger in get_triggers(rel, lang):
        # trigger_end and trigger_start are inclusive range
        for trigger_start, token in enumerate(tokens):
            trigger_end = trigger_start + len(trigger.split()) - 1
            if (trigger.split() == tokens[trigger_start: trigger_end + 1]) and \
                    (trigger_end < subj_start or trigger_start > subj_end) and \
                    (trigger_end < obj_start or trigger_start > obj_end):
                # adding +1 to match the conllu format
                trigger_toks.append((trigger_start + 1, trigger_end + 1))
    return trigger_toks if trigger_toks else []


class SampleBARTAnnotator(object):
    @staticmethod
    def get_ud_parser(lang: str):
        if lang == "en":
            nlp = spacy_udpipe.load_from_path(lang="en",
                                              path="C:/Users/t-ofarvi/Desktop/HUJI/relation_extraction/udpipe_models/english-ewt-ud-2.5-191206.udpipe",
                                              meta={"description": "Custom 'en' model"})
        elif lang == "ru":
            nlp = spacy_udpipe.load_from_path(lang="ru",
                                              path="C:/Users/t-ofarvi/Desktop/HUJI/relation_extraction/udpipe_models/russian-syntagrus-ud-2.5-191206.udpipe",
                                              meta={"description": "Custom 'ru' model"})
        elif lang == "ko":
            nlp = spacy_udpipe.load_from_path(lang="ko",
                                              path="C:/Users/t-ofarvi/Desktop/HUJI/relation_extraction/udpipe_models/korean-gsd-ud-2.5-191206.udpipe",
                                              meta={"description": "Custom 'ko' model"})
        else:
            raise ValueError

        conll_formatter = ConllFormatter(nlp)
        nlp.add_pipe(conll_formatter)

        return nlp

    @staticmethod
    def parse_tokenized_txt_to_ud(txt_file_path: str, output_path: str, lang: str):
        nlp = SampleBARTAnnotator.get_ud_parser(lang)

        conllu_parse_list = []
        with open(txt_file_path, 'r', encoding='utf-8') as input_file:
            for line in tqdm(input_file):
                doc = nlp([line])
                conllu_token_list = conllu.parse(doc._.conll_str)

                assert len(conllu_token_list) == 1
                conllu_token_list = conllu_token_list[0]

                relation = "Please Fill"
                tokens = [node["form"] for node in conllu_token_list]

                conllu_token_list.metadata["id"] = "Please Fill"
                conllu_token_list.metadata["docid"] = "Please Fill"
                conllu_token_list.metadata["relation"] = relation
                conllu_token_list.metadata["token"] = json.dumps(tokens, ensure_ascii=False)
                # Adding +1 because this count start for 0, but the conllu token id is starting from 1
                conllu_token_list.metadata["subj_start"] = "Please Fill"
                conllu_token_list.metadata["subj_end"] = "Please Fill (inclusive)"
                conllu_token_list.metadata["obj_start"] = "Please Fill"
                conllu_token_list.metadata["obj_end"] = "Please Fill (inclusive)"
                conllu_token_list.metadata["subj_type"] = "Please Fill"
                conllu_token_list.metadata["obj_type"] = "Please Fill"

                conllu_token_list.metadata["trigger_tokens"] = "To be added later"

                conllu_parse_list.append(conllu_token_list)

        with open(output_path, 'w', encoding='utf-8') as output_file:
            for conllu_token_list in conllu_parse_list:
                output_file.write(conllu_token_list.serialize())

    @staticmethod
    def add_info_to_parsed_ud(conllu_file_path: str, tacred_json_file_path: str, lang: str, output_path: str):

        with open(tacred_json_file_path, 'r', encoding='utf-8') as input_file:
            tacred_json = json.load(input_file)
            tacred_json_dict = {example["id"]: example for example in tacred_json}

        conllu_parse_list = []
        with open(conllu_file_path, 'r', encoding='utf-8') as conllu_file:
            for example_conllu in tqdm(conllu.parse_incr(conllu_file)):
                id = example_conllu.metadata["id"]
                example_json = tacred_json_dict[id]

                relation = example_json['relation']
                tokens = [node["form"] for node in example_conllu]

                example_conllu.metadata["id"] = example_json['id']
                example_conllu.metadata["docid"] = example_json['docid']
                example_conllu.metadata["relation"] = relation
                if json.loads(example_conllu.metadata["token"]) != tokens:
                    print(f'Sentence {id} conllu token are different from the json tokens')
                assert example_conllu.metadata["subj_start"]
                assert example_conllu.metadata["subj_end"]
                assert example_conllu.metadata["obj_start"]
                assert example_conllu.metadata["obj_end"]
                example_conllu.metadata["subj_type"] = example_json['subj_type']
                example_conllu.metadata["obj_type"] = example_json['obj_type']

                trigger_tokens = search_triggers(example_json['subj_start'], example_json['subj_end'],
                                                 example_json['obj_start'], example_json['obj_end'],
                                                 relation, tokens, lang)

                # Sorting for ease of read
                trigger_tokens_sorted = sorted(trigger_tokens, key=lambda x: x[0])
                example_conllu.metadata["trigger_tokens"] = json.dumps(trigger_tokens_sorted)

                conllu_parse_list.append(example_conllu)

        with open(output_path, 'w', encoding='utf-8') as output_file:
            for conllu_token_list in conllu_parse_list:
                output_file.write(conllu_token_list.serialize())

    @staticmethod
    def parse_tacred_json_to_ud(tacred_json_file_path: str, output_path: str, lang: str):
        nlp = SampleBARTAnnotator.get_ud_parser(lang)

        conllu_parse_list = []
        with open(tacred_json_file_path, 'r', encoding='utf-8') as input_file:
            tacred_json = json.load(input_file)
            for example_json in tqdm(tacred_json):
                doc = nlp([example_json["token"]])
                conllu_token_list = conllu.parse(doc._.conll_str)

                assert len(conllu_token_list) == 1
                conllu_token_list = conllu_token_list[0]

                relation = example_json['relation']
                tokens = [node["form"] for node in conllu_token_list]

                conllu_token_list.metadata["id"] = example_json['id']
                conllu_token_list.metadata["docid"] = example_json['docid']
                conllu_token_list.metadata["relation"] = relation
                conllu_token_list.metadata["token"] = json.dumps(tokens)
                # Adding +1 because this count start for 0, but the conllu token id is starting from 1
                conllu_token_list.metadata["subj_start"] = json.dumps(example_json['subj_start'] + 1)
                conllu_token_list.metadata["subj_end"] = json.dumps(example_json['subj_end'] + 1)
                conllu_token_list.metadata["obj_start"] = json.dumps(example_json['obj_start'] + 1)
                conllu_token_list.metadata["obj_end"] = json.dumps(example_json['obj_end'] + 1)
                conllu_token_list.metadata["subj_type"] = example_json['subj_type']
                conllu_token_list.metadata["obj_type"] = example_json['obj_type']

                trigger_tokens = search_triggers(example_json['subj_start'], example_json['subj_end'],
                                                 example_json['obj_start'], example_json['obj_end'],
                                                 relation, tokens, lang)

                # Sorting for ease of read
                trigger_tokens_sorted = sorted(trigger_tokens, key=lambda x: x[0])
                conllu_token_list.metadata["trigger_tokens"] = json.dumps(trigger_tokens_sorted)

                conllu_parse_list.append(conllu_token_list)

        with open(output_path, 'w', encoding='utf-8') as output_file:
            for conllu_token_list in conllu_parse_list:
                output_file.write(conllu_token_list.serialize())

    @staticmethod
    def annotate_sample(example_conllu: conllu.TokenList, lang: str, use_triggers: bool = True) -> List[
        AnnotatedSample]:
        sent, metadata = parse_conllu(example_conllu.serialize())

        assert len(sent) == len(metadata) == 1
        sent = sent[0]
        _ = sent.pop(0)  # for internal use, we remove the stub root-node
        metadata = metadata[0]

        tokens = [node.get_conllu_field("form") for node in sent.values()]
        tags = [node.get_conllu_field("xpos") for node in sent.values()]
        lemmas = [node.get_conllu_field("lemma") for node in sent.values()]
        entities = get_entities(example_conllu)
        chunks = ["O"] * len(tokens)  # chunks - not interesting

        # create a networkX graph from the returned graph. one multiDi and one not - for later use.
        g = nx.Graph()
        mdg = nx.MultiDiGraph()
        for node in sent.values():
            for parent, label in node.get_new_relations():
                if parent.get_conllu_field("id") == 0:
                    continue

                # TODO: Why (-1)
                g.add_node(parent.get_conllu_field("id") - 1, label=parent.get_conllu_field("form"))
                g.add_node(node.get_conllu_field("id") - 1, label=node.get_conllu_field("form"))
                g.add_edge(parent.get_conllu_field("id") - 1, node.get_conllu_field("id") - 1, label=label)
                mdg.add_node(parent.get_conllu_field("id") - 1, label=parent.get_conllu_field("form"))
                mdg.add_node(node.get_conllu_field("id") - 1, label=node.get_conllu_field("form"))
                mdg.add_edge(parent.get_conllu_field("id") - 1, node.get_conllu_field("id") - 1, label=label)

        # add an annotated sample to the list for each trigger on the path
        rel = example_conllu.metadata['relation']
        ann_samples = []
        trigger_toks = search_triggers(int(example_conllu.metadata['subj_start']),
                                       int(example_conllu.metadata['subj_end']),
                                       int(example_conllu.metadata['obj_start']),
                                       int(example_conllu.metadata['obj_end']),
                                       rel, tokens, lang) if use_triggers else []

        # The AnnotatedSample class expect exclusive range that start from 0
        trigger_tokens_fixed = []
        for trigger_range in trigger_toks:
            (start, end) = trigger_range
            trigger_tokens_fixed.append((start - 1, end))

        if len(trigger_tokens_fixed) == 0:
            trigger_tokens_fixed = [None]

        for trigger_tok in trigger_tokens_fixed:
            ann_samples.append(
                AnnotatedSample(
                    " ".join(tokens),
                    " ".join(tokens),
                    rel,
                    example_conllu.metadata['subj_type'].title(),
                    example_conllu.metadata['obj_type'].title(),
                    tokens, tags, entities, chunks, lemmas,
                    # add -1 to start as the AnnotatedSample expect an exclusive range that start from 0
                    # and the conllu on start from 1 and is inclusive
                    (int(example_conllu.metadata['subj_start']) - 1, int(example_conllu.metadata['subj_end'])),
                    (int(example_conllu.metadata['obj_start']) - 1, int(example_conllu.metadata['obj_end'])),
                    trigger_tok,
                    g, mdg))

        assert len(ann_samples) != 0

        return ann_samples


def main_annotate(lang: str, tacred_json_file_path: str, ud_output_path: str, ):
    raise NotImplemented


# endregion

# region Generation

def generate_patterns(conllu_example_list: List[conllu.TokenList], use_triggers: bool, lang: str):
    # first annotate all samples
    ann_samples = defaultdict(list)
    print(f'Annotating examples...')
    for example_conllu in tqdm(conllu_example_list):
        # annotate
        new_ann_samples = SampleBARTAnnotator.annotate_sample(example_conllu, use_triggers=use_triggers, lang=lang)

        # store the annotated samples per their relation type
        _ = [ann_samples[ann_sample.relation].append(ann_sample) for ann_sample in new_ann_samples]
    print(f'Finished annotating examples')

    pattern_dict_no_lemma = dict()
    pattern_dict_with_lemma = dict()
    pattern_dict_text_agnostic = dict()
    # per generation option
    for node_selector, pattern_dict in [(WordNodeSelector, pattern_dict_no_lemma),
                                        (LemmaNodeSelector, pattern_dict_with_lemma),
                                        (None, pattern_dict_text_agnostic)]:
        errs = dict()
        # per annotated sample
        print(f'Generating patterns for {node_selector}')
        for rel, ann_samples_per_rel in tqdm(ann_samples.items()):
            # triggers = get_triggers(rel, lang)
            triggers = [relation_type_to_trigger_type[rel]] if rel != 'no_relation' else []
            # instantiate generators and then generate patters
            pattern_generator_with_trigger = PatternGenerator([TriggerVarNodeSelector(triggers)],
                                                              LabelEdgeSelector(),
                                                              [])
            pattern_generator_no_trigger = PatternGenerator([], LabelEdgeSelector(),
                                                            [node_selector()] if node_selector else [])
            pattern_dict_pre_filter, error_list = GenerationFromAnnotatedSamples \
                .gen_pattern_dict(ann_samples_per_rel, pattern_generator_with_trigger, pattern_generator_no_trigger)

            # TODO: Don't think I need this filter
            # filter  misparsed patterns
            pattern_dict[rel] = pattern_dict_pre_filter  # filter_misparsed_patterns(pattern_dict_pre_filter, rel, errs)

            errs[rel] = error_list

            error_percentage = len(error_list) / float(len(ann_samples_per_rel))
            if error_percentage > 0.1:
                print(
                    f'Relation {rel}: {error_percentage}% errors ({len(error_list)} out of {len(ann_samples_per_rel)}')

        print(f'Finished Generating patterns for {node_selector}')
        # print("%d/%d patterns can’t be created for %s" % (
        #   total_d, sum([len(a) for a in ann_samples.values()]), str(node_selector)))

        global_error_count = sum(len(value) for value in errs.values())
        global_example_count = reduce(lambda prev, curr: prev + len(curr), ann_samples.values(), 0)
        error_percentage = global_error_count / global_example_count
        print(f'{error_percentage}% errors ({global_error_count} out of {global_example_count}')

    return pattern_dict_no_lemma, pattern_dict_with_lemma, pattern_dict_text_agnostic


def main_generate(tacred_conllu_path: str, pattern_output_dir: str,
                  lang: str, skip_no_relation=False, use_triggers=True):
    # load data
    print(f'Loading  data...')
    with open(tacred_conllu_path, 'r', encoding='utf-8') as f:
        tacred_conllu = conllu.parse(f.read())

    if skip_no_relation:
        tacred_conllu = list(filter(lambda x: x.metadata['relation'] != "no_relation", tacred_conllu))

    print(f'Finished loading data')

    # generate patterns

    print(f'Starting generating patterns')
    pattern_dict_no_lemma, pattern_dict_with_lemma, pattern_dict_text_agnostic = \
        generate_patterns(tacred_conllu, use_triggers, lang)

    # dump the generated patterns
    print(f'Finished generating patterns')

    filename = (tacred_conllu_path.split("/")[-1]).split(".")[0]

    print(f'Writing patterns...')
    with open(f'{pattern_output_dir}/{filename}_pattern_dict_no_lemma{"" if use_triggers else "_no_trig"}.pkl',
              "wb") as f:
        pickle.dump(pattern_dict_no_lemma, f)
    with open(f'{pattern_output_dir}/{filename}_pattern_dict_with_lemma{"" if use_triggers else "_no_trig"}.pkl',
              "wb") as f:
        pickle.dump(pattern_dict_with_lemma, f)
    with open(f'{pattern_output_dir}/{filename}_pattern_dict_text_agnostic{"" if use_triggers else "_no_trig"}.pkl',
              "wb") as f:
        pickle.dump(pattern_dict_text_agnostic, f)
    print(f'Finished patterns')


# endregion

# region Evaluation

def get_patterns_stat(train_pattern_dict: Dict, eval_pattern_dict: Dict):
    eval_pattern_dict_example_pivot = change_pattern_dict_pivot_to_example(eval_pattern_dict)

    train_pattern_dict_pattern_pivot = change_pattern_dict_pivot_to_pattern(train_pattern_dict)
    eval_pattern_dict_pattern_pivot = change_pattern_dict_pivot_to_pattern(eval_pattern_dict)

    pattern_stat_dict = dict()
    for train_pattern in train_pattern_dict_pattern_pivot.keys():
        retrieved_and_relevant_count = 0
        retrieved_and_not_relevant_count = 0

        train_pattern_rel_list = train_pattern_dict_pattern_pivot[train_pattern]
        if len(train_pattern_rel_list) > 1:
            continue
        if len(train_pattern_rel_list) == 0:
            continue
        train_pattern_rel = list(train_pattern_rel_list.keys())[0]
        if train_pattern in eval_pattern_dict_pattern_pivot:
            dev_pattern_rel_list = eval_pattern_dict_pattern_pivot[train_pattern]
            for dev_pattern_rel, dev_pattern_occurrence_count in dev_pattern_rel_list.items():
                if dev_pattern_rel == train_pattern_rel:
                    retrieved_and_relevant_count = retrieved_and_relevant_count + dev_pattern_occurrence_count
                else:
                    retrieved_and_not_relevant_count = retrieved_and_not_relevant_count + dev_pattern_occurrence_count

        relevant_count = len(eval_pattern_dict_example_pivot[train_pattern_rel])
        recall = retrieved_and_relevant_count / float(relevant_count)
        precision = retrieved_and_relevant_count / float(
            retrieved_and_relevant_count + retrieved_and_not_relevant_count) if float(
            retrieved_and_relevant_count + retrieved_and_not_relevant_count) > 0 else -1
        f1 = 2 * recall * precision / float(recall + precision) if (recall + precision) > 0 else -1
        pattern_stat_dict[train_pattern] = {
            "f1": f1,
            "recall": recall,
            "precision": precision,
            "relevant_count": relevant_count,
            "retrieved_and_relevant_count": retrieved_and_relevant_count,
            "retrieved_and_not_relevant_count": retrieved_and_not_relevant_count,
        }

    return pattern_stat_dict


def get_f_scores(train_pattern_dict: Dict, eval_pattern_dict: Dict):
    eval_pattern_dict = change_pattern_dict_pivot_to_example(eval_pattern_dict)

    global_relevant_count = 0
    global_retrieved_and_relevant_count = 0
    global_retrieved_and_not_relevant_count = 0

    scores = {}
    for relation in train_pattern_dict.keys():
        relevant_count = 0
        retrieved_and_relevant_count = 0
        retrieved_and_not_relevant_count = 0

        train_patterns = train_pattern_dict[relation].keys()

        for dev_relation in eval_pattern_dict.keys():
            for dev_example in eval_pattern_dict[dev_relation].keys():
                if dev_relation == relation:
                    relevant_count = relevant_count + 1

                dev_patterns = eval_pattern_dict[dev_relation][dev_example]
                if len(set(dev_patterns).intersection(set(train_patterns))) > 0:
                    if dev_relation == relation:
                        retrieved_and_relevant_count = retrieved_and_relevant_count + 1
                    else:
                        retrieved_and_not_relevant_count = retrieved_and_not_relevant_count + 1

        global_relevant_count = global_relevant_count + relevant_count
        global_retrieved_and_relevant_count = global_retrieved_and_relevant_count + retrieved_and_relevant_count
        global_retrieved_and_not_relevant_count = global_retrieved_and_not_relevant_count + retrieved_and_not_relevant_count

        recall = retrieved_and_relevant_count / float(relevant_count) if relevant_count != 0 else -1
        precision = retrieved_and_relevant_count / float(
            retrieved_and_relevant_count + retrieved_and_not_relevant_count) if float(
            retrieved_and_relevant_count + retrieved_and_not_relevant_count) > 0 else -1
        f1 = 2 * recall * precision / float(recall + precision) if (recall + precision) > 0 else -1

        scores[relation] = {
            "f1": f1,
            "recall": recall,
            "precision": precision,
            "relevant_count": relevant_count,
            "retrieved_and_relevant_count": retrieved_and_relevant_count,
            "retrieved_and_not_relevant_count": retrieved_and_not_relevant_count,
        }

    global_recall = global_retrieved_and_relevant_count / float(global_relevant_count)
    global_precision = global_retrieved_and_relevant_count / float(
        global_retrieved_and_relevant_count + global_retrieved_and_not_relevant_count) if float(
        global_retrieved_and_relevant_count + global_retrieved_and_not_relevant_count) > 0 else -1
    global_f1 = 2 * global_recall * global_precision / float(global_recall + global_precision)

    scores["All"] = {
        "f1": global_f1,
        "recall": global_recall,
        "precision": global_precision,
        "relevant_count": global_relevant_count,
        "retrieved_and_relevant_count": global_retrieved_and_relevant_count,
        "retrieved_and_not_relevant_count": global_retrieved_and_not_relevant_count,
    }

    return scores


def main_eval(train_pattern_path: str, dev_pattern_path: str, test_pattern_path: str):
    with open(
            f'{train_pattern_path}', "rb") as f:
        train_pattern_dict = pickle.load(f)
    with open(
            f'{dev_pattern_path}', "rb") as f:
        dev_pattern_dict = pickle.load(f)
    with open(
            f'{test_pattern_path}', "rb") as f:
        test_pattern_dict = pickle.load(f)

    if "no_relation" in train_pattern_dict:
        del train_pattern_dict["no_relation"]

    train_pattern_stat_dict = get_patterns_stat(train_pattern_dict, dev_pattern_dict)
    train_pattern_stat_dict_positive_f1 = {k: v for k, v in train_pattern_stat_dict.items() if v['f1'] > 0}
    print(
        f'train_pattern_stat_dict_positive_f1: #{len(train_pattern_stat_dict_positive_f1)} / #{len(train_pattern_stat_dict)}')

    filtered_train_pattern_dict = dict()
    for relation in train_pattern_dict:
        filtered_train_pattern_dict[relation] = {k: v for k, v in train_pattern_dict[relation].items() if
                                                 k in train_pattern_stat_dict_positive_f1}

    dev_f_score_all_patterns = get_f_scores(train_pattern_dict, dev_pattern_dict)
    print(f'dev_f_score_all_patterns:')
    print(dev_f_score_all_patterns['All'])
    dev_f_score_filtered_patterns = get_f_scores(filtered_train_pattern_dict, dev_pattern_dict)
    print(f'dev_f_score_filtered_patterns:')
    print(dev_f_score_filtered_patterns['All'])

    test_f_score_all_patterns = get_f_scores(train_pattern_dict, test_pattern_dict)
    print(f'test_f_score_all_patterns:')
    print(test_f_score_all_patterns['All'])
    test_f_score_filtered_patterns = get_f_scores(filtered_train_pattern_dict, test_pattern_dict)
    print(f'test_f_score_filtered_patterns:')
    print(test_f_score_filtered_patterns['All'])

    print("done")


# endregion


# IMPORTANT NOTE! The scores are not over all of the eval set, about %18 of the test examples are not used in
# the evaluation as their subject, object or trigger do not share a head.

get_trigger_words_stats("data/pattern_dicts/en/dev_pattern_dict_text_agnostic.pkl")
exit()

# region Evaluate example
main_eval("data/pattern_dicts/en/train_pattern_dict_text_agnostic.pkl",
          "data/pattern_dicts/en/dev_pattern_dict_text_agnostic.pkl",
          "data/pattern_dicts/ko/sample_pattern_dict_text_agnostic.pkl")

# endregion

# region English data pipeline


main_eval("data/pattern_dicts/en/old/pattern_dict_with_lemma_train.pkl",
          "data/pattern_dicts/en/old/pattern_dict_with_lemma_dev.pkl",
          "data/pattern_dicts/en/old/pattern_dict_with_lemma_test.pkl")

SampleBARTAnnotator.parse_tacred_json_to_ud("dataset/tacred/data/json/test.json",
                                            "data/annotated_tacred/en/test.conllu",
                                            "en")
main_generate("data/annotated_tacred/en/dev.conllu", "data/pattern_dicts/en/", "en", skip_no_relation=False)

main_eval("data/pattern_dicts/en/old/pattern_dict_with_lemma_train.pkl",
          "data/pattern_dicts/en/old/pattern_dict_with_lemma_dev.pkl",
          "data/pattern_dicts/en/old/pattern_dict_with_lemma_test.pkl")

# endregion


# region Translation data pipeline
SampleBARTAnnotator.parse_tokenized_txt_to_ud(
    "data/annotated_tacred_tanslation/russian_tokenized_with_extra_info.conllu",
    "data/annotated_tacred_ru_sample/sample.conllu",
    "ru")

SampleBARTAnnotator.add_info_to_parsed_ud("data/annotated_tacred_tanslation/korean_tokenized_with_extra_info.conllu",
                                          "dataset/tacred/data/json/train.json",
                                          "ko",
                                          "data/annotated_tacred/ko/sample.conllu")

main_generate("data/annotated_tacred/ko/sample.conllu", "data/pattern_dicts/ko", "ko")

# endregion


# region Utils
get_patterns_in_text_format("data/pattern_dicts/en/test_pattern_dict_with_lemma.pkl",
                            "data/pattern_dicts/en/test_pattern_dict_with_lemma.txt")

# endregion Example
