import uuid
from .graph_token import Token, add_basic_edges


def parse_conllu(text):
    """Purpose: parses the given CoNLL-U formatted text.

    Args:
        (str) The text.

    returns:
        (list(dict(Token))) returns a list of sentence dicts.
            a sentence dict is a mapping from id to token/word.
        (list(list(str))) returns a list of comments list per sentence.

     Raises:
         ValueError: text must be a basic CoNLL-U, received an enhanced one.
         ValueError: text must be a basic CoNLL-U format, received a CoNLL-X format.
]    """
    sentences = []
    all_comments = []

    # for each sentence
    for sent in text.strip().split('\n\n'):
        lines = sent.strip().split('\n')
        if not lines:
            continue
        comments = []
        sentence = dict()

        # for each line (either comment or token)
        for line in lines:
            # store comments
            if line.startswith('#'):
                comments.append(line)
                continue

            # split line by any whitespace, and store the first 10 columns.
            parts = line.split()
            if len(parts) > 10:
                parts = line.split("\t")
                if len(parts) > 10:
                    raise ValueError("text must be a basic CoNLL-U format, received too many columns or separators.")

            new_id, form, lemma, upos, xpos, feats, head, deprel, deps, misc = parts[:10]

            # validate input
            if '-' in new_id:
                raise ValueError("text must be a basic CoNLL-U format, received a CoNLL-X format.")
            if deps != '_' or '.' in new_id:
                raise ValueError("text must be a basic CoNLL-U, received an enhanced one.")

            # fix xpos if empty to a copy of upos
            xpos = upos if xpos == '_' else xpos

            # add current token to current sentence
            sentence[int(new_id)] = Token(
                int(new_id), form, lemma, upos, xpos, feats, int(head), deprel, deps, misc)

        # add root
        sentence[0] = Token(0, None, None, None, None, None, None, None, None, None)

        # after parsing entire sentence, add basic deprel edges,
        # and add sentence to output list
        add_basic_edges(sentence)
        sentences.append(sentence)
        all_comments.append(comments)

    return sentences, all_comments


def serialize_conllu(converted, all_comments, preserve_comments=False):
    """Purpose: create a CoNLL-U formatted text from a sentence list.

    Args:
        (list(dict(Token))) The sentence list.

    returns:
        (str) the text corresponding to the sentence list in the CoNLL-U format.
     """
    text = []
    comments = []
    for (sentence, per_sent_comments) in zip(converted, all_comments):
        # recover comments from original file
        if preserve_comments:
            comments = ["\n".join(per_sent_comments)]

        # TODO - fix case of more than 9 copy nodes - needs special ordering e.g 1.1 ... 1.9 1.10 and not 1.1 1.10 ... 1.9
        text.append(
            comments + [token.get_conllu_string() for (cur_id, token) in sorted(sentence.items()) if cur_id != 0])

    return "\n".join(["\n".join(sent) + "\n" for sent in text])


# fw.conllu_to_odin(converter.convert(fw.parse_conllu(fw.odin_to_conllu(json_buf)[0])))
# or better off: fw.conllu_to_odin(converter.convert(fw.parse_odin(json_buf))))
def parse_odin(odin_json):
    sentences = []
    for sent in odin_json['sentences']:
        sentence = {0: Token(0, None, None, None, None, None, None, None, None, None)}
        for i, (word, tag, lemma) in enumerate(zip(sent['words'], sent['tags'], sent['lemmas'])):
            sentence[i + 1] = Token(i + 1, word, lemma, "_", tag, "_", "_", "_", "_", "_")
        for edge in sent['graphs']['universal-basic']['edges']:
            sentence[edge['destination'] + 1].set_conllu_field('head', edge['source'] + 1)
            sentence[edge['destination'] + 1].set_conllu_field('deprel', edge['relation'])
        for root in sent['graphs']['universal-basic']['roots']:
            sentence[root + 1].set_conllu_field('head', 0)
            sentence[root + 1].set_conllu_field('deprel', "root")

        add_basic_edges(sentence)
        sentences.append(sentence)

    return sentences


def _fix_sentence_keep_order(conllu_sentence):
    sorted_sent = sorted(conllu_sentence.items())
    addon = 0
    fixed = dict()

    for iid, token in sorted_sent:
        if round(iid) != iid:
            if "CopyOf" in token.get_conllu_field("misc"):
                token.set_conllu_field("form", token.get_conllu_field("form") + "[COPY_NODE]")
            addon += 1

        new_id = round(iid) + addon
        token.set_conllu_field("id", new_id)
        fixed[new_id] = token

    return fixed


def _fix_sentence_push_to_end(conllu_sentence):
    fixed = dict()

    for i, (iid, token) in enumerate([(iid2, t) for (iid2, t) in conllu_sentence.items() if iid2 != 0]):
        if round(iid) != iid:
            token.set_conllu_field("id", i + 1)

        fixed[i + 1] = token

    return fixed


def fix_sentence(conllu_sentence, push_new_to_end=True):
    if push_new_to_end:
        return _fix_sentence_push_to_end(conllu_sentence)
    else:
        return _fix_sentence_keep_order(conllu_sentence)


def fix_graph(conllu_sentence, odin_sentence, is_basic):
    if is_basic:
        odin_sentence["graphs"] = {"universal-basic": {"edges": [], "roots": []}}
    else:
        if 'graphs' in odin_sentence:
            odin_sentence["graphs"]["universal-enhanced"] = {"edges": [], "roots": []}
        else:
            odin_sentence["graphs"] = {"universal-enhanced": {"edges": [], "roots": []}}

    for iid, token in conllu_sentence.items():
        if iid == 0:
            continue

        if is_basic:
            if token.get_conllu_field("deprel").lower().startswith("root"):
                odin_sentence["graphs"]["universal-basic"]["roots"].append(iid - 1)
            else:
                odin_sentence["graphs"]["universal-basic"]["edges"].append(
                    {"source": token.get_conllu_field("head") - 1, "destination": iid - 1,
                     "relation": token.get_conllu_field("deprel")})
        else:
            for head, rel in token.get_new_relations():
                if rel.lower().startswith("root"):
                    odin_sentence["graphs"]["universal-enhanced"]["roots"].append(iid - 1)
                else:
                    odin_sentence["graphs"]["universal-enhanced"]["edges"].append(
                        {"source": head.get_conllu_field("id") - 1, "destination": iid - 1, "relation": rel})

    return odin_sentence


def append_odin(odin_sent, fixed_sentence, text):
    cur_sent_text = text
    cur_offset = 0

    for node in list(fixed_sentence.values())[len(odin_sent['words']):]:
        if node.get_conllu_field('id') == 0:
            continue

        if 'words' in odin_sent:
            odin_sent['words'].append(node.get_conllu_field('form'))
        if 'raw' in odin_sent:
            odin_sent['raw'].append(node.get_conllu_field('form'))
        if 'tags' in odin_sent:
            odin_sent['tags'].append(node.get_conllu_field('xpos'))
        if 'entities' in odin_sent:
            odin_sent['entities'].append('O')
        if ('startOffsets' in odin_sent) and ('endOffsets' in odin_sent):
            odin_sent['startOffsets'].append(odin_sent['endOffsets'][-1] + 1)
            odin_sent['endOffsets'].append(odin_sent['startOffsets'][-1] + len(node.get_conllu_field('form')))
        if 'lemmas' in odin_sent:
            odin_sent['lemmas'].append(node.get_conllu_field('lemma'))
        if 'chunks' in odin_sent:
            odin_sent['chunks'].append('O')

        cur_sent_text += " " + node.get_conllu_field('form')
        cur_offset += len(" " + node.get_conllu_field('form'))

    return odin_sent, cur_sent_text, cur_offset


def fix_offsets(odin_sent, all_offset):
    if ('startOffsets' in odin_sent) and ('endOffsets' in odin_sent):
        odin_sent['startOffsets'] = [(current + all_offset) for current in odin_sent['startOffsets']]
        odin_sent['endOffsets'] = [(current + all_offset) for current in odin_sent['endOffsets']]


def conllu_to_odin(conllu_sentences, odin_to_enhance=None, is_basic=False, push_new_to_end=True):
    odin_sentences = []
    fixed_sentences = []
    texts = []
    summed_offset = 0

    for i, conllu_sentence in enumerate(conllu_sentences):
        fixed_sentence = conllu_sentence

        if odin_to_enhance:
            text = odin_to_enhance['text'][
                   odin_to_enhance['sentences'][i]['startOffsets'][0]: odin_to_enhance['sentences'][i]['endOffsets'][
                       -1]]

            # fixing offsets may be to all sentences, as previous sentences may have become longer, changing all following offsets
            fix_offsets(odin_to_enhance['sentences'][i], summed_offset)

        # when added nodes appear fix sent
        if any([round(iid) != iid for iid in conllu_sentence.keys()]):
            fixed_sentence = fix_sentence(fixed_sentence, push_new_to_end)
            if odin_to_enhance:
                odin_to_enhance['sentences'][i], text, cur_offset = append_odin(odin_to_enhance['sentences'][i],
                                                                                fixed_sentence, text)
                summed_offset += cur_offset

        # store updated text for each sentence
        if odin_to_enhance:
            texts.append(text)

        # fix graph
        fixed_sentences.append(fixed_sentence)
        odin_sentences.append(fix_graph(
            fixed_sentence, odin_to_enhance['sentences'][i] if odin_to_enhance else
            {'words': [token.get_conllu_field("form") for token in fixed_sentence.values() if
                       token.get_conllu_field("id") != 0],
             'tags': [token.get_conllu_field("xpos") for token in fixed_sentence.values() if
                      token.get_conllu_field("id") != 0]},
            is_basic))

    if odin_to_enhance:
        odin_to_enhance['sentences'] = odin_sentences
        odin_to_enhance['text'] = "\n".join(texts)
        odin = odin_to_enhance
    else:
        odin = {"documents": {"": {
            "id": str(uuid.uuid4()),
            "text": " ".join([token.get_conllu_field("form") for conllu_sentence in fixed_sentences for (_, token) in
                              (sorted(conllu_sentence.items()) if not push_new_to_end else conllu_sentence.items()) if
                              token.get_conllu_field("id") != 0]),
            "sentences": odin_sentences
        }}, "mentions": []}

    return odin


def parsed_tacred_json(data):
    sentences = []
    for d in data:
        sentence = dict()
        for i, (t, p, h, dep) in enumerate(
                zip(d["token"], d["stanford_pos"], d["stanford_head"], d["stanford_deprel"])):
            sentence[i + 1] = Token(i + 1, t, t, p, p, "_", int(h), dep, "_", "_")
        sentence[0] = Token(0, None, None, None, None, None, None, None, None, None)
        add_basic_edges(sentence)
        [child.remove_edge(rel, sentence[0]) for child, rel in sentence[0].get_children_with_rels()]
        _ = sentence.pop(0)
        sentences.append(sentence)

    return sentences
