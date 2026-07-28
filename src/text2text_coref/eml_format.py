
import udapi
from udapi.block.corefud.movehead import MoveHead
from collections import defaultdict
import logging
from .convert import read_data, shift_empty_node_recreate, shift_empty_node, write_data, remove_empty_node, reduce_discontinuous_mention
import re


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()

def parse_eml_word(text):
    """Parse XML-like formatted word into opening tags, word text, and closing tags.
    
    Args:
        text: String like '<e25821><e25756>L2</e25756>'
    
    Returns:
        tuple: (opening_tags, word_text, closing_tags)
        Example: (['e25821', 'e25756'], 'L2', ['e25756'])
    """
    import re
    
    opening_tags = []
    closing_tags = []
    word_text = text
    
    # Extract all opening tags from the beginning
    while word_text.startswith('<') and not word_text.startswith('</'):
        match = re.match(r'^<([^/>]+)>', word_text)
        if match:
            opening_tags.append(match.group(1))
            word_text = word_text[match.end():]
        else:
            break
    
    # Extract all closing tags from the end
    while '</e' in word_text:
        match = re.search(r'<\/([^>]+)>$', word_text)
        if match:
            closing_tags.insert(0, match.group(1))  # Insert at beginning to maintain order
            word_text = word_text[:match.start()]
        else:
            break
    
    return opening_tags, word_text, closing_tags

def convert_eml_file_to_conllu(filename, skeleton_filename, output_filename, zero_mentions=False):
    if not output_filename:
        output_filename = filename.replace(".eml", ".conllu")
    with open(filename, encoding="utf-8") as f:
        text_docs = f.read().splitlines()
        convert_eml_to_conllu(text_docs, skeleton_filename, output_filename, zero_mentions)

def convert_eml_to_conllu(text_docs, conllu_skeleton_file, output_file, use_gold_empty_nodes=True):
    udapi_docs = read_data(conllu_skeleton_file)
    # udapi_docs2 = read_data(conllu_skeleton_file)
    move_head = MoveHead()
    for doc in udapi_docs:
        doc._eid_to_entity = {}
    assert len(udapi_docs) == len(text_docs)
    for text, udapi_doc in zip(text_docs, udapi_docs):
        words = text.split(" ")
        udapi_words = [word for word in udapi_doc.nodes]
        for word in udapi_doc.nodes_and_empty:   
            word.misc["Entity"] = None
            word.misc["Bridge"] = None
            word.misc["SplitAnte"] = None
            # Remove empty nodes
            if not use_gold_empty_nodes and word.is_empty():
                remove_empty_node(word)
            elif word.is_empty():
                shift_empty_node_recreate(word)
        if not use_gold_empty_nodes:
            j = 1
            for i in range(len(udapi_words)):
                word = udapi_words[i]
                while j < len(words) and re.sub(r"</?e\d+>", "", words[j]).startswith("##"):
                    w = words[j]
                    print(f"Creating empty node for word: {words[j]} at position {j}")
                    word.create_empty_child("dep", after=True)
                    j += 1
                j += 1
        udapi_words = [word for word in udapi_doc.nodes_and_empty]
        assert len(udapi_words) == len(words)
        mention_starts = defaultdict(list)
        entities = {}
        
        # Parse XML-like format
        for i, (word, udapi_word) in enumerate(zip(words, udapi_words)):
            # Extract the actual word and tags from XML-like format
            opening_tags, word_text, closing_tags = parse_eml_word(word)
            
            if word_text != udapi_word.form:
                logger.warning(f"WARNING: words do not match. DOC: {udapi_doc.meta['docname']}, word1: {word_text}, word2: {udapi_word.form}")
            
            # Process opening tags
            for eid in opening_tags:
                if eid not in entities:
                    entities[eid] = udapi_doc.create_coref_entity(eid=eid)
                mention_starts[eid].append(i)
            
            # Process closing tags
            for eid in closing_tags:
                if not mention_starts[eid]:
                    logger.warning(f"WARNING: Closing mention which was not opened. DOC: {udapi_doc.meta['docname']}, EID: {eid}")
                    continue
                entities[eid].create_mention(words=udapi_words[mention_starts[eid][-1]: i + 1])
                mention_starts[eid].pop()

        udapi.core.coref.store_coref_to_misc(udapi_doc)
        move_head.run(udapi_doc)
    # debug_udapi(udapi_docs, udapi_docs2)
    with open(output_file, "w", encoding="utf-8") as f:
        write_data(udapi_docs, f)

def convert_conllu_file_to_eml(filename, output_filename, zero_mentions, blind=False, sequential_ids=True, no_empty_node_form=False):
    if not output_filename:
        output_filename = filename.replace(".conllu", ".eml")
    docs = read_data(filename)
    convert_to_eml(docs, output_filename, zero_mentions, not blind, sequential_ids, not no_empty_node_form)

def convert_to_eml(docs, out_file, solve_empty_nodes=True, mark_entities=True, sequential_ids=False, empty_node_form=True):
    with open(out_file, "w", encoding="utf-8") as f:
        for doc in docs:
            eids = {}
            out_words = []
            if solve_empty_nodes:
                for node in doc.nodes_and_empty:
                    if node.is_empty():
                        # node.shift_before_node(node.deps[0]["parent"])
                        shift_empty_node(node)
                udapi_words = [word for word in doc.nodes_and_empty]
            else:
                udapi_words = [word for word in doc.nodes]
            for word in udapi_words:
                out_word = word.form.replace(" ", "_")
                if word.is_empty():
                    out_word = "##" + (out_word if out_word != "_" and empty_node_form else "") # empty nodes start with ##
                mentions = []
                # Collect mention start and end positions for proper nesting of XML-like tags
                mention_starts = []
                mention_ends = []
                if mark_entities:
                    for mention in sorted(set(word.coref_mentions)):
                        if sequential_ids:
                            if mention.entity.eid not in eids:
                                eids[mention.entity.eid] = f"e{len(eids) + 1}"
                            eid = eids[mention.entity.eid]
                        else:
                            eid = mention.entity.eid
                        if "," in mention.span:
                            reduce_discontinuous_mention(mention)
                        span = mention.span
                        mention_start = float(span.split("-")[0])
                        mention_end = float(span.split("-")[1]) if "-" in span else mention_start
                        if mention_start == float(word.ord) or mention_end == float(word.ord):
                            mention_starts.append(mention_start)
                            mention_ends.append(mention_end)
                        if mention_start == float(word.ord) and mention_end == float(word.ord):
                            mentions.append(f"[{eid}]")
                        elif mention_start == float(word.ord):
                            mentions.append(f"[{eid}")
                        elif mention_end == float(word.ord):
                            mentions.append(f"{eid}]")
                # Convert bracket format to XML-like tags
                opening_tags = []
                closing_tags = []
                # Span lengths to help with proper nesting
                opened_ends = []
                closed_starts = []
                for mention, start, end in zip(mentions, mention_starts, mention_ends):
                    if mention.startswith('[') and mention.endswith(']'):
                        # Single-word mention: [e1]
                        eid = mention[1:-1]
                        opening_tags.append(f"<{eid}>")
                        closing_tags.append(f"</{eid}>")
                        opened_ends.append(end)
                        closed_starts.append(start)
                    elif mention.startswith('['):
                        # Opening: [e1
                        eid = mention[1:]
                        opening_tags.append(f"<{eid}>")
                        opened_ends.append(end)
                    else:
                        # Closing: e1]
                        eid = mention[:-1]
                        closing_tags.append(f"</{eid}>")
                        closed_starts.append(start)

                # Ensure proper nesting by sorting tags:
                # The closing tag with the highest corresponding start comes first
                closing_tags = [tag for _, tag in sorted(zip(closed_starts, closing_tags), key=lambda x: (-x[0], x[1]))]
                # The opening tag with the highest corresponding end comes first
                opening_tags = [tag for _, tag in sorted(zip(opened_ends, opening_tags), reverse=True)]
                
                # Combine tags and word
                out_words.append(''.join(opening_tags) + out_word + ''.join(closing_tags))
            f.write(" ".join(out_words) + "\n")

