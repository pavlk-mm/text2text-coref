from collections import defaultdict
from itertools import chain
from typing import List
import re
import logging

logger = logging.getLogger(__name__)


def _correct_tags(tok_sentence):
    """
    This function takes a tokenized sentence and ensures that all tags are closed.
    When a tag is not closed or opened properly, the entity is converted into a
    single-token tag.

    Tags that cannot be parsed are thrown out.
    """
    num_wrong_para = 0

    if not tok_sentence:
        return []

    entity_stacks = defaultdict(lambda: [])
    clean_toks = []

    for word_idx, word in enumerate(tok_sentence):
        splits = word.split("|")

        if len(splits) == 1:
            clean_toks.append(word)

        elif len(splits) == 2:
            tags = splits[1].split(",")

            ptags = map(lambda x: re.match(r"(\[?)e(\d+)(]?)", x), tags)
            ptags = map(lambda x: x.groups() if x else ("", "", ""), ptags)

            clean_tags = []

            for left_bracket, entity_id, right_bracket in ptags:
                tag = f"{left_bracket}e{entity_id}{right_bracket}"

                if left_bracket and right_bracket:
                    clean_tags.append(tag)

                elif left_bracket:
                    entity_stacks[entity_id].append((word_idx, len(clean_tags)))
                    clean_tags.append(tag)

                elif right_bracket:
                    entity_stack = entity_stacks[entity_id]
                    if len(entity_stack) > 0:
                        entity_stacks[entity_id].pop()
                        clean_tags.append(tag)
                    else:
                        num_wrong_para += 1
                        clean_tags.append("[" + tag)

                else:
                    logging.debug(f"warning: completely invalid tag in: {word}")

            if clean_tags:
                clean_toks.append(f"{splits[0]}|{','.join(clean_tags)}")
            else:
                clean_toks.append(splits[0])

        else:
            logging.debug(f"warning: multiple pipes in word {word}- stripping tags")
            clean_toks.append(splits[0])

    # convert all unclosed entities to 1-word entities
    for entity, stack in entity_stacks.items():
        for word_idx, tag_idx in stack:
            try:
                num_wrong_para += 1

                token = clean_toks[word_idx]
                word, tags = token.split("|")

                tags = tags.split(",")

                assert tags[tag_idx] == f"(e{entity}", (
                    "Mismatched entity when correcting tags"
                )

                tags[tag_idx] = tags[tag_idx] + "]"

                clean_toks[word_idx] = f"{word}|{','.join(tags)}"

            except Exception as ex:
                logging.debug(f"{ex} while converting unclosed entitites")

    if num_wrong_para:
        sentence = " ".join(tok_sentence)
        logging.debug(
            f'{num_wrong_para} mismatched parantheses in sentence: "{sentence}"'
        )

    return clean_toks

def _correct_tags_eml(tok_sentence):
    """
    This function takes a tokenized sentence and ensures that all tags are closed.
    When a tag is not closed or opened properly, the entity is converted into a
    single-token tag.

    Tags that cannot be parsed are thrown out.
    """

    num_wrong_para = 0

    if not tok_sentence:
        return []

    entity_stacks = defaultdict(lambda: [])
    clean_toks = []

    for word_idx, word in enumerate(tok_sentence):
        ptags = re.findall(r"(</?e\d+>)", word)
        stripped_word = re.sub(r"(</?e\d+>)", "", word)

        if not ptags:
            clean_toks.append(([], stripped_word, []))

        else:
            clean_opening_tags = []
            clean_closing_tags = []

            for tag in ptags:
                m = re.match(r"</?e(\d+)>", tag)
                if m:
                    entity_id = m.group(1)

                    if tag.startswith("</"):
                        entity_stack = entity_stacks[entity_id]
                        if len(entity_stack) > 0:
                            entity_stacks[entity_id].pop()
                            clean_closing_tags.append(tag)
                        else:
                            num_wrong_para += 1
                            clean_opening_tags.append(tag.replace("</", "<"))
                            clean_closing_tags.insert(0, tag)

                    elif tag.startswith("<"):
                        entity_stacks[entity_id].append((word_idx, len(clean_opening_tags)))
                        clean_opening_tags.append(tag)

                else:
                    logging.debug(f"warning: completely invalid tag in: {word}")

            clean_toks.append((clean_opening_tags, stripped_word, clean_closing_tags))

    # convert all unclosed entities to 1-word entities
    for entity, stack in entity_stacks.items():
        for word_idx, tag_idx in stack:
            try:
                num_wrong_para += 1

                opening_tags, word, closing_tags = clean_toks[word_idx]

                assert opening_tags[tag_idx] == f"<e{entity}>", (
                    "Mismatched entity when correcting tags"
                )

                opening_tag = opening_tags[tag_idx]
                closing_tag = opening_tag.replace("<", "</")
                closing_tag_idx = len(opening_tags) - tag_idx - 1
                closing_tags.insert(closing_tag_idx, closing_tag)

                clean_toks[word_idx] = (opening_tags, word, closing_tags)

            except Exception as ex:
                logging.debug(f"{ex} while converting unclosed entitites")

    for i, (opening_tags, word, closing_tags) in enumerate(clean_toks):
        clean_toks[i] = "".join(opening_tags) + word + "".join(closing_tags)

    if num_wrong_para:
        sentence = " ".join(tok_sentence)
        logging.debug(
            f'{num_wrong_para} mismatched parantheses in sentence: "{sentence}"'
        )

    return clean_toks

def _word_level_edit_distance(words1, words2, tagged_words1, gold_zeros=False):
    """
    Uses an edit-distance-like algorithm to match up the words between
    two versions of a document. Tagged words are used to carry over
    as many entity annotations as possible - any words that remain the
    same or can be tracked back to a "replace" operation keep their tags.
    """
    m, n = len(words1), len(words2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # initialize the first row and column
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # fill the dp table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if not gold_zeros and re.sub(r"</?e\d+>", "", words1[i - 1]).startswith("##"):
                # empty nodes are ignored (deleted for free)
                dp[i][j] = dp[i - 1][j]
            elif words1[i - 1] == words2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1

    # backtrack to extract sentence with appropriate tags.
    result = []
    word_problems = defaultdict(int)

    i, j = m, n

    while i > 0 and j > 0:
        if not gold_zeros and re.sub(r"</?e\d+>", "", words1[i - 1]).startswith("##"):
            # empty nodes always copied over
            result.append(tagged_words1[i - 1])
            i -= 1
        elif words1[i - 1] == words2[j - 1]:
            # same case - actually use tags
            result.append(tagged_words1[i - 1])
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j - 1] + 1:
            # replace case
            result.append(words2[j - 1])
            word_problems["replace"] += 1
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            # delete case
            word_problems["delete"] += 1
            i -= 1
        else:
            # insert case
            result.append(words2[j - 1])
            word_problems["insert"] += 1
            j -= 1

    while j > 0:
        result.append(words2[j - 1])
        word_problems["insert"] += 1
        j -= 1

    if word_problems:
        logger.debug(f"word_problems: {dict(word_problems)}")

    result.reverse()
    return result

def _correct_basic_eml_syntax(document):
    """
    Corrects missing brackets and whitespaces in EML (XML-like) annotations.
    """
    # add missig closing brackets
    document = re.sub(r"</?e\w+\b(?!>)", r"\g<0>>", document)
    # add missing "<" in opening tags that are missing it
    document = re.sub(r"(?<![</])(e\w+>)", r"<\1", document)
    # add missing "<" in closing tags that are missing it
    document = re.sub(r"(?<!<)(/e\w+>)", r"<\1", document)


    # ensure there is a whitespace between closing and opening tags
    document = re.sub(r"</(e\w+)><(e\w+)>", r"</\1> <\2>", document)
    #ensure there is a whitespace before the first opening tag in a row of opening tags
    document = re.sub(r"(?<![\s>])(<e\w+>)+", r" \g<0>", document)
    # ensure there is a whitespace after the last closing tag in a row of closing tags
    document = re.sub(r"(</e\w+>)+(?![\s<])", r"\g<0> ", document)
    # ensure there is no whitespace between the openining tag and the next word
    document = re.sub(r"(<e\w+>)(\s+)(\S)", r"\1\3", document)
    # ensure there is no whitespace between the previous word and the closing tag
    document = re.sub(r"(\S)(\s+)(</e\w+>)", r"\1\3", document)
    # ensure there is no whitespace between multiple opening tags
    document = re.sub(r"<(e\w+)>\s*<(e\w+)>", r"<\1><\2>", document)
    # ensure there is no whitespace between multiple closing tags
    document = re.sub(r"</(e\w+)>\s*</(e\w+)>", r"</\1></\2>", document)
    # ensure there are no multiple whitespaces in a row
    document = re.sub(r"\s+", " ", document)
    return document.strip()


def _clean_document(document, gold_tok2, gold_zeros=False, format="txt"):
    """
    Applies both stages of cleaning on one document.
    """
    if format == "eml":
        document = _correct_basic_eml_syntax(document)

    doc_words = document.split()

    if format == "eml":
        stripped_doc = [re.sub(r"</?e\d+>", "", word) for word in doc_words]
    else:
        stripped_doc = [word.split("|")[0] for word in doc_words]

    flattened_gold = list(chain(*gold_tok2))

    correct_words = _word_level_edit_distance(
        stripped_doc, flattened_gold, doc_words, gold_zeros
    )

    final_sentences = []

    offset = 0
    for ref_sentence in gold_tok2:
        ln = len(ref_sentence)
        i = 0
        zeros = 0
        if not gold_zeros:
            while i < ln:
                if not re.sub(r"</?e\d+>", "", correct_words[offset + i + zeros]).startswith("##"):
                    # empty nodes are always copied over
                    i += 1
                else:
                    zeros += 1
        sentence = correct_words[offset : offset + ln + zeros]
        offset += ln + zeros
        correct_sentence = _correct_tags_eml(sentence) if format == "eml" else _correct_tags(sentence)
        assert len(correct_sentence) == len(ref_sentence) + zeros
        final_sentences.append(" ".join(correct_sentence))

    return " ".join(final_sentences)


def read_conllu(filename: str, zero_mentions: bool) -> List[List[List[str]]]:
    """
    Parses a CoNLL-U file into a list structure. Only loads the minimal information
    needed to correct sentence structure.

    The list structure is as follows:
    - first outer list corresponds to documents
    - the next list corresponds to sentences
    - final inner list corresponds to word tokens

    The zero mentions switch determines whether zero mentions should be included
    (True) or skipped (False).
    """
    with open(filename, "r", encoding="utf-8") as f:
        gold = f.readlines()

        gold_docs_tok2 = []
        next_doc = []
        next_sent: List[str] = []

        for line in gold:
            if not line.strip():
                continue

            if line.startswith("#"):
                begins_new_doc = line.startswith("# newdoc id")

                if line.startswith("# sent_id") or begins_new_doc:
                    if next_sent:
                        next_doc.append(next_sent)
                    next_sent = []

                if begins_new_doc:
                    if next_doc:
                        gold_docs_tok2.append(next_doc)
                    next_doc = []

                continue

            number, word = line.split()[:2]
            word = word.replace(" ", "_")
            if not zero_mentions and "." in number:
                continue  # skip empty nodes

            if "-" in number:
                continue  # always skip multitokens

            next_sent.append(word)

        next_doc.append(next_sent)
        gold_docs_tok2.append(next_doc)

    return gold_docs_tok2


def read_input_file(filename: str) -> List[str]:
    """
    Reads an input file as a list of documents.
    """
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def clean_data(
    docs: List[str], gold: List[List[List[str]]], gold_zeros: bool = False,
    format: str = "txt"
) -> List[str]:
    return [_clean_document(doc, gold_doc, gold_zeros, format) for doc, gold_doc in zip(docs, gold)]


def clean_file(
    filename: str,
    gold_filename: str,
    output_filename: str | None = None,
    zero_mentions: bool = True,
    format: str = "txt"
):
    logging.info(f"Reading input file: {filename}")
    data = read_input_file(filename)

    logging.info(f"Reading gold file: {gold_filename}")
    gold_docs_tok2 = read_conllu(gold_filename, zero_mentions)

    logging.info("Cleaning data")
    clean = clean_data(data, gold_docs_tok2, gold_zeros=zero_mentions, format=format)

    if not output_filename:
        output_filename = filename.replace(".txt", "-cleaned.txt")

    logging.info(f"Writing output file: {output_filename}")
    with open(output_filename, "w", encoding="utf-8") as f:
        clean = [line + "\n" for line in clean]
        f.writelines(clean)

def _test_eml_cleaning():
    doc = "This is a <e21 test and test <e56> ## </e56></e21> document  > with  <e2>entities/e2>e5> and</e5   some<e3> e4>invalid </e4> </e3>tags."
    #doc = "< # Example output: <e1>El Girona Convention Bureau</e1> i <e2>l' Associació Espanyola d' Agències d' Esdeveniments</e2> han signat avui <e3>un conveni</e3> per promoure <e4>l' organització d' <e5>esdeveniments a <e6>la demarcació</e6></e5></e4> . L' objectiu de el <e3>conveni</e3> és potenciar <e6>les comarques de Girona</e6> com a zona receptora d' esdeveniments culturals , esportius i turisme de negocis i beneficiar se , d' aquesta manera , d' un turisme d' alt poder adquisitiu . A través de el <e3>conveni</e3> , <e2>l' Associació Espanyola d' Agències d' Esdeveniments , <e2>que</e2> es va constituir el mes de novembre de l' any passat ,</e2> es compromet a promoure la col·laboració entre les agències d' esdeveniments i <e7>les empreses <e7>que</e7> formen part de el <e1>Girona Convention Bureau</e1></e7> . A més , <e3>l' acord</e3> permetrà obrir una línia de nous productes i serveis per a les empreses gironines relacionades amb <e8>el sector turístic</e8> ; difondre <e6>les comarques de Girona</e6> , tant a nivell nacional com internacional ; crear i desenvolupar <e9>programes de formació qualitativa</e9> impartint seminaris i cursos de postgrau adreçats a els empresaris de el <e8>sector</e8> i també incentivar i promoure <e10>la creació de <e11>noves agències d' esdeveniments</e11></e10> a la demarcació . <e1>El Girona Convention Bureau</e1> facilitarà , a través de el <e3>conveni</e3> , informació sobre <e7>les empreses de les comarques de Girona</e7> susceptibles de participar en <e4>l' organització d' <e5>esdeveniments</e5></e4> , promovent la col·laboració entre els seus membres . A més , <e12>l' entitat</e12> promourà <e10>la creació de <e11>noves agències</e11></e10> i farà difusió de <e9>els programes de formacions</e9> i de <e13>les accions promocionals de <e2>l' Associació Espanyola d' Agències d' Esdeveniments</e2></e13> . <e2>L' Associació Espanyola d' Agències d' Esdeveniments</e2> és <e14>una entitat d' agències de tot l' estat que es dediquen a <e15>la captació d' <e5>esdeveniments</e5> i viatges d' incentius</e15></e14> . <e16>El seu objectiu</e16> és acabar amb <e17>l' atomització de el <e8>sector</e8></e17> , organitzar activitats formatives i lluitar contra <e18>l' intrusisme professional en el <e8>sector</e8></e18> . <e19>El mercat espanyol de congressos i convencions</e19> mou uns 4 mil milions d' euros anuals de negoci ."
    cleaned_doc = _correct_basic_eml_syntax(doc)
    print(cleaned_doc)

if __name__ == "__main__":
    _test_eml_cleaning()
