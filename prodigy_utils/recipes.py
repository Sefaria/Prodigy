import prodigy
import spacy
import re
import os
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from spacy.lang.he import Hebrew
from spacy.util import minibatch, compounding
from prodigy.components.preprocess import add_tokens, split_spans
from db_manager import MongoProdigyDBManager
from pathlib import Path


@spacy.registry.tokenizers("inner_punct_tokenizer")
def inner_punct_tokenizer_factory():
    def inner_punct_tokenizer(nlp):
        # infix_re = spacy.util.compile_infix_regex(nlp.Defaults.infixes)
        infix_re = re.compile(r'''[\.\,\?\:\;…\‘\’\`\“\”\"\'~\–\-/\(\)]''')
        prefix_re = spacy.util.compile_prefix_regex(nlp.Defaults.prefixes)
        suffix_re = spacy.util.compile_suffix_regex(nlp.Defaults.suffixes)

        return Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
                         suffix_search=suffix_re.search,
                         infix_finditer=infix_re.finditer,
                         token_match=None)
    return inner_punct_tokenizer


def load_model(model_dir, labels, lang):
    model_exists = model_dir is not None
    try:
        if model_dir is None:
            raise OSError("model_dir is None")
        nlp = spacy.load(model_dir)
    except OSError:
        model_exists = False
        nlp = Hebrew() if lang == 'he' else English()
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer", first=True)
    if "ner" not in nlp.pipe_names:
        nlp.add_pipe("ner", last=True)
    ner = nlp.get_pipe("ner")
    for label in labels:
        ner.add_label(label)
    if not model_exists:
        nlp.begin_training()
    nlp.tokenizer = inner_punct_tokenizer_factory()(nlp)
    return nlp, model_exists


def save_model(nlp, model_dir):
    model_dir = Path(model_dir)
    if not model_dir.exists():
        model_dir.mkdir()
    nlp.to_disk(model_dir)


def train_model(nlp, annotations, model_dir):
    batches = minibatch(annotations, size=compounding(4.0, 32.0, 1.001))
    losses = {}
    # TODO fix training no longer compatible with spacy 3.  The Language.update method takes a list of Example objects, but got: {<class 'dict'>}
    # for batch in batches:
    #     nlp.update(batch, losses=losses)  # TODO add drop=0.5
    save_model(nlp, model_dir)
    return losses


def add_model_predictions(nlp, stream, min_found=None):
    """
    Return new generator that wraps stream
    add model predictions to each example of stream
    """
    for example in stream:
        pred_doc = nlp(example['text'])
        pred_spans = [{"start": pred_doc[ent.start].idx, "end": pred_doc[ent.end-1].idx+len(pred_doc[ent.end-1]), "label": ent.label_} for ent in pred_doc.ents]
        example['spans'] = pred_spans
        if min_found is not None and len(pred_spans) < min_found:
            continue
        yield example


def filter_labels(stream, labels: set):
    for example in stream:
        example['spans'] = [span for span in example['spans'] if span['label'] in labels]
        yield example


def score_stream(nlp, stream):
    ner = nlp.get_pipe("ner")
    for example in stream:
        docs = nlp(example['text'])
        score = ner.predict(docs)
        yield score[0], example


def filter_existing_in_output(in_data, my_db:MongoProdigyDBManager):
    def get_key(doc):
        return tuple(sorted(doc['meta'].items(), key=lambda x: x[0]))
    existing_keys = set()
    for doc in my_db.output_collection.find({}):
        existing_keys.add(get_key(doc))
    for in_doc in in_data:
        in_doc_key = get_key(in_doc)
        if in_doc_key in existing_keys: continue
        yield in_doc


def filter_long_texts(stream, max_length):
    for example in stream:
        if len(example['text']) > max_length: continue
        yield example


def split_sentences_nltk(stream):
    """
    NLTK seems to have a better sentencizer than spacy
    Still, messing up on Deut. 23 and other similar cases
    """
    from nltk.tokenize import sent_tokenize

    for example in stream:
        sentences = sent_tokenize(example['text'])
        for sent in sentences:
            assert isinstance(example, dict)
            sent_example = example.copy()
            sent_example['text'] = sent
            yield sent_example


def train_on_current_output(output_collection='examples2_output'):
    model_dir = '/prodigy-disk/ref_tagging_model_output'
    nlp, model_exists = load_model(model_dir)
    my_db = MongoProdigyDBManager(output_collection, 'mongo', 27017)
    prev_annotations = list(my_db.output_collection.find({}, {"_id": 0}))
    print(len(prev_annotations))
    losses = train_model(nlp, prev_annotations, model_dir)
    print(losses.get('ner', None))


@prodigy.recipe(
    "ner-recipe",
    dataset=("Dataset to save answers to", "positional", None, str),
    input_collection=("Mongo collection to input data from", "positional", None, str),
    output_collection=("Mongo collection to output data to", "positional", None, str),
    labels=("Labels for classification", "positional", None, str),
    model_dir=("Spacy model location", "option", None, str),
    view_id=("Annotation interface", "option", "v", str),
    db_host=("Mongo host", "option", None, str),
    db_port=("Mongo port", "option", None, int),
    user=("Mongo Username", "option", None, str),
    replicaset_name=("Mongo Replicaset Name", "option", None, str),
    dir=("Direction of text to display. Either 'ltr' or 'rtl'", "option", None, str),
    lang=("Lang of training data. Either 'en' or 'he'", "option", None, str),
    train_on_input=("Should empty model be trained on input spans?", "option", None, int),
    should_add_predictions=("When there is an existing model, should you use it to add predictions to input", "option", None, int),
)
def ref_tagging_recipe(dataset, input_collection, output_collection, labels, model_dir=None, view_id="text", db_host="localhost", user="", replicaset_name="", db_port=27017, dir='rtl', lang='he',train_on_input=1, should_add_predictions=1):
    password = os.getenv('MONGO_PASSWORD', '')
    my_db = MongoProdigyDBManager(output_collection, host=db_host, port=db_port, user=user, password=password, replicaset_name=replicaset_name)
    print("OUTPUT: " + str(list(my_db.client.list_databases())))
    print(f"collection in output db: {my_db.output_collection.count_documents({})}")
    print(my_db.client)
    labels = labels.split(',')
    nlp, model_exists = load_model(model_dir, labels, lang)
    if not model_exists and train_on_input == 1 and model_dir is not None:
        print("Training on input to initialize model")
        temp_stream = getattr(my_db.db, input_collection).find({}, {"_id": 0})
        train_model(nlp, temp_stream, model_dir)
    all_data = list(getattr(my_db.db, input_collection).find({}, {"_id": 0}))  # TODO loading all data into ram to avoid issues of cursor timing out
    stream = filter_existing_in_output(all_data, my_db)
    # stream = split_sentences_nltk(stream)
    stream = filter_long_texts(stream, max_length=5000)
    if model_exists and should_add_predictions == 1:
        stream = add_model_predictions(nlp, stream, min_found=1)
    stream = add_tokens(nlp, stream, skip=True)
    if view_id == "ner":
        stream = split_spans(stream)

    def update(annotations):
        prev_annotations = my_db.db.examples.find({}, {"_id": 0}).limit(1000).sort([("_id", -1)])
        all_annotations = list(prev_annotations) + list(annotations)
        losses = train_model(nlp, all_annotations, model_dir)
        return losses.get('ner', None)

    def progress(ctrl, update_return_value):
        return update_return_value
        #return ctrl.session_annotated / getattr(my_db.db, input_collection).count_documents({})

    return {
        "db": my_db,
        "dataset": dataset,
        "view_id": view_id,
        "stream": stream,
        "progress": progress,
        # "update": update,
        "config": {
            "labels": labels,
            "global_css": f"""
                [data-prodigy-view-id='{view_id}'] .prodigy-content {{
                    direction: {dir};
                    text-align: {'right' if dir == 'rtl' else 'left'};
                }}
            """,
            "javascript": """
            function scrollToFirstAnnotation() {
                var scrollableEl = document.getElementsByClassName('prodigy-annotator')[0];
                var markEl = document.getElementsByTagName('mark')[0];
                scrollableEl.scrollTop = markEl.offsetTop;
            }
            document.addEventListener('prodigymount', function(event) {
                scrollToFirstAnnotation();
            })
            document.addEventListener('prodigyanswer', function(event) {
                scrollToFirstAnnotation();
            })
            """
        }
    }


def validate_tokenizer(model_dir, s, lang):
    nlp, _ = load_model(model_dir, ['na'], lang)
    for token in nlp.tokenizer(s):
        print(token.text)


def validate_alignment(model_dir, lang, text, entities):
    nlp, exists = load_model(model_dir, ['na'], lang)
    print("Model Exists:", exists)
    print(spacy.training.offsets_to_biluo_tags(nlp.make_doc(text), entities))


if __name__ == "__main__":
    model_dir = "/home/nss/sefaria/data/research/prodigy/output/webpages/model-last"
    validate_tokenizer(model_dir, "ה, א-ב", 'he')
    validate_alignment(model_dir, 'he', "פסוקים א-ו", [(0, 8, 'מספר'), (8, 9, 'סימן-טווח'), (9, 10, 'מספר')])
