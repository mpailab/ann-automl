import gc
import sys
import os
import traceback

import torch

from ..core.nnfuncs import nnDB
from ..core.nn_auto import create_classification_model
from ..utils.time import time_as_str


def _get_classes(*categories):
    classes = nnDB.get_all_category_names()
    print(f"Available {len(classes)} classes")

    cat_classes = {}
    rest_categories = []
    for cat in categories:
        if cat in classes:
            cat_classes[cat] = [cat]
        else:
            rest_categories.append(cat)
    categories = rest_categories
    if not categories:
        return cat_classes

    print("Loading dialog language model to ask what classes correspond to request ...")
    from transformers import T5Tokenizer, AutoModelForSeq2SeqLM

    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl", device_map="auto")
    # get probability for yes and no answers

    print("Language model loaded")

    def get_cg_response(text, n=100):
        input_ids = tokenizer.encode(text, return_tensors="pt").to("cuda")
        max_length = input_ids.shape[1] + n

        output = model.generate(input_ids, max_length=max_length, do_sample=True, top_k=1,
                                pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(output[0], skip_special_tokens=True)

    cs = set(c.lower() for c in categories)
    # now we will choose only the classes that are in the categories

    if len(categories) > 1:
        ll = [f"Q: To what category does rat belong to: wild animals, pets, or neither?\nA: wild animals\n" \
              f"Q: To what category does sea belong to: vehicles, animals, plants, or neither?\nA: neither\n" \
              f"Q: To what category does car belong to: cars, bicycles, plants, or neither?\nA: cars\n" \
              f"Q: To what category does {class_name} belong to: {', '.join(categories)}, or neither?" for class_name in classes]
    else:
        ll = ["Q: Is rat an animal?\nA: Yes\n"
              "Q: Is Sun a planet?\nA: No\n"
              f"Q: Is {class_name} a {categories[0]}?" for class_name in classes]
    print(f"lll = {ll[0]}")
    # pad batch to the same length by adding pad_token_id
    pad_token_id = tokenizer.eos_token_id
    max_gen_len = 10
    max_batch_size = 2
    for i in range(len(classes)):
        token = get_cg_response(ll[i], max_gen_len).strip()
        if not token.startswith("A: "):
            print(f"Error in answer for {classes[i]}: {token}")
            continue

        token = token[3:]

        for j in range(0, len(token)):
            if token[j] in ',}':
                token = token[:j]
                break
        cat = token.strip().lower()
        print(f"{classes[i]} -> {token}")
        if len(categories) > 1:
            if cat in cs:
                cat_classes.setdefault(cat,[]).append(classes[i])
            elif cat != "neither":
                print(f"Error in lm classification: unknown category {cat}")
        else:
            if cat == "yes":
                cat_classes.setdefault(categories[0],[]).append(classes[i])
            elif cat != "no":
                print(f"Error in lm classification: unknown category {cat}")
    for cat in cs:
        if cat not in cat_classes:
            print(f"Warning: no classes found for category {cat}")
    model = None
    tokenizer = None
    return cat_classes


def get_classes(*categories):
    result = _get_classes(*categories)
    gc.collect()
    torch.cuda.empty_cache()
    return result


def get_subclasses(*categories):
    classes = get_classes(*categories)
    return [c for v in classes.values() for c in v]


classification_params = None


def gen_model(classes, output, target_accuracy=None, script_type='tf', time_limit=None, for_mobile=...):
    if target_accuracy is None:
        target_accuracy = 1
    if for_mobile is ...:
        for_mobile = output.endswith(".mlmodel") or output.endswith(".tflite")
    if time_limit is None:
        time_limit = 60*60
        choose_arch = not for_mobile and target_accuracy < 1
    else:
        choose_arch = not for_mobile

    print("Input:")
    print(f"classes = [{', '.join(classes)}]")
    print(f"output = {output}")
    if target_accuracy < 1:
        print(f"target_accuracy = {target_accuracy}")
        optimize_over_target = False
    else:
        optimize_over_target = True
    print(f"script_type = {script_type}")
    print(f"time_limit = {time_as_str(time_limit)}")
    print(f"for_mobile = {for_mobile}")
    if not classes:
        print("Error: no classes that matches your request were found. Try to reformulate in different way.")
        return
    if len(classes) == 1:
        print("Error: only one class found, cannot train classifier for a single class.")
        return
    global classification_params
    return dict(classes=classes, output_dir=output,
                target_accuracy=target_accuracy, script_type=script_type,
                time_limit=time_limit, for_mobile=for_mobile,
                optimize_over_target=optimize_over_target, choose_arch=choose_arch)


class RequestCodeGen:
    def __init__(self):
        self._model = None
        self._tokenizer = None
        # load text from file
        with open("data/lm/init.txt", "r", encoding='utf8') as f:
            self._init_prefix = f.read()
        self._chunk_size = 20

    def _load_model(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print("Loading CodeGen model ...")
        checkpoint = "Salesforce/codegen-6B-multi"
        self._model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda")
        self._tokenizer = AutoTokenizer.from_pretrained(checkpoint, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        print("CodeGen model loaded")

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    def close(self):
        self._model = None
        self._tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()

    def generate(self, text, max_new_tokens=1000, open='{', close='}'):
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to("cuda")
        length = input_ids.shape[1]
        print("length of input: ", length)

        pad_token_id = self.tokenizer.eos_token_id

        num_open = 1
        result = ""
        print("Generating code from request:")
        for i in range(0, max_new_tokens, self._chunk_size):
            outputs = self.model.generate(input_ids, max_length=self._chunk_size + i + length, do_sample=True, pad_token_id=pad_token_id,
                                     top_k=1)
            token = self.tokenizer.decode(outputs[0, i + length:self._chunk_size + i + length].to("cpu"))
            for j in range(0, len(token)):
                if token[j] == open:
                    num_open += 1
                elif token[j] == close:
                    num_open -= 1
                if num_open == 0:
                    token = token[:j]
                    break
            print(token, end="")
            result += token
            if num_open == 0:
                break
            input_ids = outputs
        return result

    def gen_initial_code(self, request):
        text = self._init_prefix + "\nRequest: {" + request + "}\nCode:{\n"
        return self.generate(text)


def protected_exec(code):
    context = {'get_classes': get_classes, 'get_subclasses': get_subclasses, 'gen_model': gen_model}
    loc = {}
    exec("from ann_automl.utils.time import *\n", context, loc)
    context["__builtins__"] = {}  # prevent access to builtins
    exec(code, context, loc)
    context.update(loc)
    return context


def get_params_from_request(request):
    cg = RequestCodeGen()
    code = cg.gen_initial_code(request)
    cg.close()

    print("Execute generated code ...")
    try:
        context = protected_exec(code)
        params = context["params"]
    except Exception as e:
        traceback.print_exc()
        print(f"Error: {repr(e)}")
        return f"Error: {repr(e)}"
    return params


def run(request, emulation=False):
    cg = RequestCodeGen()
    code = cg.gen_initial_code(request)
    cg.close()

    print("Execute generated code ...")
    try:
        context = protected_exec(code)
        params = context["params"]
    except Exception as e:
        traceback.print_exc()
        print(f"Error: {e}")
        return
    print("Code evaluated")
    if isinstance(params, dict):
        print("Start model creation")
        if emulation:
            print("Emulation mode, skipping model creation")
        else:
            create_classification_model(**params)
    else:
        raise Exception("Cannot correctly interpret input text. Please try to explain your request in a different way.")


class LMWorker:
    def run(self, f, *args, **kwargs):
        return f(*args, **kwargs)

