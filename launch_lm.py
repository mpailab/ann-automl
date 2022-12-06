import ast
import sys

import torch
import time
import datetime


from ann_automl.core.nnfuncs import nnDB


def get_classes(*categories):
    print("Loading dialog language model to ask what classes correspond to request ...")
    from transformers import T5Tokenizer, T5ForConditionalGeneration

    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto")
    print("Language model loaded")

    def get_cg_response(text, n=100):
        input_ids = tokenizer.encode(text, return_tensors="pt").to("cuda")
        max_length = input_ids.shape[1] + n
        output = model.generate(input_ids, max_length=max_length, do_sample=True, top_k=1,
                                pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(output[0], skip_special_tokens=True)

    classes = nnDB.get_all_category_names()
    print(f"Available {len(classes)} classes")
    cs = set(c.lower() for c in categories)
    # now we will choose only the classes that are in the categories
    with open("data/lm/class_to_cat.txt", "r") as f:
        prefix = f.read()
        ll = [f"Q: To which categories does match {class_name}: {', '.join(categories)}, or no?\nA:" for class_name in classes]
        print(f"lll = {ll[0]}")
        # pad batch to the same length by adding pad_token_id
        pad_token_id = tokenizer.eos_token_id
        max_gen_len = 10
        max_batch_size = 2
        cat_classes = {}
        for i in range(len(classes)):
            token=get_cg_response(ll[i], max_gen_len)
            for j in range(0, len(token)):
                if token[j] in ',}':
                    token = token[:j]
                    break
            cat = token.strip().lower()
            print(f"{classes[i]} -> {token}")
            if cat in cs:
                cat_classes.setdefault(cat,[]).append(classes[i])
            elif cat != "no":
                print(f"Error in lm classification: unknown category {cat}")
        for cat in cs:
            if cat not in cat_classes:
                print(f"Warning: no classes found for category {cat}")
    return cat_classes


def get_subclasses(*categories):
    classes = get_classes(*categories)
    return [c for v in classes.values() for c in v]


def today(hour=23,minute=59,second=59):
    return datetime.datetime.combine(datetime.date.today(), datetime.time(hour,minute,second))


def tomorrow(hour=0,minute=0,second=0):
    return datetime.datetime.combine(datetime.date.today()+datetime.timedelta(days=1), datetime.time(hour,minute,second))


def this_week(day=6,hour=23,minute=59,second=59):
    return datetime.datetime.combine(datetime.date.today()+datetime.timedelta(days=day-datetime.date.today().weekday()), datetime.time(hour,minute,second))


def next_week(day=6,hour=23,minute=59,second=59):
    return datetime.datetime.combine(datetime.date.today()+datetime.timedelta(days=7+day-datetime.date.today().weekday()), datetime.time(hour,minute,second))


def this_month(day=31,hour=23,minute=59,second=59):
    return datetime.datetime.combine(datetime.date.today()+datetime.timedelta(days=day-datetime.date.today().day), datetime.time(hour,minute,second))


def next_month(day=31,hour=23,minute=59,second=59):
    return datetime.datetime.combine(datetime.date.today()+datetime.timedelta(days=31+day-datetime.date.today().day), datetime.time(hour,minute,second))


def until(t: datetime.datetime):
    return (t-datetime.datetime.now()).total_seconds()


def time_as_str(t):
    if t < 60:
        return f"{t:.2f} seconds"
    elif t < 60*60:
        return f"{t/60:.2f} minutes"
    elif t < 60*60*24:
        return f"{t/60/60:.2f} hours"
    else:
        return f"{t/60/60/24:.2f} days"


def gen_model(classes,output,target_accuracy=None,script_type='tf',time_limit=...,for_mobile=...):
    if target_accuracy is None:
        target_accuracy = 1
    if for_mobile is ...:
        for_mobile = output.endswith(".mlmodel") or output.endswith(".tflite")
    if time_limit is ...:
        time_limit = 30*24*60*60

    print("Input:")
    print(f"classes = [{', '.join(classes)}]")
    print(f"output = {output}")
    if target_accuracy<1:
        print(f"target_accuracy = {target_accuracy}")
    print(f"script_type = {script_type}")
    print(f"time_limit = {time_as_str(time_limit)}")
    print(f"for_mobile = {for_mobile}")


def main(request):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("Loading CodeGen model ...")
    checkpoint = "Salesforce/codegen-6B-multi"
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    print("CodeGen model loaded")

    t0 = time.time()
    # load text from file
    with open("data/lm/init.txt", "r", encoding='utf8') as f:
        text = f.read() + "\nRequest: {" + request + "}\nCode:{\n"
    input_ids = tokenizer.encode(text, return_tensors="pt").to("cuda")
    length = input_ids.shape[1]
    print("length of input: ", length)

    pad_token_id = tokenizer.eos_token_id

    ll = 1000
    d = 20
    num_open = 1
    result = ""
    print("Generating code from request:")
    for i in range(0,ll,d):
        outputs = model.generate(input_ids, max_length=d+i+length, do_sample=True, pad_token_id=pad_token_id, top_k=1)
        token = tokenizer.decode(outputs[0, i+length:d+i+length].to("cpu"))
        for j in range(0,len(token)):
            if token[j] == "{":
                num_open += 1
            elif token[j] == "}":
                num_open -= 1
            if num_open == 0:
                token = token[:j]
                break
        print(token, end="")
        result += token
        if num_open == 0:
            break
        input_ids = outputs
    model = None
    tokenizer = None  # free memory
    print()
    print(f"Time: {time.time() - t0:.2f}s, {(outputs.shape[1]-length)/(time.time() - t0):.2f} it/s")
    print("Code generated")

    print("Execute generated code ...")
    exec(result)
    print("Code evaluated")


if __name__ == "__main__":
    # takes 1 argument: the request (as a string)
    main(sys.argv[1])
