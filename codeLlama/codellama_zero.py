import torch
from transformers import (
    TrainingArguments,
    Trainer,
    ResNetModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    CodeLlamaTokenizer
)
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
import accelerate
import bitsandbytes
from transformers.modeling_outputs import SequenceClassifierOutput
import json
import time
import csv
import re
import pandas as pd
import shutil
from datasets import load_dataset

from utils import * 

device = torch.device('cuda')

mbpp = load_dataset("mbpp")  # train, validation, and test
humaneval = load_dataset("openai_humaneval")  # test only

model_id = "codellama/CodeLlama-7b-Instruct-hf"
quantization_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
)

def generate_output(prompts):
    output_texts = []
    total = len(prompts)
    count = 0
    for prompt  in prompts :
        count += 1
        print(count/total, end = ' ', flush = True)
        match = re.search(r'def (\w+)\(', prompt)
        if match:
            function_name = match.group(1)
        else:
            function_name=''
        full_prompt = (
            f"""
            Please refactor the following Python program to a more readable, efficient, and maintainable one:
            - The given program is correct but needs improvement
            - DO NOT change the name of the program
            - DO NOT change the input or output behavior of the program (e.g., number of inputs / outputs, input / output types, etc.)
            - Put your response in a markdown code block
            - Respond with only the code block
            - Don't explain the changes made

            Again, do not change the name of the function in any way. The function name should remain "{function_name}".

            \nCode: \n{prompt}\n\n
            """.strip()
        )

        system_instruction = "You are an expert programmer."
        PROMPT = "<s>[INST] <<SYS>>\\n{system_instruction}\\n<</SYS>>\\n\\n{user}[/INST]"
        prompt = PROMPT.format(system_instruction = system_instruction, user=full_prompt)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        output = model.generate(
            inputs["input_ids"],
            max_new_tokens=2048,
            do_sample=True,
            top_p=0.9,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id
        )
        output = output[0].to(device)
        output_text = tokenizer.decode(output)
        output_dict = {'prompt': prompt, 'output': output_text}
        output_texts.append(output_dict)
    return output_texts


def refactor_zero(humaneval_run=False, mbpp_run=False):
    if mbpp_run: 
        mbpp_prompts=[]
        for i in mbpp["test"]:
            concat_prompt = i['code']
            mbpp_prompts.append(concat_prompt)
        
        mbpp_outputs = generate_output(mbpp_prompts)
        parsed_csv('./mbpp_zero.csv',mbpp_outputs)

    if humaneval_run: 
        humaneval_prompts = []
        for i in humaneval["test"]:
            concat_prompt = i['prompt'] + i['canonical_solution']
            humaneval_prompts.append(concat_prompt)

        humaneval_outputs = generate_output(humaneval_prompts)
        parsed_csv('./humaneval_zero.csv',humaneval_outputs)


refactor_zero(humaneval_run = False, mbpp_run = True)