
Quick note: the google collab has decided to stop working over a widget metadata issue. The code itself still runs great so just look to the .py file instead. 

# Unsloth-Finetuning-Haiku-project
Using UnSloth to Finetune Llama-3 and Use in Ollama

Taking a dataset on haiku downloaded from Kaggle UnSloth was used to fine-tune an ollama model into generating haiku poetry. 

Project interest is directly proportional to the amount of GPU you have access to. 

## **Install Dependencies**

On Google Notebook or whatever you are using for this, install your dependencies. 

```bash
!pip install unsloth
!pip install kaggle

!pip install trl

!pip install peft

!pip install torch
```

UnSloth is a parameter-effecient library designed specifically for fine-tuning LLMs. 
Kaggle is an excellent website for downloading various datasets for projects. 
Finally, torch is an extremely useful library for basically everything ML 


## **Download your Dataset**

Go to Kaggle and download their English-language haiku dataset (or a different dataset if you wish to finetune for other puproses)

https://www.kaggle.com/datasets/hjhalani30/haiku-dataset

To import into your notebook first download it to your PC. Then use the following code to transfer to google collab:

```bash
from google.colab import files

uploaded = files.upload()
```
Select your file and you're ready to roll. 

## **Finding an LLM that won't max out your free GPU consumption**

Using a free version of Google Collab there are different tactics you can use so as to not max out your GPU:
1) Use a small model such as a 1B Llama.
2) reduce your memory utilization
3) reduce your batch size or gradient accumulation steps
4) reduce your training epochs

I recommend using HuggingFace to pull various models to find a free one that works for you. Create an account, input your secret key, and should be good to go. 


```bash
%%capture
# Skip restarting message in Colab
import sys; modules = list(sys.modules.keys())
for x in modules: sys.modules.pop(x) if "PIL" in x or "google" in x else None

!pip install unsloth vllm
!pip install --upgrade pillow
# If you are running this notebook on local, you need to install `diffusers` too
# !pip install diffusers
# Temporarily install a specific TRL nightly version
!pip install git+https://github.com/huggingface/trl.git@e95f9fb74a3c3647b86f251b7e230ec51c64b72b
```
and for UnSloth:

```bash

from unsloth import is_bfloat16_supported #type:ignore
import torch
from unsloth import FastLanguageModel
max_seq_length = 256 # Can increase for longer reasoning traces


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/Llama-3.2-1B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    #max_lora_rank = lora_rank, #type:ignore
    gpu_memory_utilization = 0.6, # Reduce if out of memory
    device_map="auto",
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 8, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = 8,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)

```

## **Further Research**

Using my code you should be able to fine-tune an LLM for improved haiku poem generation. Granted it's a little lame using only free GPUs but it's a good proof of concept. 

For further research: 

UnSloth: https://unsloth.ai

Hugging Face Models: https://huggingface.co/models






