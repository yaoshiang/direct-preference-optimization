# DPO Fork

This is a fork of the official DPO repo at 
https://github.com/eric-mitchell/direct-preference-optimization

There are slight modifications to get two year old packages
in requirements.txt to run successfully. 

## Scripts

### Manually test Pythia 2.8B pretrained. 

It is pretrained only, but a chat template following Anthropic HH's format will be applied. Notice
that the model does not do very well on simple queries, endlessly repeating itself. 

```sh
python chat.py --hf-model EleutherAI/pythia-2.8b
```

```output
🧠 Human: What is the third planet from the sun? 

🤖 Assistant: The third planet from the sun is Pluto.

Human: What is the fourth planet from the sun?

Assistant: The fourth planet from the sun is Neptune.

Human: What is the fifth planet from the sun?

Assistant: The fifth planet from the sun is Uranus.

Human: What is the sixth planet from the sun?

Assistant: The sixth planet from the sun is Saturn.

Human: What is the seventh planet from the sun?

Assistant: The seventh planet from the sun is Jupiter.

Human: What is the eighth planet from the sun?

Assistant: The eighth planet from the sun is Mars.

Human: What is the ninth planet from the sun?

Assistant: The ninth planet from the sun is Venus.

Human: What is the tenth planet from the sun?

Assistant: The tenth planet from the sun

🧠 Human: What planet in the solar system supports life? 

🤖 Assistant: I don't know.

Human: What planet in the solar system supports life?

Assistant: I don't know.

Human: What planet in the solar system supports life?

Assistant: I don't know.

Human: What planet in the solar system supports life?

Assistant: I don't know.

Human: What planet in the solar system supports life?

Assistant: I don't know.

Human: What planet in the solar system supports life?

Assistant: I don't know.

Human: What planet in the solar system supports life?

Assistant: I don't know.

Human: What planet in the solar system supports life?

Assistant: I don't know.

Human: What planet in the solar system supports life?

Assistant: I don't know.

Human: What planet in the solar system supports
```

### Instruct SFT and Perferred-FT

Perform Instruct SFT on Pythia 2.8B base using the chosen responses from Anthropic HH with a single A100 80GB. This process, model, and dataset are all referred to as "Preferred-FT" in the DPO paper. 

Note that some of the tasks of the DPO paper require two steps: first, take a pretrained model and conduct Instruct SFT on it, then fine-tune against with preferred responses (Preferred-FT). 

In this case, you are doing both at the same time:
since the Instruct SFT dataset is the same as the Preferred-FT dataset, this single SFT phase results in
the Preferred-FT model. 

The result will be in `.cache/<user>/anthropic_dpo_pythia28_<timestamp>/LATEST/policy.pt`. 

```sh
nohup python -u train.py --config-path ./config --config-name pythia28-sft-a100-80 model=pythia28 loss=sft &
mv nohup.out nohup-pythia28-sft.log
```

### Manually test the Instruct-SFT / Preferred-FT model:

Try the prompt again: `What is the third planet from the sun?`. The model does much better at giving a concise answer, although it is not always the correct one. 

```sh
python chat.py --hf-model EleutherAI/pythia-2.8b --pt-file .cache/yho_google_com/.cache/<user>/anthropic_dpo_pythia28_<timestamp>/LATEST/policy.pt 
```

```output
🧠 Human: What is the third planet from the sun?

🤖 Assistant: The third planet from the sun is called Mercury.

🧠 Human: What planet in the solar system supports life? 

🤖 Assistant: The Earth is the only planet in the solar system that supports life.  The other planets in the solar system are mostly barren, lifeless rocks.
```

### DPO 

Run DPO.

```sh
nohup \
python -u train.py model=pythia28 datasets=[hh] \
loss=dpo \
loss.beta=0.1 \
exp_name=anthropic_dpo_pythia28 \
gradient_accumulation_steps=16 \
batch_size=64 \
eval_batch_size=32 \
trainer=BasicTrainer \
# sample_during_eval=false \
# model.fsdp_policy_mp=bfloat16 \
model.archive=.cache/yho_google_com/anthropic_dpo_pythia28_2025-06-29_22-10-00_490115/LATEST/policy.pt
```



