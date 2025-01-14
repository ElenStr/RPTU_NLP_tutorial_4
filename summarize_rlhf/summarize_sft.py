import os

import datasets
import torch
from torch.utils.data import Dataset
from mingpt.bpe import BPETokenizer




class SFTSummarize(Dataset):
    def __init__(self, split: str, block_size: int = 1024):
        self.split = split
        self.tokenizer = BPETokenizer()
        self.voc = 50257
        self.block_size = block_size
        ds = datasets.load_dataset("CarperAI/openai_summarize_tldr", split=split)
        def drop_long_examples(examples):
            prompts = []
            completions = []
            for prompt, completion in zip(examples['prompt'], examples['label']):
                if self.tokenizer(prompt + completion).size(1) <= block_size + 1:
                    prompts.append(prompt)
                    completions.append(completion)

            return {"prompt": prompts, "completion": completions}

        self.ds = ds.map(drop_long_examples, batched=True, remove_columns=ds.column_names, num_proc=os.cpu_count())


    def __len__(self):
        return len(self.ds)

    def get_vocab_size(self):
        return self.voc

    def get_block_size(self):
        return self.block_size

    def __getitem__(self, idx):
        sample = self.ds[idx]
        prompt = self.tokenizer(sample["prompt"]).squeeze(0)
        completion = self.tokenizer(sample["completion"]).squeeze(0)
        toks = torch.cat((prompt, completion))

        # attend to all tokens except the padding tokens
        mask = torch.full((self.block_size + 1,), False, dtype=bool)

        if len(toks) >= self.block_size + 1:
            toks = toks[-self.block_size - 1:]
        else:
            pad = torch.full((self.block_size + 1,), self.tokenizer.eot_token, dtype=torch.long)
            pad[:len(toks)] = toks

            # include a final eot token to predict
            mask[len(toks) + 1:] = True
            toks = pad

        x = toks[:-1]
        y = toks[1:].clone()

        # we only use the completion tokens to learn on
        y[mask[1:]] = -1 # ignore the loss from padding tokens
        # y[:len(prompt)-1] = -1 # and ignore the loss from the prompt tokens
        return x, y, mask[:-1]

