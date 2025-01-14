
import json
import os
import random
import sys
from ast import literal_eval
from contextlib import nullcontext
from torch.utils.data import Dataset
import datasets
import numpy as np
import torch

from mingpt.char_tokenizer import CharTokenizer

# -----------------------------------------------------------------------------

def lr_schedule(max_lr, max_iters, n_warmup=100, min_percent=0.1):
    "lr scheduler with linear warmup steps to max_lr, then a linear decay to min_percent*max_lr"
    def get_lr(current_step):
        if current_step < n_warmup:
            # Warm-up phase: Linear increase from 0 to lr
            return max_lr * ((current_step + 1) / n_warmup)
        elif current_step <= max_iters:
            # Decay phase: Linear decrease from lr to 0.1 * lr
            return max_lr - (max_lr - min_percent * max_lr) * (
                (current_step - n_warmup) / (max_iters - n_warmup)
            )
        else:
            # Constant learning rate after max_iters
            return min_percent * max_lr
    return get_lr

def masked_mean(x, mask, dim=None):
    return (x * mask).sum(dim=dim) / mask.sum(dim=dim)

def try_auto_cast(device):
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type=device, dtype=torch.bfloat16)
    return nullcontext()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_logging(config):
    """ monotonous bookkeeping """
    work_dir = config.system.work_dir
    # create the work directory if it doesn't already exist
    os.makedirs(work_dir, exist_ok=True)
    # log the args (if any)
    with open(os.path.join(work_dir, 'args.txt'), 'w') as f:
        f.write(' '.join(sys.argv))
    # log the config itself
    with open(os.path.join(work_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(config.to_dict(), indent=4))

class CfgNode:
    """ a lightweight configuration class inspired by yacs """
    # TODO: convert to subclass from a dict like in yacs?
    # TODO: implement freezing to prevent shooting of own foot
    # TODO: additional existence/override checks when reading/writing params?

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent):
        """ need to have a helper to support nested indentation for pretty printing """
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [' ' * (indent * 4) + p for p in parts]
        return "".join(parts)

    def to_dict(self):
        """ return a dict representation of the config """
        return { k: v.to_dict() if isinstance(v, CfgNode) else v for k, v in self.__dict__.items() }

    def merge_from_dict(self, d):
        self.__dict__.update(d)

    def merge_from_args(self, args):
        """
        update the configuration from a list of strings that is expected
        to come from the command line, i.e. sys.argv[1:].

        The arguments are expected to be in the form of `--arg=value`, and
        the arg can use . to denote nested sub-attributes. Example:

        --model.n_layer=10 --trainer.batch_size=32
        """
        for arg in args:

            keyval = arg.split('=')
            assert len(keyval) == 2, "expecting each override arg to be of form --arg=value, got %s" % arg
            key, val = keyval # unpack

            # first translate val into a python object
            try:
                val = literal_eval(val)
                """
                need some explanation here.
                - if val is simply a string, literal_eval will throw a ValueError
                - if val represents a thing (like an 3, 3.14, [1,2,3], False, None, etc.) it will get created
                """
            except ValueError:
                pass

            # find the appropriate object to insert the attribute into
            assert key[:2] == '--'
            key = key[2:] # strip the '--'
            keys = key.split('.')
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)
            leaf_key = keys[-1]

            # ensure that this attribute exists
            assert hasattr(obj, leaf_key), f"{key} is not an attribute that exists in the config"

            # overwrite the attribute
            print("command line overwriting config attribute %s with %s" % (key, val))
            setattr(obj, leaf_key, val)


class TweetDataset(Dataset):
    # Character level tweet dataset: https://huggingface.co/datasets/mteb/tweet_sentiment_extraction
    def __init__(self, block_size, split="train", label=None, tokenizer=None):
        assert split in ["train", "test"]
        self.block_size = block_size

        eot = "⏎"
        def chunk_examples(examples):
            chunks = [(eot+text+eot) for text, lbl in zip(examples['text'], examples['label_text']) if len(text) > 0 and (label is None or lbl == label)]
            return {"content": chunks}

        dataset = datasets.load_dataset("mteb/tweet_sentiment_extraction", split=split)
        self.dataset = dataset.map(chunk_examples, batched=True, remove_columns=dataset.column_names)

        if tokenizer is None:
            token_text = "".join(row["content"] for row in self.dataset)
            self.tokenizer = CharTokenizer(token_text)
        else:
            self.tokenizer = tokenizer

    def get_vocab_size(self):
        return self.tokenizer.vocab_size

    def get_block_size(self):
        return self.block_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        chunk = self.dataset[idx]["content"]
        toks = self.tokenizer(chunk)
        assert len(toks) > 0
        if len(toks) >= self.block_size + 1:
            toks = toks[:self.block_size + 1]
        else:
            pad = torch.full((self.block_size + 1,), self.tokenizer.pad_token, dtype=torch.long)
            pad[:len(toks)] = toks
            toks = pad

        x = toks[:-1]
        y = toks[1:].clone()
        y[y == self.tokenizer.pad_token] = -1

        # Our mask is true for padding/unused tokens
        # to match the causal masking inside the minGPT model
        attn_mask = x == self.tokenizer.pad_token
        return x, y, attn_mask

