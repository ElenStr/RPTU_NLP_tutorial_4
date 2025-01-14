import os
import time

import datasets
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.bpe import BPETokenizer
from mingpt.logger import Logger
from mingpt.model import GPT
from mingpt.rewards import RewardModel
from mingpt.trainer import CN
from mingpt.utils import set_seed, try_auto_cast


class RewardModelSummarize(Dataset):
    def __init__(self, split: str, block_size: int = 1024):
        self.split = split
        self.tokenizer = BPETokenizer()
        self.voc = 50257
        self.block_size = block_size
        ds = datasets.load_dataset("CarperAI/openai_summarize_comparisons", split=split)
        def drop_long_examples(examples):
            prompts = []
            all_chosen = []
            all_rejected = []
            for prompt, chosen, rejected in zip(examples['prompt'], examples['chosen'], examples['rejected']):
                prompt_len = self.tokenizer(prompt).size(1)
                chosen_len = self.tokenizer(chosen).size(1)
                rejected_len = self.tokenizer(rejected).size(1)
                if prompt_len + chosen_len <= block_size + 1 and prompt_len + rejected_len <= block_size + 1:
                    prompts.append(prompt)
                    all_chosen.append(chosen)
                    all_rejected.append(rejected)

            return {"prompt": prompts, "chosen": all_chosen, "rejected": all_rejected}

        self.ds = ds.map(drop_long_examples, batched=True, remove_columns=ds.column_names, num_proc=os.cpu_count())

    def __len__(self):
        return len(self.ds)

    def get_vocab_size(self):
        return self.voc

    def get_block_size(self):
        return self.block_size

    def pad_toks(self, toks):
        # The padding here differs from the SFT since we don't need the LM targets
        mask = torch.full((self.block_size,), False, dtype=bool)
        if len(toks) >= self.block_size:
            toks = toks[-self.block_size:]
        else:
            pad = torch.full((self.block_size,), self.tokenizer.eot_token, dtype=torch.long)
            pad[:len(toks)] = toks

            # include a final eot token to predict
            mask[len(toks) + 1:] = True
            toks = pad

        return toks, mask

    def __getitem__(self, idx):
        row = self.ds[idx]
        prompt, chosen, rejected = row['prompt'], row['chosen'], row['rejected']
        prompt = self.tokenizer(prompt).squeeze(0)
        chosen = self.tokenizer(chosen).squeeze(0)
        rejected = self.tokenizer(rejected).squeeze(0)

        chosen, cmask = self.pad_toks(torch.cat((prompt, chosen)))
        rejected, rmask = self.pad_toks(torch.cat((prompt, rejected)))

        return {
            "pos_toks": chosen,
            "pos_mask": cmask,
            "neg_toks": rejected,
            "neg_mask": rmask,
        }


@torch.no_grad()
def evaluate(model, config, ds, iters=32):
    train_loader = DataLoader(
        ds,
        shuffle=False,
        batch_size=config.batch_size * 2,
        drop_last=True
    )
    total_loss = 0
    total_acc = 0
    i = 0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss, acc = model(
            batch["neg_toks"],
            attn_mask=batch["neg_mask"],
            positive_tokens=batch["pos_toks"],
            positive_mask=batch["pos_mask"]
        )
        total_loss += loss.item()
        total_acc += acc.item()
        i += 1
        if i == iters:
            break
    return total_loss / i, total_acc / i


@torch.no_grad()
def set_reward_bias(model, config, ds, iters=128, device='gpu'):
    train_loader = DataLoader(
        ds,
        shuffle=False,
        batch_size=config.batch_size * 2,
        drop_last=True
    )
    all_rewards = []
    i = 0
    for batch in train_loader:
        x = torch.cat((batch["pos_toks"], batch["neg_toks"]))
        mask = torch.cat((batch["pos_mask"], batch["neg_mask"]))
        x, mask = [v.to(device) for v in (x, mask)]
        model.to(device)
        rewards = model(x, attn_mask=mask)
        all_rewards.append(rewards)
        i += 1
        if i == iters:
            break

    reward_bias = torch.mean(torch.cat(all_rewards))
    model.prediction_head.bias.sub_(reward_bias)
    print("Set reward bias to", model.prediction_head.bias.item())



