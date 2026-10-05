import os
import re
import random
import torch
import pandas as pd
import math
from datasets import Dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from trl.trainer.utils import selective_log_softmax, entropy_from_logits
from torch.utils.tensorboard import SummaryWriter
import json

from transformers import GenerationConfig

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

STAGES = ["stimulus","primary_appraisal","secondary_appraisal","reaction","mental_state"]
# -----------------------------
# System prompt (with structure)
# -----------------------------
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant solves it. The Assistant must explicitly think through "
    "the reasoning process before giving the final answer.\n"
    "The reasoning process should strictly follow the stages below:\n"
    "1. <stimulus> Identify the key event, situation, or object described by the user without interpretation. </stimulus>\n"
    "2. <primary_appraisal> Assess the personal relevance and potential impact of the stimulus. </primary_appraisal>\n"
    "3. <secondary_appraisal> Evaluate available resources and options for coping with the situation. </secondary_appraisal>\n"
    "4. <reaction> Describe the likely affective and behavioral responses. </reaction>\n"
    "5. <mental_state> Conclude the probable mental health condition or state. </mental_state>\n"
    "The reasoning process and answer must be enclosed within <think> </think> and <answer> </answer> tags respectively, i.e.:"
    "<think>"
    "<stimulus> stimulus here </stimulus>"
    "<primary_appraisal> primary appraisal here </primary_appraisal>"
    "<secondary_appraisal> secondary appraisal here </secondary_appraisal>"
    "<reaction> reaction here </reaction>"
    "<mental_state> mental state here </mental_state>"
    "</think>"
    "<answer> answer here </answer>"
)
# ---------------------------------
# Stage-wise entropy weights (β_i)
# ---------------------------------
SER_M, SER_TAU = 0.06, 3.5
beta_schedule = {
    stage: SER_M * math.tanh(t - SER_TAU)
    for t, stage in enumerate(STAGES, start=1)
}
# -----------------
# Data prep (clean)
# -----------------
# def make_conversation(example):
#     return {
#         "prompt": [
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user", "content": example["mental_prompt"] + example["text"]},
#             # {"role": "assistant", "content": "<think>\n<stimulus>"}
#         ],
#         "answer": example["label"]
#     }
def make_conversation(example):
    user_text = example["mental_prompt"] + example["text"]

    prompt_str = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n<stimulus>"
    )
    
    return {
        "prompt": prompt_str,
        "answer": str(example["label"])
    }


def build_stage_masks_via_offsets(tokenizer, completion_ids, completion_mask):
    """
    Build per-stage masks over COMPLETION tokens using text->token offset mapping.
    Args:
      completion_ids:  (B, Tc) LongTensor
      completion_mask: (B, Tc) LongTensor (1 for valid tokens, 0 for padding)
    Returns:
      dict {stage: (B, Tc) float32 mask}
    """
    B, Tc = completion_ids.shape
    device = completion_ids.device
    masks = {stage: torch.zeros((B, Tc), dtype=torch.float32, device=device) for stage in STAGES}

    valid_lens = completion_mask.sum(dim=-1).to(torch.int64).tolist()

    for b in range(B):
        L = valid_lens[b]
        if L <= 0:
            continue

        # 1) Decode exactly the valid completion tokens
        row_ids = completion_ids[b, :L].tolist()
        text = tokenizer.decode(row_ids, skip_special_tokens=True)

        # 2) Find character spans for each stage using regex
        for stage in STAGES:
            open_pat  = f"<{stage}>"
            close_pat = f"</{stage}>"
            m_open  = re.search(re.escape(open_pat), text)
            m_close = re.search(re.escape(close_pat), text)

            if stage == "stimulus" and not m_open:

                if m_close:
                    char_start = 0
                    char_end   = m_close.start()
                else:
                    continue
            else:

                if not (m_open and m_close and m_open.end() <= m_close.start()):
                    continue
                char_start = m_open.end()
                char_end   = m_close.start()
            # ----------------------------
            # if not (m_open and m_close and m_open.end() <= m_close.start()):
            #     continue
            # char_start = m_open.end()
            # char_end   = m_close.start()

            # 3) Re-tokenize this SAME text with offsets to map chars -> token indices
            enc = tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
                return_tensors=None,
            )
            # enc["input_ids"] should now match tokenization of `text`
            # But we need to align to the ORIGINAL ids we sliced by mask.
            # Since both come from `text` we just created, lengths should match.
            offsets = enc["offset_mapping"]  # list of (start_char, end_char)
            # Sanity: some fast tokenizers put (0, 0) for specials—skip those.
            for t, (cs, ce) in enumerate(offsets[:L]):
                if cs is None or ce is None:
                    continue
                # Mark tokens whose character span lies fully within the stage content span
                if cs >= char_start and ce <= char_end:
                    masks[stage][b, t] = 1.0
        tokens = tokenizer.convert_ids_to_tokens(row_ids)
        # for stage in STAGES:
        #     active_tokens = [tokens[i] for i, val in enumerate(masks[stage][b, :L]) if val > 0]
        #     if active_tokens:
        #         content = tokenizer.convert_tokens_to_string(active_tokens)
        #         print(f"[{stage}] -> {content}")
        # print(text)
        # exit()
    return masks

# -------------------
# Rewards (strict + acc)
# -------------------
def format_reward(completions, **kwargs):
    """
    Strict validator:
      1) Exactly one <think>...</think> and one <answer>...</answer>
      2) All five stages once, in order:
         <stimulus> -> <primary_appraisal> -> <secondary_appraisal> -> <reaction> -> <mental_state>
      3) Non-empty content inside each stage and inside <answer>.
    """
    rewards = []
    for completion in completions:
        # content = completion[0]["content"]
        content = completion
        if not content.startswith("<think>"):
            content = "<think>\n<stimulus>" + content
        # print(content)
        # exit()
        # print(content)
        # print("分隔符")
        # exit()
        ok = True
        # print(content)
        # 1) outer tags exactly once
        if content.count("<think>") != 1 or content.count("</think>") != 1:
            ok = False
        if content.count("<answer>") != 1 or content.count("</answer>") != 1:
            ok = False

        # order: <think>...</think> then <answer>...</answer>
        if ok:
            t0 = content.find("<think>")
            t1 = content.find("</think>")
            a0 = content.find("<answer>")
            a1 = content.find("</answer>")
            if not (0 <= t0 < t1 < a0 < a1):
                ok = False
        # print(1,ok)
        # 2) stages ordered, once, inside <think>, non-empty
        if ok:
            prev_end = -1
            for tag in STAGES:
                ot = f"<{tag}>"; ct = f"</{tag}>"
                if content.count(ot) != 1 or content.count(ct) != 1:
                    ok = False; break
                s = content.find(ot); e = content.find(ct)
                if s == -1 or e == -1 or not (s < e): ok = False; break
                if not (t0 <= s and e <= t1): ok = False; break
                if s < prev_end: ok = False; break
                prev_end = e
                inner = content[s+len(ot):e].strip()
                if len(inner) == 0: ok = False; break
        # print(2,ok)
        # 3) non-empty answer
        if ok:
            m = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
            ans = m.group(1).strip() if m else ""
            if len(ans) == 0: ok = False
        # print(3,ok)
        # if ok:
        #     print(content)
        #     exit()
        # rewards.append(1.0 if ok else -1.0)
        rewards.append(0.5 if ok else -0.5)
    return rewards

def accuracy_reward(completions, **kwargs):
    gt = kwargs['answer']
    dataset_ids = kwargs['dataset_id'] 
    
    rewards = []
    for completion, expected, ds_id in zip(completions, gt, dataset_ids):
        content = completion
        m = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        ans = m.group(1).strip() if m else ""

        reward_scale = W_COMBINED.get(ds_id, {}).get(str(expected), 1.0)
        # print(ds_id,expected,reward_scale)
        # exit()

        # print("expected: ",expected)
        # print("ans: ",ans)
        # if ans == "":
        #     print(content)
        if re.match(r'^' + re.escape(expected) + r'(?!\w)', ans, re.IGNORECASE):
            rewards.append(reward_scale)
            # print("answer_matched")
        else:
            rewards.append(-reward_scale)
            # print("answer_unmatched")
            
    return rewards

# ---------------------------------------------------------
# Dynamic-only stage-aware GRPO (uses entropy iff available)
# ---------------------------------------------------------
class StageAwareGRPODynamicOnlyTrainer(GRPOTrainer):
    def __init__(self, *args, processing_class=None, **kwargs):
        super().__init__(*args, processing_class=processing_class, **kwargs)
        self.processing_class = processing_class  # tokenizer / processor
        # cache tag token ids once
        self._tag_token_ids = {}
        for tag in ["stimulus","primary_appraisal","secondary_appraisal","reaction","mental_state"]:
            self._tag_token_ids[f"<{tag}>"]  = self.processing_class(f"<{tag}>", add_special_tokens=False)["input_ids"]
            self._tag_token_ids[f"</{tag}>"] = self.processing_class(f"</{tag}>", add_special_tokens=False)["input_ids"]

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        **kwargs,
    ):

        if any(value is not None for value in kwargs.values()):
            raise ValueError("This override supports text-only training.")

        chunk_size = batch_size or input_ids.size(0)
        logps_parts = []
        entropy_parts = []

        for start in range(0, input_ids.size(0), chunk_size):
            ids = input_ids[start:start + chunk_size]
            mask = attention_mask[start:start + chunk_size]

            forward_args = {
                "input_ids": ids,
                "attention_mask": mask,
            }


            if "logits_to_keep" in self.model_kwarg_keys:
                forward_args["logits_to_keep"] = logits_to_keep + 1

            logits = model(**forward_args).logits


            logits = logits[:, :-1, :]
            logits = logits[:, -logits_to_keep:, :]


            logits = logits / self.temperature
            targets = ids[:, -logits_to_keep:]

            logps_parts.append(
                selective_log_softmax(logits, targets)
            )

            if compute_entropy:

                entropy_parts.append(
                    entropy_from_logits(logits.float())
                )

        logps = torch.cat(logps_parts, dim=0)
        entropies = (
            torch.cat(entropy_parts, dim=0)
            if compute_entropy
            else None
        )
        return logps, entropies
    # ---- override TRL's _compute_loss to add stage-wise entropy, reusing its logits/entropies
    def _compute_loss(self, model, inputs):
        # === BEGIN: original TRL code (unaltered) ===
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, completion_mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        advantages = inputs["advantages"]
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(f"Unknown importance sampling level: {self.importance_sampling_level}.")

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        # === END: original TRL code (unaltered) ===

        # === NEW: stage-wise entropy bonus on completion tokens ===
        # entropies shape: (B, Tc), same as completion_mask
        stage_masks = build_stage_masks_via_offsets(self.processing_class, completion_ids, completion_mask)

        stage_bonus = 0.0
        for stage, stage_beta in beta_schedule.items():
            m = stage_masks[stage]  # (B, Tc), float
            # masked mean entropy per sample; clamp denom to avoid NaN if stage missing
            denom = m.sum(dim=-1).clamp(min=1.0)          # (B,)
            stage_ent = ((entropies * m).sum(dim=-1) / denom)  # (B,)
            # print(stage_beta * stage_ent.mean())
            stage_bonus = stage_bonus + stage_beta * stage_ent.mean()
        # print("entropies.requires_grad =", entropies.requires_grad)
        # print("stage_bonus.grad_fn =", stage_bonus.grad_fn)
        print(loss, stage_bonus)
        loss = loss + stage_bonus
        # print(loss, stage_bonus)

        # === keep TRL metrics unchanged ===
        mode = "train" if self.model.training else "eval"
        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:
                return x.mean()
            else:
                return (x * completion_mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(torch.nan_to_num(gathered_low_clip, nan=1e9).min().item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(torch.nan_to_num(gathered_high_clip, nan=-1e9).max().item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())

        return loss

# -----------------------------
# Pretty console logger (optional)
# -----------------------------
STAGE_COLORS = {
    "stimulus": "\033[94m",
    "primary_appraisal": "\033[92m",
    "secondary_appraisal": "\033[96m",
    "reaction": "\033[93m",
    "mental_state": "\033[91m"
}
RESET_COLOR = "\033[0m"

def highlight_stages(text):
    for stage, color in STAGE_COLORS.items():
        text = text.replace(f"<{stage}>", f"{color}<{stage}>") \
                   .replace(f"</{stage}>", f"</{stage}>{RESET_COLOR}")
    return text

def log_generated_examples(model, processing_class, dataset, writer, num_samples=3, step=0):
    model.eval()
    samples = random.sample(list(dataset), min(num_samples, len(dataset)))
    for i, sample in enumerate(samples):
        print([sample["prompt"]])
        inputs = processing_class([sample["prompt"]], return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.85)
        gen = processing_class.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\n--- Example {i+1} ---")
        print("Prompt:", sample["prompt"])
        print("Generated:\n", highlight_stages(gen))
        writer.add_text(f"example_{i+1}",
                        f"**Prompt:**\n{sample['prompt']}\n\n**Generated:**\n{gen}",
                        step)

# -----------------------------
# Model / tokenizer (processing_class)
# -----------------------------
# Pick ONE:
model_name = "/root/autodl-tmp/Qwen/Qwen3-8B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    # device_map="auto"   # remove for torchrun DDP if you want full-replica training
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
processing_class = tokenizer  # for HF "processing_class" API


# -----------------------------
# Build datasets & Calculate Balanced Weights
# -----------------------------
root_folder = "/root/Mental-Entropy/dataset_new"
train_dataset = None

dataset_stats = {} 

for dataset_name in os.listdir(root_folder):
    dpath = os.path.join(root_folder, dataset_name)
    if not os.path.isdir(dpath):
        continue
    tr = os.path.join(dpath, "train_clean.csv")
    if not os.path.exists(tr):
        continue

    print(f"Processing {dataset_name} ...")
    train_df = pd.read_csv(tr)[["text", "label", "mental_prompt"]]

    class_counts = train_df['label'].astype(str).value_counts().to_dict()
    dataset_stats[dataset_name] = class_counts

    train_df["dataset_id"] = dataset_name

    dtr = Dataset.from_pandas(train_df).map(make_conversation, remove_columns=['text', 'label', 'mental_prompt'])

    train_dataset = dtr if train_dataset is None else concatenate_datasets([train_dataset, dtr])

print(f"Combined train dataset: {len(train_dataset)} samples.")

# -----------------------------
# Compute w_d and w_c dynamically
# -----------------------------
W_D = {}
W_C = {}

if dataset_stats:

    D = len(dataset_stats)
    n_d = {d: sum(counts.values()) for d, counts in dataset_stats.items()}
    sum_inv_n_d = sum(1.0 / count for count in n_d.values())
    avg_inv_n_d = sum_inv_n_d / D
    W_D = {d: (1.0 / count) / avg_inv_n_d for d, count in n_d.items()}


    for d, counts in dataset_stats.items():
        C = len(counts)
        sum_inv_n_j = sum(1.0 / count for count in counts.values())
        avg_inv_n_j = sum_inv_n_j / C

        W_C[d] = {c: (1.0 / count) / avg_inv_n_j for c, count in counts.items()}
W_COMBINED = {}
for d in dataset_stats.keys():
    W_COMBINED[d] = {}
    curr_w_d = W_D.get(d, 1.0)
    for c in W_C[d].keys():
        curr_w_c = W_C[d].get(c, 1.0)

        W_COMBINED[d][c] = math.sqrt(curr_w_c * curr_w_d)

print("\nComputed Weights combined (W_COMBINED):", W_COMBINED)
# exit()

# -----------------------------
# GRPO config
# -----------------------------
training_args = GRPOConfig(
    ddp_find_unused_parameters=False,
    output_dir="/root/tf-logs/qwen3_mental_r1",
    per_device_train_batch_size=4,     # tune per GPU;
    temperature=0.85,
    top_p=0.95,
    learning_rate=5e-6,
    remove_unused_columns=False,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    bf16=True,
    max_completion_length=512,
    num_generations=4,
    max_prompt_length=512,
    report_to=["tensorboard"],
    logging_steps=10,
    # save_steps=10000,
    save_strategy="no",          # <- no automatic checkpoints
    beta=0.01,
    deepspeed="deepspeed_zero2.json"    # <— key line
)

writer = SummaryWriter(log_dir=os.path.join(training_args.output_dir, "logs"))

# -----------------------------
# Trainer (dynamic-only + processing_class)
# -----------------------------
trainer = StageAwareGRPODynamicOnlyTrainer(
    model=model,
    processing_class=processing_class,              # <— no deprecation warning
    reward_funcs=[format_reward, accuracy_reward],
    # reward_weights=[0.5, 1.0],
    args=training_args,
    train_dataset=train_dataset
)

# log_generated_examples(model, processing_class, train_dataset, writer, num_samples=3, step=0)
for epoch in range(int(training_args.num_train_epochs)):
    trainer.train(resume_from_checkpoint=False)
    # log_generated_examples(model, processing_class, train_dataset, writer, num_samples=3, step=epoch)

writer.close()
trainer.save_model(training_args.output_dir)
