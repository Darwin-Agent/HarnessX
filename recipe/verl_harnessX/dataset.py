"""
Text-only dataset adapter for veRL GRPO training.
"""

import logging

import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask

from verl_harnessX.prompt import SYSTEM_PROMPT, PROMPT_TEXT_ONLY

logger = logging.getLogger(__name__)


class TextOnlyDataset(RLHFDataset):
    def maybe_filter_out_long_prompts(self, dataframe=None):
        if not self.filter_overlong_prompts or dataframe is None:
            return dataframe
        tokenizer = self.tokenizer
        prompt_key = self.prompt_key
        max_len = self.max_prompt_length
        apply_kwargs = dict(self.apply_chat_template_kwargs)
        apply_kwargs.pop("tokenize", None)
        apply_kwargs.pop("return_dict", None)
        apply_kwargs.pop("return_tensors", None)

        keep = []
        for i in range(len(dataframe)):
            try:
                ids = tokenizer.apply_chat_template(
                    dataframe[i][prompt_key],
                    add_generation_prompt=True,
                    tokenize=True,
                    **apply_kwargs,
                )
                keep.append(len(ids) <= max_len)
            except Exception:
                keep.append(False)

        before = len(dataframe)
        dataframe = dataframe.select([i for i, k in enumerate(keep) if k])
        print(f"Filtered {before} -> {len(dataframe)} prompts (max_len={max_len})")
        return dataframe

    def __getitem__(self, item):
        row_dict: dict = self.dataframe[item]

        question = row_dict.get("question", "")
        if not question:
            prompt_msgs = row_dict[self.prompt_key]
            if isinstance(prompt_msgs, (list, tuple)) and len(prompt_msgs) > 0:
                first = prompt_msgs[0]
                if isinstance(first, dict):
                    question = first.get("content", "")

        row_dict[self.prompt_key] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": PROMPT_TEXT_ONLY + question},
        ]

        messages = self._build_messages(row_dict)

        raw_prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=True,
        )
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} exceeds {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = {}

        row_dict["extra_info"]["question"] = question
        row_dict["extra_info"]["ground_truth"] = row_dict["reward_model"]["ground_truth"]

        row_dict["agent_name"] = "harnessx_agent"
        return row_dict
