import numpy as np
from transformers import Qwen3OmniMoeProcessor


class Qwen3OmniCollator:
    def __init__(self, processor: Qwen3OmniMoeProcessor, mask_instruction: bool = True, max_length: int = 32768):
        self.processor = processor
        self.mask_instruction = mask_instruction
        self.max_length = max_length

        assert hasattr(processor, "tokenizer"), "Processor must have a tokenizer attribute"
        self.pad_token_id = processor.tokenizer.pad_token_id
        self.im_start_id = None

        im_tokens = processor.tokenizer.encode("<|im_start|>", add_special_tokens=False)
        if len(im_tokens) > 0:
            self.im_start_id = im_tokens[0]

    def __call__(self, features):
        # 1. Handle Input Format
        if isinstance(features, dict):
            keys = features.keys()
            features = [dict(zip(keys, vals)) for vals in zip(*features.values())]

        texts = []
        audios = []

        # 2. Extract Data
        for feature in features:
            messages = feature["messages"]
            feature_audio = None

            for msg in messages:
                if msg["role"] == "user":
                    for content in msg["content"]:
                        if content["type"] == "audio":
                            if "audio_waveform" in content:
                                feature_audio = content.pop("audio_waveform")
                                content["audio_url"] = "place_holder"

            if feature_audio is None:
                feature_audio = np.zeros(16000)

            audios.append(feature_audio)

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)

        # 3. Process Batch
        batch = self.processor(
            text=texts,
            audio=audios,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )

        # 4. Create Labels
        labels = batch["input_ids"].clone()

        # Apply Padding Mask
        if self.pad_token_id is not None:
            labels[labels == self.pad_token_id] = -100

        # Apply Instruction Masking
        if self.mask_instruction and self.im_start_id is not None:
            for i in range(len(batch["input_ids"])):
                starts = (batch["input_ids"][i] == self.im_start_id).nonzero(as_tuple=True)[0]
                if len(starts) > 0:
                    last_start = starts[-1]
                    labels[i, :last_start] = -100

        batch["labels"] = labels
        return batch
