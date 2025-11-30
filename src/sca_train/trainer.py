import torch
from transformers import Trainer

from . import logger
from .config import SCATrainingConfig


class QwenTrainer(Trainer):
    def __init__(self, config: SCATrainingConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        logger.debug(self.config, f"Custom compute_loss called with inputs keys: {list(inputs.keys())}")
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        inputs_for_model = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

        if "input_features" in inputs:
            logger.debug(self.config, f"Adding input_features to model inputs.")
            inputs_for_model["input_features"] = inputs["input_features"]
        if "pixel_values" in inputs:
            logger.debug(self.config, f"Adding pixel_values to model inputs.")
            inputs_for_model["pixel_values"] = inputs["pixel_values"]

        logger.debug(self.config, f"Forward pass with inputs keys: {list(inputs_for_model.keys())}")
        outputs = model(**inputs_for_model)
        logger.debug(self.config, f"Model forward pass completed.")

        logger.debug(self.config, f"Outputs keys: {outputs.keys() if hasattr(outputs, 'keys') else 'N/A'}")
        if hasattr(outputs, "loss") and outputs.loss is not None:
            loss = outputs.loss

        elif labels is not None:
            logger.debug(self.config, f"Calculating loss manually using CrossEntropyLoss.")
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        else:
            raise ValueError("Model output did not contain loss, and no labels were provided.")

        logger.debug(self.config, f"Computed loss: {loss.item()}")
        return (loss, outputs) if return_outputs else loss