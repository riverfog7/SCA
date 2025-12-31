import os


def prepare_model_for_kbit_training(model, use_gradient_checkpointing: bool = True, gradient_checkpointing_kwargs = None):
    if use_gradient_checkpointing:
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": False}
        elif "use_reentrant" not in gradient_checkpointing_kwargs:
            gradient_checkpointing_kwargs["use_reentrant"] = False

        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)

    if hasattr(model, "thinker") and hasattr(model, "talker"):
        thinker_embeds = model.thinker.model.embed_tokens
        talker_embeds = model.talker.model.embed_tokens
        thinker_embeds.register_forward_hook(make_inputs_require_grad)
        talker_embeds.register_forward_hook(make_inputs_require_grad)
    elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        embed_tokens = model.model.embed_tokens
        embed_tokens.register_forward_hook(make_inputs_require_grad)
    else:
        raise AttributeError("Could not locate embed_tokens in model structure.")


def is_fsdp() -> bool:
    return os.environ.get("ACCELERATE_USE_FSDP", "false") == "true"


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))
