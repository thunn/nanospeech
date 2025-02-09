from functools import partial

from torch.optim import AdamW

from datasets import load_dataset

from accelerate import DataLoaderConfiguration

from nanospeech.nanospeech_torch import (
    DurationTransformer,
    DurationPredictor,
    list_str_to_tensor,
    list_str_to_vocab_tensor,
    SAMPLES_PER_SECOND,
)
from nanospeech.trainer_torch import DurationTrainer


def train():
    # vocab-based tokenizer

    with open("vocab.txt", "r") as f:
        vocab = {v: i for i, v in enumerate(f.read().splitlines())}
    tokenizer = partial(list_str_to_vocab_tensor, vocab=vocab)
    text_num_embeds = len(vocab)

    # or use a utf-8 byte tokenizer

    # tokenizer = list_str_to_tensor
    # text_num_embeds = 256

    model = DurationPredictor(
        transformer=DurationTransformer(
            dim=512,
            depth=8,
            heads=8,
            text_dim=512,
            ff_mult=2,
            conv_layers=4,
            text_num_embeds=text_num_embeds,
        ),
        tokenizer=tokenizer,
    )
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = AdamW(model.parameters(), lr=1e-4)

    trainer = DurationTrainer(
        model,
        optimizer,
        num_warmup_steps=1000,
        accelerate_kwargs={
            "mixed_precision": "bf16",
            # "log_with": "wandb",  # if you want to enable logging with wandb
            # "dataloader_config": DataLoaderConfiguration(dispatch_batches=False)  # optional DDP configuration for multi-GPU training
        },
    )

    dataset = load_dataset("amphion/Emilia-Dataset", split="train", streaming=True)

    print("Training...")

    total_steps = 1_000_000
    batch_size = 128
    max_duration_sec = 10
    max_duration = int(max_duration_sec * SAMPLES_PER_SECOND)
    max_batch_frames = int(batch_size * max_duration)

    trainer.train(
        dataset,
        total_steps,
        batch_size=batch_size,
        max_batch_frames=max_batch_frames,
        max_duration=max_duration,
        num_workers=8,
        save_step=10_000,
        restore_step=None,
    )


train()
