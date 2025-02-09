from __future__ import annotations

from functools import partial
import os

import torch

import torch.nn.functional as F

from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader

from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from ema_pytorch import EMA

from nanospeech.nanospeech_torch import DurationPredictor, MelSpec, Nanospeech, exists

HOP_LENGTH = 256


class DynamicBatchDataLoader:
    def __init__(
        self,
        dataset,
        collate_fn,
        batch_size=32,
        max_batch_frames=4096,
        max_duration=None,
        **dataloader_kwargs,
    ):
        self.max_batch_frames = max_batch_frames
        self.max_duration = max_duration
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=self._collate_wrapper,
            **dataloader_kwargs,
        )
        self.collate_fn = collate_fn

    def _collate_wrapper(self, batch):
        return dynamic_batch_collate_fn(batch, self.collate_fn, self.max_batch_frames, self.max_duration)

    def __iter__(self):
        batch_iterator = iter(self.dataloader)
        while True:
            try:
                batch = next(batch_iterator)

                while batch is None:
                    batch = next(batch_iterator)
                yield batch
            except StopIteration:
                break


# collation


def collate_fn(batch, audio=None, tokenizer=None):
    if audio is None:
        audio = [torch.Tensor(item["mp3"]["array"], device="cpu", dtype=torch.float32) for item in batch]

    if audio is None or len(audio) == 0:
        return None

    audio_lengths = torch.LongTensor([item.shape[-1] for item in audio])
    max_audio_length = audio_lengths.amax()

    # round up to the nearest multiple of Xs
    padding_interval = 1 * 24_000
    max_audio_length = (max_audio_length + padding_interval - 1) // padding_interval * padding_interval

    padded_audio = []
    for item in audio:
        padding = (0, max_audio_length - item.size(-1))
        padded_spec = F.pad(item, padding, value=0)
        padded_audio.append(padded_spec)

    padded_audio = torch.stack(padded_audio, dim=0)

    text = [item["json"]["text"] for item in batch]
    text = tokenizer(text)

    return dict(audio=padded_audio, audio_lengths=audio_lengths, text=text)


def dynamic_batch_collate_fn(batch, batch_collate_fn, max_batch_frames, max_duration=None):
    cum_length = 0
    valid_items = []
    audio_tensors = []

    max_duration = max_duration if max_duration is not None else 4096

    for idx, item in enumerate(batch):
        item_audio = torch.tensor(item["mp3"]["array"], device="cpu", dtype=torch.float32)
        item_length = item_audio.shape[-1] // HOP_LENGTH

        if item_length > max_batch_frames or item_length > max_duration:
            continue

        if cum_length + item_length <= max_batch_frames:
            valid_items.append({k: v for k, v in item.items() if k != "mp3"})
            audio_tensors.append(item_audio)
            cum_length += item_length
        else:
            if valid_items:
                break

    if not valid_items:
        print("warning: no valid items in batch")
        return None

    collated_items = batch_collate_fn(valid_items, audio=audio_tensors)

    return collated_items


class NanospeechTrainer:
    def __init__(
        self,
        model: Nanospeech,
        optimizer,
        num_warmup_steps=2_000,
        max_grad_norm=1.0,
        sample_rate=24_000,
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
    ):
        ddp_kwargs = DistributedDataParallelKwargs()

        self.accelerator = Accelerator(
            kwargs_handlers=[ddp_kwargs],
            **accelerate_kwargs,
        )

        self.target_sample_rate = sample_rate

        self.model = model
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.mel_spectrogram = MelSpec()

        self.ema_model = EMA(model, include_online_model=False, **ema_kwargs)

        self.ema_model, self.model, self.optimizer = self.accelerator.prepare(self.ema_model, self.model, self.optimizer)
        self.max_grad_norm = max_grad_norm

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def checkpoint_path(self, step: int):
        return f"nanospeech_{step}.pt"

    def save_checkpoint(self, step, finetune=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict=self.accelerator.unwrap_model(self.optimizer).state_dict(),
                ema_model_state_dict=self.ema_model.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                step=step,
            )

            self.accelerator.save(checkpoint, self.checkpoint_path(step))

    def load_checkpoint(self, step=0):
        if not exists(self.checkpoint_path(step)) or not os.path.exists(self.checkpoint_path(step)):
            return 0

        checkpoint = torch.load(self.checkpoint_path(step), map_location="cpu", weights_only=True)
        self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
        self.accelerator.unwrap_model(self.optimizer).load_state_dict(checkpoint["optimizer_state_dict"])

        if self.is_main:
            self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])

        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint["step"]

    def train(
        self,
        train_dataset,
        total_steps,
        batch_size=12,
        max_batch_frames=4096,
        max_duration=4096,
        num_workers=0,
        restore_step=None,
        save_step=1000,
    ):
        decay_steps = total_steps - self.num_warmup_steps
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=self.num_warmup_steps,
        )
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_steps)
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[self.num_warmup_steps],
        )
        self.scheduler = self.accelerator.prepare(self.scheduler)

        if restore_step is not None:
            start_step = self.load_checkpoint(restore_step)
        else:
            start_step = 0

        global_step = start_step

        hps = {
            "total_steps": total_steps,
            "num_warmup_steps": self.num_warmup_steps,
            "max_grad_norm": self.max_grad_norm,
            "batch_size": batch_size,
            "max_batch_frames": max_batch_frames,
            "max_duration": max_duration,
        }
        self.accelerator.init_trackers("nanospeech", config=hps)

        train_collate_fn = partial(collate_fn, tokenizer=self.model.tokenizer)

        epoch = 0
        total_frames = 0

        while global_step < total_steps:
            self.model.train()

            dataset = train_dataset.shuffle(seed=99 + epoch).filter(lambda x: x["json"]["duration"] <= 10.0)

            train_dataloader = DynamicBatchDataLoader(
                dataset,
                collate_fn=train_collate_fn,
                batch_size=batch_size,
                max_batch_frames=max_batch_frames,
                max_duration=max_duration,
                num_workers=num_workers,
                persistent_workers=True if num_workers > 0 else False,
                pin_memory=True,
                drop_last=True,
            )

            train_dataloader.dataloader = self.accelerator.prepare(train_dataloader.dataloader)

            if self.accelerator.is_local_main_process:
                print(f"Epoch {epoch + 1}, batch size: {batch_size}, max batch tokens: {max_batch_frames}")

            progress_bar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch + 1}",
                unit="",
                disable=not self.accelerator.is_local_main_process,
            )
            epoch_loss = 0.0
            item_count = 0

            for batch in progress_bar:
                if batch is None:
                    continue

                with self.accelerator.accumulate(self.model):
                    text = batch["text"]
                    audio = batch["audio"].to(self.accelerator.device, dtype=torch.bfloat16, non_blocking=True)
                    audio_lengths = batch["audio_lengths"].to(self.accelerator.device)

                    # adjust for mel length
                    audio_lengths = (audio_lengths / HOP_LENGTH).long()

                    with torch.autocast(device_type=self.accelerator.device.type, dtype=torch.bfloat16):
                        loss = self.model(audio, text=text, lens=audio_lengths)

                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                self.accelerator.unwrap_model(self.ema_model).update()

                total_frames += audio_lengths.sum().item()

                if self.accelerator.is_local_main_process:
                    self.accelerator.log(
                        {
                            "loss": loss.item(),
                            "lr": self.scheduler.get_last_lr()[0],
                            "frames": total_frames,
                        },
                        step=global_step,
                    )

                global_step += 1
                epoch_loss += loss.item()
                item_count += 1
                formatted_total_frames = f"{total_frames / 1e6:.2f}M"
                progress_bar.set_postfix(loss=loss.item(), frames=formatted_total_frames)

                if global_step % save_step == 0:
                    self.save_checkpoint(global_step)

                if global_step >= total_steps:
                    break

            epoch_loss /= item_count
            if self.accelerator.is_local_main_process:
                self.accelerator.log({"epoch average loss": epoch_loss}, step=global_step)
                print(f"Epoch {epoch + 1} loss: {epoch_loss}")

            epoch += 1

        self.accelerator.end_training()


class DurationTrainer:
    def __init__(
        self,
        model: DurationPredictor,
        optimizer,
        num_warmup_steps=1_000,
        max_grad_norm=1.0,
        sample_rate=24_000,
        accelerate_kwargs: dict = dict(),
    ):
        ddp_kwargs = DistributedDataParallelKwargs()

        self.accelerator = Accelerator(
            kwargs_handlers=[ddp_kwargs],
            **accelerate_kwargs,
        )

        self.target_sample_rate = sample_rate

        self.model = model
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.mel_spectrogram = MelSpec()

        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        self.max_grad_norm = max_grad_norm

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def checkpoint_path(self, step: int):
        return f"duration_{step}.pt"

    def save_checkpoint(self, step, finetune=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict=self.accelerator.unwrap_model(self.optimizer).state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                step=step,
            )

            self.accelerator.save(checkpoint, self.checkpoint_path(step))

    def load_checkpoint(self, step=0):
        if not exists(self.checkpoint_path(step)) or not os.path.exists(self.checkpoint_path(step)):
            return 0

        checkpoint = torch.load(self.checkpoint_path(step), map_location="cpu", weights_only=True)
        self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
        self.accelerator.unwrap_model(self.optimizer).load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint["step"]

    def train(
        self,
        train_dataset,
        total_steps,
        batch_size=12,
        max_batch_frames=4096,
        max_duration=4096,
        num_workers=0,
        restore_step=None,
        save_step=1000,
    ):
        decay_steps = total_steps - self.num_warmup_steps
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=self.num_warmup_steps,
        )
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_steps)
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[self.num_warmup_steps],
        )
        self.scheduler = self.accelerator.prepare(self.scheduler)

        if restore_step is not None:
            start_step = self.load_checkpoint(restore_step)
        else:
            start_step = 0

        global_step = start_step

        hps = {
            "total_steps": total_steps,
            "num_warmup_steps": self.num_warmup_steps,
            "max_grad_norm": self.max_grad_norm,
            "batch_size": batch_size,
            "max_batch_frames": max_batch_frames,
            "max_duration": max_duration,
        }
        self.accelerator.init_trackers("nanospeech_duration", config=hps)

        train_collate_fn = partial(collate_fn, tokenizer=self.model.tokenizer)

        epoch = 0
        total_frames = 0

        while global_step < total_steps:
            self.model.train()

            dataset = train_dataset.shuffle(seed=99 + epoch).filter(lambda x: x["json"]["duration"] <= 15.0)

            train_dataloader = DynamicBatchDataLoader(
                dataset,
                collate_fn=train_collate_fn,
                batch_size=batch_size,
                max_batch_frames=max_batch_frames,
                max_duration=max_duration,
                num_workers=num_workers,
                persistent_workers=True if num_workers > 0 else False,
                pin_memory=True,
                drop_last=True,
            )

            train_dataloader.dataloader = self.accelerator.prepare(train_dataloader.dataloader)

            if self.accelerator.is_local_main_process:
                print(f"Epoch {epoch + 1}, batch size: {batch_size}, max batch tokens: {max_batch_frames}")

            progress_bar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch + 1}",
                unit="",
                disable=not self.accelerator.is_local_main_process,
            )
            epoch_loss = 0.0
            item_count = 0

            for batch in progress_bar:
                if batch is None:
                    continue

                with self.accelerator.accumulate(self.model):
                    text = batch["text"]
                    audio = batch["audio"].to(self.accelerator.device, dtype=torch.bfloat16, non_blocking=True)
                    audio_lengths = batch["audio_lengths"].to(self.accelerator.device)

                    # adjust for mel length
                    audio_lengths = (audio_lengths / HOP_LENGTH).long()

                    with torch.autocast(device_type=self.accelerator.device.type, dtype=torch.bfloat16):
                        loss = self.model(audio, text=text, lens=audio_lengths, return_loss=True)

                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                total_frames += audio_lengths.sum().item()

                if self.accelerator.is_local_main_process:
                    self.accelerator.log(
                        {"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]},
                        step=global_step,
                    )

                global_step += 1
                epoch_loss += loss.item()
                item_count += 1
                formatted_total_frames = f"{total_frames / 1e6:.2f}M"
                progress_bar.set_postfix(loss=loss.item(), frames=formatted_total_frames)

                if global_step % save_step == 0:
                    self.save_checkpoint(global_step)

                if global_step >= total_steps:
                    break

            epoch_loss /= item_count
            if self.accelerator.is_local_main_process:
                self.accelerator.log({"epoch average loss": epoch_loss}, step=global_step)
                print(f"Epoch {epoch + 1} loss: {epoch_loss}")

            epoch += 1

        self.accelerator.end_training()
