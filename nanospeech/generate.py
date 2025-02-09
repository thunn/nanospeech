import argparse
from collections import deque
import datetime
import importlib.util
from pathlib import Path
import re
import sys
from threading import Event, Lock
from typing import Literal, Optional

import numpy as np

import sounddevice as sd
import soundfile as sf

from tqdm import tqdm

if importlib.util.find_spec("mlx") is not None:
    from nanospeech.nanospeech_mlx import Nanospeech
elif importlib.util.find_spec("torch") is not None:
    from nanospeech.nanospeech_torch import Nanospeech


SAMPLE_RATE = 24_000


# utilities


def split_sentences(text):
    sentence_endings = re.compile(r"([.!?;:])")
    sentences = sentence_endings.split(text)
    sentences = [sentences[i] + sentences[i + 1] for i in range(0, len(sentences) - 1, 2)]
    return [sentence.strip() for sentence in sentences if sentence.strip()]


# playback


class AudioPlayer:
    def __init__(self, sample_rate=SAMPLE_RATE, buffer_size=2048):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.audio_buffer = deque()
        self.buffer_lock = Lock()
        self.playing = False
        self.drain_event = Event()

    def callback(self, outdata, frames, time, status):
        with self.buffer_lock:
            if len(self.audio_buffer) > 0:
                available = min(frames, len(self.audio_buffer[0]))
                chunk = self.audio_buffer[0][:available].copy()
                self.audio_buffer[0] = self.audio_buffer[0][available:]

                if len(self.audio_buffer[0]) == 0:
                    self.audio_buffer.popleft()
                    if len(self.audio_buffer) == 0:
                        self.drain_event.set()

                outdata[:, 0] = np.zeros(frames)
                outdata[:available, 0] = chunk
            else:
                outdata[:, 0] = np.zeros(frames)
                self.drain_event.set()

    def play(self):
        if not self.playing:
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self.callback,
                blocksize=self.buffer_size,
            )
            self.stream.start()
            self.playing = True
            self.drain_event.clear()

    def queue_audio(self, samples):
        self.drain_event.clear()

        with self.buffer_lock:
            self.audio_buffer.append(np.array(samples))
        if not self.playing:
            self.play()

    def wait_for_drain(self):
        return self.drain_event.wait()

    def stop(self):
        if self.playing:
            self.wait_for_drain()
            sd.sleep(100)

            self.stream.stop()
            self.stream.close()
            self.playing = False


# generation


def generate_one(
    model: Nanospeech,
    text: str,
    ref_audio: np.ndarray,
    ref_audio_text: str,
    steps: int = 8,
    method: Literal["euler", "midpoint"] = "rk4",
    cfg_strength: float = 2.0,
    sway_sampling_coef: float = -1.0,
    speed: float = 1.0,
    seed: Optional[int] = None,
    player: Optional[AudioPlayer] = None,
):
    generation_text = [ref_audio_text + " " + text]

    start_date = datetime.datetime.now()

    wave, _ = model.sample(
        np.expand_dims(ref_audio, axis=0),
        text=generation_text,
        steps=steps,
        ode_method=method,
        speed=speed,
        cfg_strength=cfg_strength,
        sway_sampling_coef=sway_sampling_coef,
        seed=seed,
    )

    generated_duration = wave.shape[0] / SAMPLE_RATE
    print(f"Generated {generated_duration:.2f}s of audio in {datetime.datetime.now() - start_date}.")

    if player is not None:
        player.queue_audio(wave)

    return wave


def generate(
    text: str,
    model_name: str = "lucasnewman/nanospeech",
    voice: Literal["celeste", "luna", "nash", "orion", "rhea"] = "celeste",
    ref_audio_path: Optional[str] = None,
    ref_audio_text: Optional[str] = None,
    steps: int = 8,
    method: Literal["euler", "midpoint"] = "rk4",
    cfg_strength: float = 2.0,
    sway_sampling_coef: float = -1.0,
    speed: float = 1.0,
    seed: Optional[int] = None,
    output_path: Optional[str] = None,
):
    player = AudioPlayer(sample_rate=SAMPLE_RATE) if output_path is None else None

    nanospeech = Nanospeech.from_pretrained(model_name)

    if ref_audio_path is None or ref_audio_text is None:
        d = Path(sys.modules["nanospeech"].__file__).parent
        ref_audio_text = (d / f"voices/{voice}.txt").read_text().strip()
        audio_path = d / f"voices/{voice}.wav"
        audio, _ = sf.read(audio_path)
    else:
        # load reference audio
        audio, sr = sf.read(ref_audio_path)
        if sr != SAMPLE_RATE:
            raise ValueError("Reference audio must but mono with a sample rate of 24kHz")

    ref_audio_duration = audio.shape[0] / SAMPLE_RATE
    print(f"Got reference audio with duration: {ref_audio_duration:.2f} seconds")

    sentences = split_sentences(text)
    is_single_generation = len(sentences) <= 1
    
    if is_single_generation:
        wave = generate_one(
            model=nanospeech,
            text=text,
            ref_audio=audio,
            ref_audio_text=ref_audio_text,
            steps=steps,
            method=method,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            seed=seed,
            player=player,
        )

        if output_path is not None:
            sf.write(output_path, wave, SAMPLE_RATE)

        if player is not None:
            player.stop()
    else:
        start_date = datetime.datetime.now()

        output = []

        for sentence_text in tqdm(sentences):
            wave = generate_one(
                model=nanospeech,
                text=sentence_text,
                ref_audio=audio,
                ref_audio_text=ref_audio_text,
                steps=steps,
                method=method,
                speed=speed,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                seed=seed,
                player=player,
            )

            output.append(wave)

        if player is not None:
            player.stop()

        wave = np.concatenate(output, axis=0)

        generated_duration = wave.shape[0] / SAMPLE_RATE
        print(f"Generated {generated_duration:.2f}s of audio in {datetime.datetime.now() - start_date}.")

        if output_path is not None:
            sf.write(output_path, np.array(wave), SAMPLE_RATE)

        if player is not None:
            player.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate speech from text")

    parser.add_argument(
        "--model",
        type=str,
        default="lucasnewman/nanospeech",
        help="Name of the model to use",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text to generate speech from (leave blank to input via stdin)",
    )
    
    parser.add_argument(
        "--voice",
        type=str,
        default="celeste",
        choices=["celeste", "luna", "nash", "orion", "rhea"],
        help="Voice to use for the reference audio",
    )
    
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help="Path to the reference audio file",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default=None,
        help="Text spoken in the reference audio",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the generated audio output",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=8,
        help="Number of steps to take when sampling the neural ODE",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="rk4",
        choices=["euler", "midpoint", "rk4"],
        help="Method to use for sampling the neural ODE",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=2.0,
        help="Strength of classifer free guidance",
    )
    parser.add_argument(
        "--sway-coef",
        type=float,
        default=-1.0,
        help="Coefficient for sway sampling",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speed factor for the duration heuristic",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for noise generation",
    )

    args = parser.parse_args()

    if args.text is None:
        if not sys.stdin.isatty():
            args.text = sys.stdin.read().strip()
        else:
            print("Please enter the text to generate:")
            args.text = input("> ").strip()

    generate(
        text=args.text,
        model_name=args.model,
        voice=args.voice,
        ref_audio_path=args.ref_audio,
        ref_audio_text=args.ref_text,
        steps=args.steps,
        method=args.method,
        cfg_strength=args.cfg,
        sway_sampling_coef=args.sway_coef,
        speed=args.speed,
        seed=args.seed,
        output_path=args.output,
    )
