import os 
import torch
import spaces
from glob import glob
from dataclasses import dataclass

import gradio as gr
import soundfile as sf
from nanospeech.nanospeech_torch import Nanospeech
from nanospeech.generate import generate_one, SAMPLE_RATE, split_sentences
import numpy as np
from typing import Optional


PROMPT_DIR = 'nanospeech/voices'

# Note: gradio expects audio as int16, so we need to convert to float32 when loading and convert back when returning

def convert_audio_int16_to_float32(audio: np.ndarray) -> np.ndarray:
    return audio.astype(np.float32) / 32768.0

def convert_audio_float32_to_int16(audio: np.ndarray) -> np.ndarray:
    return (np.clip(audio, -1.0, 1.0) * 32768.0).astype(np.int16)

@dataclass
class VoicePrompt:
    wav_path: str
    text: str

def get_prompt_list(prompt_dir=PROMPT_DIR):
    wav_paths = glob(os.path.join(prompt_dir, '*.wav'))
    
    prompt_lookup: dict[str, VoicePrompt] = {}
    
    for wav_path in wav_paths:
        voice_name = os.path.splitext(os.path.basename(wav_path))[0]
        text_path = wav_path.replace('.wav', '.txt')

        with open(text_path, 'r') as f:
            text = f.read()

        prompt_lookup[voice_name] = VoicePrompt(
            wav_path=wav_path,
            text=text
        )

    return prompt_lookup

def create_demo(prompt_list: dict[str, VoicePrompt], model: 'Nanospeech'):
    
    def update_prompt(voice_name: str):
        return (
            prompt_list[voice_name].wav_path,
            prompt_list[voice_name].text
        )
    @spaces.GPU(progress=gr.Progress(track_tqdm=True))
    def _generate(prompt_audio: str, prompt_text: str, input_text: str, nfe_steps: int = 8, method: str = "rk4", cfg_strength: float = 2.0, sway_sampling_coef: float = -1.0, speed: float = 1.0, seed: Optional[int] = None):
        
        print(f'generating: {input_text}, prompt: {prompt_text}, prompt_audio: {prompt_audio}')
        
        # Load reference audio into memory
        if isinstance(prompt_audio, tuple):
            sr, ref_audio = prompt_audio
            
            ref_audio = convert_audio_int16_to_float32(ref_audio)
        else:
            ref_audio, sr = sf.read(prompt_audio)
            print('loaded from path')
            
        if sr != SAMPLE_RATE:
            raise ValueError("Reference audio must be mono with a sample rate of 24kHz")

        # Split input text into sentences
        sentences = split_sentences(input_text)
        is_single_generation = len(sentences) <= 1
            
        
        if is_single_generation:
            wave = generate_one(
                model=model,
                text=input_text,
                ref_audio=ref_audio,
                ref_audio_text=prompt_text,
                steps=nfe_steps,
                method=method,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                speed=speed,
                seed=seed,
                player=None,
            )
            if hasattr(wave, 'numpy'):
                wave = wave.numpy()
        else:
            # Generate multiple sentences and concatenate
            output = []
            for sentence_text in sentences:
                wave = generate_one(
                    model=model,
                    text=sentence_text,
                    ref_audio=ref_audio,
                    ref_audio_text=prompt_text,
                    steps=nfe_steps,
                    method=method,
                    cfg_strength=cfg_strength,
                    sway_sampling_coef=sway_sampling_coef,
                    speed=speed,
                    seed=seed,
                    player=None,
                )
                if hasattr(wave, 'numpy'):
                    wave = wave.numpy()
                output.append(wave)
            
            wave = np.concatenate(output, axis=0)

        return (SAMPLE_RATE, wave)
        
    
    with gr.Blocks() as demo:
        gr.Markdown("# (Unofficial) Nanospeech Demo")
        gr.Markdown("A simple, hackable text-to-speech system in PyTorch and MLX - [github](https://github.com/lucasnewman/nanospeech)")

        with gr.Group():
            gr.Markdown("## Select a voice prompt")
            voice_dropdown = gr.Dropdown(choices=list(prompt_list.keys()), value='celeste', interactive=True, label="Voice")
        
        with gr.Group():
            gr.Markdown("## Voice Prompt")
            with gr.Row():
                prompt_audio = gr.Audio(label="Audio", value=prompt_list[voice_dropdown.value].wav_path)
                prompt_text = gr.Textbox(label="Text", value=prompt_list[voice_dropdown.value].text, interactive=False)
    
            voice_dropdown.change(fn=update_prompt, inputs=voice_dropdown, outputs=[prompt_audio, prompt_text])
            
            with gr.Accordion("Advanced Settings", open=False):
                speed = gr.Slider(label="Speed", value=1.0, minimum=0.1, maximum=2.0, step=0.1)
                nfe_steps = gr.Slider(label="NFE Steps - more steps = more stable, but slower", value=8, minimum=1, maximum=64, step=1)
                
                method = gr.Dropdown(choices=["rk4", "euler", "midpoint"], value="rk4", label="Method")
                cfg_strength = gr.Slider(label="CFG Strength", value=2.0, minimum=0.0, maximum=5.0, step=0.1)
                sway_sampling_coef = gr.Slider(label="Sway Sampling Coef", value=-1.0, minimum=-5.0, maximum=5.0, step=0.1)
                
        with gr.Group():
            gr.Markdown("# Generate")
            
            input_text = gr.Textbox(label="Input Text", value="Hello, how are you?")
            generate_button = gr.Button("Generate")
        
        with gr.Group():
            output_audio = gr.Audio(label="Output Audio")
            
        generate_button.click(fn=_generate, inputs=[prompt_audio, prompt_text, input_text, nfe_steps, method, cfg_strength, sway_sampling_coef, speed], outputs=output_audio)
                
    return demo
                

if __name__ == "__main__":
    
    
    # Preload the model
    model = Nanospeech.from_pretrained("lucasnewman/nanospeech")
    prompt_list = get_prompt_list()
    
    demo = create_demo(prompt_list, model)
    demo.launch()
    