# Nanospeech

### A simple, hackable text-to-speech system in PyTorch and MLX

Nanospeech is a research-oriented project to build a minimal, easy to understand text-to-speech system that scales to any level of compute. It supports voice matching from a reference speech sample, and comes with a variety of different voices built in.

An 82M parameter pretrained model (English-only) is available, which was trained on a single H100 GPU in a few days using only public domain data. The model is intentionally small to be a reproducible baseline and allow for fast inference. On recent M-series Apple Silicon or Nvidia GPUs, speech can be generated around ~3-5x faster than realtime.

All code and pretrained models are available under the MIT license, so you can modify and/or distribute them as you'd like.

## Details

Nanospeech is based on a current [line of research](#citations) in text-to-speech systems which jointly learn text alignment and waveform generation. It's designed to use minimal input data — just audio and text — and avoid any auxiliary models, such as forced aligners or phonemizers.

There are two single-file implementations, one in [PyTorch](./nanospeech/nanospeech_torch.py) and one in [MLX](./nanospeech/nanospeech_mlx.py), which are near line-for-line equivalence where possible to make it easy to experiment with and modify. Each implementation is around 1,500 lines of code.

## Quick Start

```bash
pip install nanospeech
```

```bash
python -m nanospeech.generate --text "The quick brown fox jumps over the lazy dog."
```

### Voices

Use the `--voice` parameter to select the voice used for speech:

`celeste` — [Sample](https://s3.amazonaws.com/lucasnewman.datasets/nanospeech/samples/celeste.wav)

`luna` — [Sample](https://s3.amazonaws.com/lucasnewman.datasets/nanospeech/samples/luna.wav)

`nash` — [Sample](https://s3.amazonaws.com/lucasnewman.datasets/nanospeech/samples/nash.wav)

`orion` — [Sample](https://s3.amazonaws.com/lucasnewman.datasets/nanospeech/samples/orion.wav)

`rhea` — [Sample](https://s3.amazonaws.com/lucasnewman.datasets/nanospeech/samples/rhea.wav)

Note these voices are all based on samples from the [LibriTTS-R](https://www.openslr.org/141/) dataset.

### Voice Matching

You can also provide a speech sample and a transcript to match to a specific voice, although the pretrained model has limited voice matching capabilities. See `python -m nanospeech.generate --help` for a full list of options to customize the voice.

## Training a Model

Nanospeech includes a PyTorch-based trainer using Accelerate, and is compatible with DistributedDataParallel for multi-GPU training.

It supports streaming from any [WebDataset](https://github.com/webdataset/webdataset), but it should be straightforward to swap in your own dataloader as well. An ideal dataset consists of high-quality speech paired with clean transcriptions.

See the [examples](./examples/) for an example of training both the base model and the duration predictor on the large-scale [Emilia](https://huggingface.co/datasets/amphion/Emilia-Dataset) dataset (note: Emilia is CC BY-NC-4.0 licensed).

## Limitations

As a research project, the pretrained model that comes with Nanospeech isn't designed for production usage. It may mispronounce words, has limited capability to match out-of-distribution voices, and can't generate very long speech samples.

However, the underlying architecture should scale up well to significantly more compute and larger datasets, so if training your own model is attractive, you can extend it to perform high-quality voice matching, multilingual speech generation, emotional expression, etc.

## Citations

```bibtex
@article{chen-etal-2024-f5tts,
      title     = {F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching}, 
      author    = {Yushen Chen and Zhikang Niu and Ziyang Ma and Keqi Deng and Chunhui Wang and Jian Zhao and Kai Yu and Xie Chen},
      year      = {2024},
      url       = {https://api.semanticscholar.org/CorpusID:273228169}
}
```

```bibtex
@inproceedings{Eskimez2024E2TE,
    title     = {E2 TTS: Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS},
    author    = {Sefik Emre Eskimez and Xiaofei Wang and Manthan Thakker and Canrun Li and Chung-Hsien Tsai and Zhen Xiao and Hemin Yang and Zirun Zhu and Min Tang and Xu Tan and Yanqing Liu and Sheng Zhao and Naoyuki Kanda},
    year      = {2024},
    url       = {https://api.semanticscholar.org/CorpusID:270738197}
}
```

```bibtex
@article{Le2023VoiceboxTM,
    title     = {Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale},
    author    = {Matt Le and Apoorv Vyas and Bowen Shi and Brian Karrer and Leda Sari and Rashel Moritz and Mary Williamson and Vimal Manohar and Yossi Adi and Jay Mahadeokar and Wei-Ning Hsu},
    year      = {2023},
    url       = {https://api.semanticscholar.org/CorpusID:259275061}
}
```

```bibtex
@article{tong2023generalized,
    title     = {Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport},
    author    = {Alexander Tong and Joshua Fan and Ricky T. Q. Chen and Jesse Bettencourt and David Duvenaud},
    year      = {2023}
    url       = {https://api.semanticscholar.org/CorpusID:259847293}
}
```

```bibtex
@article{peebles2022scalable,
    title     = {Scalable Diffusion Models with Transformers},
    author    = {Peebles, William and Xie, Saining},
    year      = {2022},
    url       = {https://api.semanticscholar.org/CorpusID:254854389}
}
```

```bibtex
@article{lipman2022flow,
    title     = {Flow Matching for Generative Modeling},
    author    = {Yaron Lipman and Ricky T. Q. Chen and Heli Ben-Hamu and Maximilian Nickel and Matt Le},
    year      = {2022},
    url       = {https://api.semanticscholar.org/CorpusID:252734897}
}
```

```bibtex
@article{koizumi2023librittsr,
    title     = {LibriTTS-R: A Restored Multi-Speaker Text-to-Speech Corpus},
    author    = {Yuma Koizumi and Heiga Zen and Shigeki Karita and Yifan Ding and Kohei Yatabe and Nobuyuki Morioka and Michiel Bacchiani and Yu Zhang and Wei Han and Ankur Bapna},
    year      = {2023},
    url       = {https://api.semanticscholar.org/CorpusID:258967444}
}
```

## License

The code in this repository is released under the MIT license as found in the
[LICENSE](LICENSE) file.
