import numpy as np
import torch
from typing import Dict
from tortoise.api import TextToSpeech
from tortoise.inference import tts_multi_nosave, tts_single_nosave
from tortoise.models.vocoder import VocConf
from tortoise.utils.audio import load_voices
from tortoise.utils.text import split_and_recombine_text

SPLIT_CRITERION_CHARS = (
    200  # how long do messages go to split inference/how many chars per split
)


class Model:
    def __init__(self, name: str):
        # super().__init__(name)
        self.name = name
        self.ready = False
        self.tts = None
        self.voices = None
        # self.model_name = options["MODEL_NAME"] TODO

    def call_tts(self, text: str, voice: tuple):
        return self.tts.tts_with_preset(
            text,
            k=1,
            preset="ultra_fast",
            diffusion_iterations=30,
            voice_samples=voice[0],
            conditioning_latents=voice[1],
            use_deterministic_seed=1,
            cvvp_amount=0.0,
            half=False,
        )

    def load(self):
        self.tts = TextToSpeech(
            high_vram=True,
            kv_cache=True,
            vocoder=VocConf.BigVGAN_Base,
        )
        self.voices = {
            "male": load_voices(["MALE1"]),
            "female": load_voices(["FEMALE1"]),
        }
        self.ready = True

    def predict(self, request: Dict, *args, **kwargs) -> Dict:
        genAudio = None
        if len(request["text"]) < SPLIT_CRITERION_CHARS:
            genAudio = tts_single_nosave(
                self.call_tts, request["text"], self.voices[request["voice"]]
            )
        else:
            desired_length = SPLIT_CRITERION_CHARS
            texts = split_and_recombine_text(
                request["text"], desired_length, desired_length + 100
            )
            genAudio = tts_multi_nosave(
                self.call_tts, texts, self.voices[request["voice"]]
            )
        return np.squeeze(genAudio)


if __name__ == "__main__":
    with torch.no_grad():
        model = Model("Tortoise")
        model.load()
        audio = model.predict({"text": "The expressiveness of autoregressive transformers is literally nuts! I absolutely adore them.", "voice": "female"})
        # convert audio to wav
        import soundfile as sf

        sf.write("test.wav", audio, 44100)
        print("done")
