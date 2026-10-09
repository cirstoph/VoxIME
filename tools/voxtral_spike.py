import json, math, wave, sys
from tokenizers import Tokenizer
_tok = Tokenizer.from_file("/workspace/voxtral-config/tokenizer.json")

import numpy as np
import onnxruntime as ort

CFG = "/workspace/voxtral-config"
ONNX = "/workspace/onnx"
AUDIO_TOKEN_ID = 24
BEGIN_AUDIO_TOKEN_ID = 25
NUM_AUDIO_TOKENS = 375
BOS, EOS = 1, 2

pre = json.load(open(f"{CFG}/preprocessor_config.json"))

def load_wav(path):
    w = wave.open(path)
    assert w.getframerate() == 16000 and w.getsampwidth() == 2 and w.getnchannels() == 1
    data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return data

def mel_spectrogram(audio, n_mels=128, n_fft=400, hop=160):
    if len(audio) < pre["n_samples"]:
        audio = np.pad(audio, (0, pre["n_samples"] - len(audio)))
    else:
        audio = audio[:pre["n_samples"]]

    # exact transformers.js mel_filter_bank (slaney scale + slaney norm)
    def hz_to_mel_slaney(f):
        return 15.0 + math.log(f / 1000.0 + 1e-10) / math.log(6.4) * 27.0 if f >= 1000 else (3.0 * f) / 200.0
    def mel_to_hz_slaney(m):
        if m >= 15.0: return 1000.0 * math.exp(math.log(6.4) / 27.0 * (m - 15.0))
        return (200.0 / 3.0) * m
    def linspace(a, b, n):
        if n == 1: return [a]
        d = (b - a) / (n - 1)
        return [a + d * i for i in range(n)]
    mel_min, mel_max = hz_to_mel_slaney(0.0), hz_to_mel_slaney(8000.0)
    mel_freqs = linspace(mel_min, mel_max, n_mels + 2)
    filter_freqs = np.array([mel_to_hz_slaney(m) for m in mel_freqs])
    fft_freqs = np.array(linspace(0, 8000.0, n_fft // 2 + 1))
    melfb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for i in range(n_mels):
        a, b, c = filter_freqs[i], filter_freqs[i + 1], filter_freqs[i + 2]
        up = (fft_freqs - a) / (b - a)
        down = (c - fft_freqs) / (c - b)
        melfb[i] = np.maximum(0.0, np.minimum(up, down))
        melfb[i] *= 2.0 / (filter_freqs[i + 2] - filter_freqs[i])

    window = np.hanning(n_fft + 1)[:-1].astype(np.float64)
    n_frames = 1 + (len(audio) - n_fft) // hop
    frames = np.lib.stride_tricks.sliding_window_view(audio, n_fft)[::hop][:n_frames].astype(np.float64)
    frames = frames * window[None, :]
    spec = np.fft.rfft(frames, axis=1)
    stft = (spec.real ** 2 + spec.imag ** 2)
    mel = stft @ melfb.T  # (frames, mels)
    log_spec = np.log10(np.clip(mel, a_min=1e-10, a_max=None))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    out = log_spec.astype(np.float32)
    if out.shape[0] < 3000:
        out = np.pad(out, ((0, 3000 - out.shape[0]), (0, 0)))
    return out.T  # (128, 3000)

def tokenize(text):
    return _tok.encode(text, add_special_tokens=False).ids

def main():
    audio = load_wav(sys.argv[1] if len(sys.argv) > 1 else "/workspace/spike/b.wav")
    print(f"audio: {len(audio)/16000:.1f}s")
    feats = mel_spectrogram(audio)
    print("mel:", feats.shape, "range", feats.min(), feats.max())

    so = ort.SessionOptions()
    so.log_severity_level = 4
    enc = ort.InferenceSession(f"{ONNX}/audio_encoder_q4f16.onnx", so, providers=["CPUExecutionProvider"])
    audio_feats = enc.run(None, {"audio_values": feats[None]})[0]
    print("audio_features:", audio_feats.shape, flush=True)
    del enc
    emb = ort.InferenceSession(f"{ONNX}/embed_tokens_q4.onnx", so, providers=["CPUExecutionProvider"])

    # prompt: <s>[INST][BEGIN_AUDIO][AUDIO]*375 lang:en [TRANSCRIBE][/INST]
    ids = [BOS]
    ids += tokenize("[INST]")
    ids += [BEGIN_AUDIO_TOKEN_ID] + [AUDIO_TOKEN_ID] * NUM_AUDIO_TOKENS
    ids += tokenize(" lang:en [TRANSCRIBE]")
    ids += tokenize("[/INST]")
    print("prompt tokens:", len(ids), flush=True)
    embeds = emb.run(None, {"input_ids": np.array([ids], dtype=np.int64)})[0]

    # replace audio token embeddings with encoder output
    positions = [i for i, t in enumerate(ids) if t == AUDIO_TOKEN_ID]
    assert len(positions) == audio_feats.shape[0], (len(positions), audio_feats.shape)
    for idx, pos in enumerate(positions):
        embeds[0, pos] = audio_feats[idx]

    del emb
    dec = ort.InferenceSession(f"{ONNX}/decoder_model_merged_q4.onnx", so, providers=["CPUExecutionProvider"])
    emb = ort.InferenceSession(f"{ONNX}/embed_tokens_q4.onnx", so, providers=["CPUExecutionProvider"])
    input_names = [i.name for i in dec.get_inputs()]
    num_layers = 30
    past = {f"past_key_values.{i}.key": np.zeros((1, 8, 0, 128), dtype=np.float32) for i in range(num_layers)}
    past.update({f"past_key_values.{i}.value": np.zeros((1, 8, 0, 128), dtype=np.float32) for i in range(num_layers)})

    cur_embeds = embeds
    cur_len = 0
    gen_ids = []
    for step in range(30):
        attn = np.ones((1, cur_len + cur_embeds.shape[1]), dtype=np.int64)
        pos = np.arange(cur_len, cur_len + cur_embeds.shape[1], dtype=np.int64)[None]
        feeds = {"inputs_embeds": cur_embeds, "attention_mask": attn, "position_ids": pos}
        feeds.update(past)
        outs = dec.run(None, feeds)
        logits = outs[0][0, -1]
        out_map = {}
        for i, o in enumerate(dec.get_outputs()):
            out_map[o.name] = outs[i]
        nid = int(np.argmax(logits))
        gen_ids.append(nid)
        if nid == EOS:
            break
        cur_len += cur_embeds.shape[1]
        past = {k: out_map[k.replace("past_key_values", "present")] for k in past}
        cur_embeds = emb.run(None, {"input_ids": np.array([[nid]], dtype=np.int64)})[0]

    print("generated ids:", gen_ids)
    text = _tok.decode(gen_ids)
    print("TEXT:", repr(text))

if __name__ == "__main__":
    main()
