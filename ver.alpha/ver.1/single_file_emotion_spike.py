# =============================================================
# ONE‑FILE IMPLEMENTATION  |  language_to_spike.py  (v2 – λ & NLG fixes)
# -------------------------------------------------------------
# Changelog v2
#   • λ 계산을 단순화: raw = max(0, α·score − β) → λ = raw·λ_max (선형)
#     → 기본 β=0 으로 초기화하여 0‑스파이크 문제 해결.
#   • emotion_summary(): LLM 출력이 프롬프트 일부를 그대로 되돌려주는
#     경우를 감지, "Summary:" 이후 텍스트만 추출하도록 후처리.
# =============================================================

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------------------
# Optional LLM for natural‑language summary
# -------------------------------------------------------------

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore

    _LLM_NAME = os.getenv("EMO_LLM", "gpt2-medium")
    _TOK = AutoTokenizer.from_pretrained(_LLM_NAME)
    _LM = AutoModelForCausalLM.from_pretrained(_LLM_NAME)
except Exception:
    _TOK, _LM = None, None  # fallback

# -------------------------------------------------------------
# 1) Embedding helper
# -------------------------------------------------------------

class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)

# -------------------------------------------------------------
# 2) Neuromodulator seeds
# -------------------------------------------------------------

@dataclass
class NeuroSeeds:
    mapping: Dict[str, str]
    embedder: Embedder

    def __post_init__(self):
        self.names: List[str] = list(self.mapping)
        self._emb: np.ndarray = self.embedder.embed(list(self.mapping.values()))

    def embeds(self) -> np.ndarray:
        return self._emb

# -------------------------------------------------------------
# 3) Cosine‑softmax scoring
# -------------------------------------------------------------

def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def scores_for(text: str, seeds: NeuroSeeds, temperature: float = 1.0) -> Dict[str, float]:
    q = seeds.embedder.embed([text])[0]
    sims = cosine_similarity(q.reshape(1, -1), seeds.embeds()).ravel()
    probs = _softmax(sims / max(1e-8, temperature))
    return {nt: float(p) for nt, p in zip(seeds.names, probs)}

# -------------------------------------------------------------
# 4) λ & spike
# -------------------------------------------------------------

def compute_lambdas(scores: Dict[str, float], alpha: Dict[str, float], beta: Dict[str, float],
                    lambda_max: float) -> Dict[str, float]:
    """Linear: raw = max(0, α·s − β)"""
    lam = {}
    for nt, s in scores.items():
        raw = max(0.0, alpha[nt] * s - beta[nt])
        lam[nt] = raw * lambda_max
    return lam


def poisson_spikes(lam_hz: float, duration_ms: int, dt_ms: int, tau_ref_ms: int) -> np.ndarray:
    steps = duration_ms // dt_ms
    p = lam_hz * dt_ms * 1e-3
    rand = np.random.rand(steps)
    cand = np.nonzero(rand < p)[0]
    out, last = [], -1e9
    for t in cand:
        if t - last >= tau_ref_ms:
            out.append(t)
            last = t
    return np.asarray(out, dtype=int)

# -------------------------------------------------------------
# 5) RL update (REINFORCE + EMA baseline)
# -------------------------------------------------------------

class EMABaseline:
    def __init__(self, momentum: float = 0.9):
        self.m = momentum
        self.value = 0.0
        self.init = False

    def update(self, r: float):
        if not self.init:
            self.value = r
            self.init = True
        else:
            self.value = self.m * self.value + (1 - self.m) * r


def reinforce(alpha: Dict[str, float], beta: Dict[str, float], scores: Dict[str, float],
              reward: float, baseline: EMABaseline, lr: float = 0.1,
              clip: Tuple[float, float] = (0, 10)):
    adv = reward - baseline.value
    baseline.update(reward)
    for nt, s in scores.items():
        alpha[nt] = float(np.clip(alpha[nt] + lr * adv * s, *clip))
        beta[nt] = float(np.clip(beta[nt] - lr * adv * s, *clip))

# -------------------------------------------------------------
# 6) Text summary
# -------------------------------------------------------------

def _rule_summary(data: Dict[str, Dict]) -> str:
    dom = max(data.items(), key=lambda kv: kv[1]['lambda'])[0]
    return f"Dominant {dom} signalling suggests a {dom}-related emotional tone."


def _postprocess_llm(out: str) -> str:
    # Remove prompt echoes if LLM repeats it.
    if "Summary:" in out:
        out = out.split("Summary:")[-1]
    return out.strip().lstrip("- ")


def emotion_summary(data: Dict[str, Dict], temp: float = 0.8, max_tokens: int = 60) -> str:
    if _TOK is None or _LM is None:
        return _rule_summary(data)
    lines = [f"{nt}: score={d['score']:.3f}, lambda={d['lambda']:.1f}Hz" for nt, d in data.items()]
    prompt = (
        "You are an affective neuroscientist. Summarize the emotional state given these neuromodulator stats in one sentence.\n"
        + "\n".join(lines)
        + "\nSummary:"
    )
    ids = _TOK.encode(prompt, return_tensors="pt")
    out = _LM.generate(ids, do_sample=True, temperature=temp, top_p=0.9, max_new_tokens=max_tokens)
    result = _TOK.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
    return _postprocess_llm(result)

# -------------------------------------------------------------
# 7) Pipeline
# -------------------------------------------------------------

@dataclass
class EmotionSpikePipeline:
    seeds_dict: Dict[str, str]
    alpha: Dict[str, float] = field(default_factory=dict)
    beta: Dict[str, float] = field(default_factory=dict)
    lambda_max: float = 120.0
    duration_ms: int = 600
    dt_ms: int = 1
    tau_ref_ms: int = 2

    def __post_init__(self):
        self.embedder = Embedder()
        self.seeds = NeuroSeeds(self.seeds_dict, self.embedder)
        for nt in self.seeds.names:
            self.alpha.setdefault(nt, 1.0)
            self.beta.setdefault(nt, 0.0)  # β=0 by default
        self.baseline = EMABaseline()

    def forward(self, text: str, temp_embed: float = 1.0):
        sc = scores_for(text, self.seeds, temp_embed)
        lam = compute_lambdas(sc, self.alpha, self.beta, self.lambda_max)
        spikes = {nt: poisson_spikes(v, self.duration_ms, self.dt_ms, self.tau_ref_ms) for nt, v in lam.items()}
        return {nt: {"score": sc[nt], "lambda": lam[nt], "spikes": spikes[nt]} for nt in self.seeds.names}

    def process(self, text: str, temp_embed: float = 1.0, temp_lm: float = 0.8):
        stats = self.forward(text, temp_embed)
        return stats, emotion_summary(stats, temp)

    def train_step(self, text: str, reward: float, lr: float = 0.05):
        sc = scores_for(text, self.seeds)
        reinforce(self.alpha, self.beta, sc, reward, self.baseline, lr)

# -------------------------------------------------------------
# 8) Demo
# -------------------------------------------------------------

if __name__ == "__main__":
    SEEDS = {
        "DA": "motivation reward pursuit vigor",
        "5HT": "calm patience reflection stability",
        "NE": "arousal alert uncertainty vigilance",
        "ACh": "attention plasticity learning sensory gain",
        "GABA": "inhibition control balance noise suppression",
    }

    pipe = EmotionSpikePipeline(SEEDS)
    text = "I feel energized and sharply focused on my tasks."
    stats, summary = pipe.process(text, temp_embed=0.7)

    print("--- Stats ---")
    for nt, d in stats.items():
        print(f"{nt}: score={d['score']:.3f}, λ={d['lambda']:.1f}Hz, spikes={len(d['spikes'])}")

    print("\nSummary:")
    print(summary)

    # RL update example
    pipe.train_step(text, reward=1.0)
