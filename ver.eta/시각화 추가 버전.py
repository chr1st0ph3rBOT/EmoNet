# -*- coding: utf-8 -*-
"""
[Neuro-Chatbot: FIXED LOOP FIX EDITION]
- Fixed: Syntax Error (Removed stray image tags)
- Fixed: Missing 'Optional' import
- Feature: Refractory Period (3 Ticks) + Energy Decay (5%)
- Core: Double Buffered SNN + Hybrid Emotion Engine
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional  # [Fix] Optional 추가
import json, pathlib, sys, time, random
from collections import deque
import numpy as np

# ML Libraries
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ============================================================
# [PART 0] Settings & Data Loading
# ============================================================
BASE_DIR = pathlib.Path(__file__).parent.absolute()
DATA_PATH = BASE_DIR / "감성대화말뭉치(최종데이터)_Training.json"
MODEL_PATH = BASE_DIR / "emovec_real_brain.pkl"

EMOTION_VEC: Dict[str, List[float]] = {
    "E10":[0.60, 0.70, 0.20, 0.20], "E11":[0.55, 0.45, 0.40, 0.35],
    "E66":[0.85, 0.80, 0.10, 0.10], "E33":[0.50, 0.80, 0.10, 0.30],
    "E18":[0.20, 0.10, 0.90, 0.10], "E19":[0.10, 0.10, 0.95, 0.05],
    "E20":[0.25, 0.30, 0.55, 0.20],
    "E21":[0.10, 0.20, 0.30, 0.80], "E22":[0.22, 0.28, 0.60, 0.18],
    "E60":[0.10, 0.10, 0.20, 0.90], "E23":[0.48, 0.40, 0.58, 0.30],
    "E12":[0.45, 0.40, 0.55, 0.25], "E30":[0.38, 0.36, 0.60, 0.24],
    "E36":[0.44, 0.42, 0.48, 0.30], "E40":[0.28, 0.30, 0.58, 0.22],
    "E00":[0.5, 0.5, 0.5, 0.5]
}
EMO_DEFAULT = [0.5, 0.5, 0.5, 0.5]

def force_train_model():
    print("\n🔥 [System] 데이터셋 로딩 중...", end=" ", flush=True)
    if not DATA_PATH.exists():
        print("FAIL (No file)")
        return None
    try:
        data = json.loads(DATA_PATH.read_text(encoding='utf-8'))
        X, y = [], []
        for item in data:
            try:
                emo = item['profile']['emotion']['type']
                for k, v in item['talk']['content'].items():
                    if k.startswith("HS") and len(v) > 1:
                        X.append(v); y.append(emo)
            except: continue
        print(f"Done ({len(X)} sentences)")
        if len(X) < 10: X += ["테스트", "행복"]; y += ["E00", "E66"]

        print("🚀 [ML] 학습 시작 (Verbose)...")
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(min_df=1, ngram_range=(1,3), max_features=30000)),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=500, n_jobs=-1, verbose=1))
        ])
        pipe.fit(X, y)
        dump(pipe, MODEL_PATH)
        print("✅ [ML] 완료.\n")
        return pipe
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        return None

_MODEL_PIPE = force_train_model()

def predict_emotion_vector(text: str) -> Tuple[float, float, float, float]:
    text_clean = text.replace(" ", "")
    if any(w in text_clean for w in ["싫어", "짜증", "미친", "꺼져", "죽어"]): return (0.1, 0.2, 0.95, 0.1)
    if any(w in text_clean for w in ["좋아", "행복", "사랑", "최고", "신나"]): return (0.95, 0.8, 0.1, 0.1)
    if any(w in text_clean for w in ["슬퍼", "우울", "힘들", "눈물"]): return (0.2, 0.3, 0.4, 0.8)
    if _MODEL_PIPE:
        try:
            label = _MODEL_PIPE.predict([text])[0]
            return tuple(EMOTION_VEC.get(label, EMO_DEFAULT))
        except: pass
    return tuple(EMO_DEFAULT)

# ============================================================
# [PART 1] SNN Core (Loop Fixed)
# ============================================================
Vec4 = Tuple[float, float, float, float] 
def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float: return max(lo, min(hi, x))

@dataclass
class DataBox:
    __slots__ = ['K', 'V', 'trace']
    K: float; V: Vec4; trace: List[str]

@dataclass
class Edge:
    __slots__ = ['src', 'dst', 'W']
    src: "Neuron"; dst: "Neuron"; W: float
    def send(self, box: DataBox) -> None:
        self.dst._next_inbox.append((box, self))

@dataclass(eq=False)
class Neuron:
    __slots__ = ['name', 'kind', 'threshold', 'W', '_inbox', '_next_inbox', 'outgoing', 'incoming', 'off_ticks', 'alpha_exc', 'beta_inh', 'refractory_period']
    
    name: str; kind: str; threshold: float; W: float
    _inbox: deque; _next_inbox: deque
    outgoing: List[Edge]; incoming: List[Edge]
    off_ticks: int; refractory_period: int
    alpha_exc: float; beta_inh: float

    def __init__(self, name: str, kind: str):
        self.name = name; self.kind = kind
        self.threshold = 0.5; self.W = 1.0
        self._inbox = deque(); self._next_inbox = deque()
        self.outgoing = []; self.incoming = []
        self.off_ticks = 0
        self.refractory_period = 3 
        self.alpha_exc = 0.8; self.beta_inh = 0.7

    def connect_to(self, other: "Neuron") -> Edge:
        for e in self.outgoing:
            if e.dst is other: return e
        e = Edge(src=self, dst=other, W=1.0)
        self.outgoing.append(e); other.incoming.append(e); return e

    def swap_buffer(self):
        if self._next_inbox:
            self._inbox.extend(self._next_inbox)
            self._next_inbox.clear()

    def tick(self, net: "Network") -> None:
        if self.off_ticks > 0:
            self.off_ticks -= 1
            self._inbox.clear()
            return
        
        if not self._inbox: return

        Vs, Ws = [], []
        for box, edge in self._inbox: Vs.append(box.V); Ws.append(edge.src.W)
        
        if not Vs: V_in = (0.5,0.5,0.5,0.5)
        else:
            sW = sum(Ws) + 1e-9
            V_in = tuple(sum(Ws[i]*Vs[i][j] for i in range(len(Vs)))/sW for j in range(4)) # type: ignore

        trace_sample = self._inbox[-1][0].trace[-5:] if self._inbox else []
        trace_travel = trace_sample + [self.name]

        if len(self._inbox[-1][0].trace) > 15:
            self._inbox.clear()
            return

        outboxes, total_dW = [], 0.0
        fired = False 

        for box, edge in self._inbox:
            penalty = 0.3 if self.name in box.trace else 1.0
            
            if box.K * penalty < self.threshold: continue
            
            K_out, V_out, dW = self._specific_op(box.K, V_in)
            total_dW += dW
            
            K_out = min(K_out * self.W * penalty * 0.95, 1.5)
            
            outboxes.append(DataBox(K=K_out, V=tuple(clamp(x) for x in V_out), trace=trace_travel)) # type: ignore
            fired = True

            if self is net.terminal and not net.arrived:
                net.arrived = True; net.arrival_box = outboxes[-1]

        if total_dW != 0.0: self._apply_plasticity(total_dW, net)
        
        for ob in outboxes:
            for e in self.outgoing: e.send(ob)
        
        self._inbox.clear()

        if fired:
            self.off_ticks = self.refractory_period

    def _specific_op(self, K_in: float, V_in: Vec4) -> Tuple[float, Vec4, float]:
        D, S, NE, M = V_in; dW = 0.0
        if self.kind == "exc":
            factor = 1.0 + (self.alpha_exc * K_in * 1.5)
            V_out = tuple(0.5 + (v-0.5)*factor if abs(v-0.5)<0.45 else v for v in V_in) # type: ignore
            dW = (D + NE) * 0.8; return K_in, V_out, dW
        if self.kind == "inh":
            base_shrink = clamp(self.beta_inh * K_in, 0.0, 0.9)
            if D > 0.55 and NE < 0.6: real_shrink = base_shrink * 0.2
            else: real_shrink = base_shrink
            V_out = tuple(0.5 + (v-0.5)*(1.0-real_shrink) for v in V_in) # type: ignore
            dW = 0.1 if S > 0.6 else -0.8; return K_in, V_out, dW
        return K_in, V_in, 0.0

    def _apply_plasticity(self, dW_total: float, net: "Network") -> None:
        self.W = clamp(self.W + dW_total * 0.05, 0.0, 3.0)
        magnitude = abs(dW_total); steps = int(magnitude * 2)
        if dW_total > 0:
            candidates = [n for n in net.neurons if n is not self and all(e.dst is not n for e in self.outgoing)]
            if candidates:
                random.shuffle(candidates)
                for n in candidates[:steps]: self.connect_to(n)
        elif dW_total < 0:
            if not self.outgoing: return
            random.shuffle(self.outgoing)
            for _ in range(min(steps, len(self.outgoing))):
                if len(self.outgoing) <= 1: break 
                if random.random() < 0.5:
                    e = self.outgoing.pop()
                    if e in e.dst.incoming: e.dst.incoming.remove(e)

@dataclass
class Network:
    neurons: List[Neuron]
    terminal: Optional[Neuron] = None
    arrived: bool = False
    arrival_box: Optional[DataBox] = None
    
    def wire_randomly(self, p: float = 0.5, seed: int = 42):
        rng = random.Random(seed)
        for a in self.neurons:
            for b in self.neurons:
                if a is b: continue
                if rng.random() < p: a.connect_to(b)
    def tick(self):
        for n in self.neurons: n.swap_buffer()
        for n in self.neurons: n.tick(self)
    def inject(self, target: Neuron, box: DataBox):
        dummy = Neuron("Input", "exc")
        edge = Edge(src=dummy, dst=target, W=1.0)
        target._next_inbox.append((box, edge))
        self.arrived = False; self.arrival_box = None

# ============================================================
# [PART 2] Helper & Main
# ============================================================
def setup_brain(n_neurons=50, seed=42):
    rng = random.Random(seed)
    kinds = ["exc"]*20 + ["inh"]*15 + ["reg"]*15
    rng.shuffle(kinds)
    neurons = [Neuron(f"N{i}", k) for i, k in enumerate(kinds)]
    net = Network(neurons)
    net.wire_randomly(p=0.5, seed=seed)
    s, t = rng.sample(neurons, 2)
    net.terminal = t
    return net, s, t

def mix_emotions(curr: Vec4, prev: Vec4, decay: float = 0.4) -> Vec4:
    mixed = []
    for i in range(4):
        val = curr[i]*(1.0-decay) + prev[i]*decay
        mixed.append(val)
    return tuple(mixed) # type: ignore

def generate_prompt(user_text: str, vec: Vec4) -> str:
    D, S, NE, M = vec
    moods = []
    if D>0.7: moods.append("신남")
    elif D<0.3: moods.append("지루함")
    if NE>0.7: moods.append("예민/공격적")
    elif NE<0.3: moods.append("이완됨")
    if M>0.7: moods.append("무기력")
    if S>0.7: moods.append("안정적")
    mood_str = ", ".join(moods) if moods else "평온함"
    
    return f"""
[LLM 프롬프트]
상태: D={D:.2f} S={S:.2f} NE={NE:.2f} M={M:.2f}
기분: "{mood_str}"
입력: "{user_text}"
"""

def main():
    print(f"\n🧠 [Neuro-Chatbot: LOOP FIX EDITION]")
    # [수정됨] 이미지 태그 제거됨
    print("   - Refractory Period (3 Ticks) Applied.")
    print("   - Energy Decay (5%) Applied.")
    
    net, s_node, t_node = setup_brain(n_neurons=50, seed=42)
    current_mood = (0.5, 0.5, 0.5, 0.5)

    while True:
        user_input = input("\n👤 You: ").strip()
        if user_input.lower() in ["quit", "exit"]: break
        if not user_input: continue
        
        raw_vec = predict_emotion_vector(user_input)
        mixed_vec = mix_emotions(raw_vec, current_mood, decay=0.3)
        k_val = 0.9 if mixed_vec[0] > 0.6 else 0.7
        box = DataBox(K=k_val, V=mixed_vec, trace=["Input"])
        net.inject(s_node, box)
        
        print("   🧠 Simulation:", end=" ")
        final_vec = current_mood
        
        for i in range(40):
            net.tick()
            if i % 2 == 0:
                active = sum(1 for n in net.neurons if n._inbox or n._next_inbox)
                if active > 0: print(".", end="", flush=True)
            
            if net.arrived and net.arrival_box:
                final_vec = net.arrival_box.V
                print(f" DONE! (Step {len(net.arrival_box.trace)})")
                break
            time.sleep(0.01)
        else:
            print(" LOST (Faded out)")

        if net.arrived:
            current_mood = final_vec
        else:
            d, s, ne, m = current_mood
            current_mood = (d*0.9, s*0.9, clamp(ne+0.1), clamp(m+0.1))

        print(generate_prompt(user_input, current_mood))

if __name__ == "__main__":
    main()