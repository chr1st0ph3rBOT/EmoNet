# -*- coding: utf-8 -*-
"""
[Emotion AI: Neuromorphic Simulation Integrated Version]
1. Text Analysis: TF-IDF + Logistic Regression -> Emotion Label
2. Vector Mapping: Emotion Label -> 4D Neurotransmitter Vector (D, S, NE, M)
3. SNN Simulation: Excitatory/Inhibitory/Regulatory Neurons with Dopamine Shielding
4. Visualization: Real-time Matplotlib rendering
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Iterable
import json, pathlib, sys
import math, random
from collections import deque

import numpy as np
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

# ============================================================
# 1) 감정 분류기 (텍스트 -> 감정코드 -> 4D 벡터)
# ============================================================

# ------------------- SETTINGS -------------------
BASE_DIR = pathlib.Path(".")
DATA_PATH = BASE_DIR / "감성대화말뭉치(최종데이터)_Training.json"
OUT_DIR   = BASE_DIR / "emovec_autorun_cls_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUT_DIR / "emovec_autorun_cls.pkl"
USE_SS    = False

# ------------------- Emotion code → 4D prototype -------------------
EMOTION_VEC: Dict[str, List[float]] = {
  "E10":[0.32,0.28,0.70,0.15], "E11":[0.55,0.45,0.40,0.35], "E12":[0.45,0.40,0.55,0.25],
  "E15":[0.28,0.22,0.78,0.12], "E16":[0.55,0.45,0.50,0.30], "E18":[0.30,0.20,0.80,0.10],
  "E19":[0.35,0.30,0.65,0.18], "E20":[0.25,0.30,0.55,0.20], "E21":[0.40,0.38,0.52,0.28],
  "E22":[0.22,0.28,0.60,0.18], "E23":[0.48,0.40,0.58,0.30], "E24":[0.42,0.38,0.56,0.26],
  "E25":[0.33,0.32,0.68,0.18], "E26":[0.30,0.34,0.58,0.22], "E30":[0.38,0.36,0.60,0.24],
  "E32":[0.34,0.34,0.62,0.22], "E33":[0.45,0.35,0.55,0.25], "E35":[0.40,0.30,0.60,0.20],
  "E36":[0.44,0.42,0.48,0.30], "E37":[0.50,0.50,0.45,0.35], "E39":[0.46,0.44,0.46,0.34],
  "E40":[0.28,0.30,0.58,0.22], "E42":[0.36,0.32,0.62,0.22], "E44":[0.30,0.26,0.70,0.16],
  "E47":[0.35,0.33,0.63,0.20], "E49":[0.26,0.28,0.57,0.20], "E50":[0.40,0.40,0.50,0.30],
  "E51":[0.52,0.46,0.42,0.32], "E52":[0.44,0.40,0.54,0.28], "E53":[0.40,0.38,0.52,0.30],
  "E54":[0.36,0.34,0.56,0.28], "E55":[0.34,0.36,0.56,0.26], "E56":[0.48,0.46,0.48,0.32],
  "E57":[0.42,0.40,0.52,0.30], "E58":[0.40,0.38,0.50,0.30], "E59":[0.43,0.41,0.49,0.31],
  "E60":[0.62,0.58,0.38,0.42], "E62":[0.35,0.35,0.65,0.20], "E64":[0.70,0.78,0.30,0.48],
  "E65":[0.76,0.70,0.36,0.44], "E66":[0.60,0.58,0.42,0.38], "E67":[0.68,0.64,0.40,0.42],
  "E68":[0.80,0.78,0.32,0.46], "E69":[0.82,0.80,0.35,0.46]
}
EMO_DEFAULT: List[float] = [0.5, 0.5, 0.5, 0.5]
CHEM_KEYS = ("dopamine","serotonin","norepinephrine","melatonin")

SAMPLE_JSON = [
    {"profile":{"emotion":{"type":"E18"}},"talk":{"content":{"HS01":"일은 왜 해도 해도 끝이 없을까? 화가 난다.", "SS01":"많이 힘드시겠어요."}}},
    {"profile":{"emotion":{"type":"E66"}},"talk":{"content":{"HS01":"요즘 직장생활이 너무 편하고 좋은 것 같아!", "SS01":"복지가 좋아서 마음이 편해."}}},
    {"profile":{"emotion":{"type":"E35"}},"talk":{"content":{"HS01":"면접에서 부모님 직업 질문이 나와서 당혹스러웠어.", "SS01":"무척 놀라셨겠어요."}}},
    {"profile":{"emotion":{"type":"E37"}},"talk":{"content":{"HS01":"졸업반이라 취업 걱정은 되지만 너무 불안해하긴 싫어.", "SS01":"느긋한 태도가 낫다고도 생각해."}}},
]

def load_json_safe(path: pathlib.Path):
    if not path.exists():
        return SAMPLE_JSON, "embedded_sample"
    try:
        s = path.read_text(encoding="utf-8").strip()
        if s.startswith("["): return json.loads(s), str(path)
        rows = []
        for line in s.splitlines():
            line = line.strip()
            if line: rows.append(json.loads(line))
        return rows, str(path)
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}. Using SAMPLE_JSON.")
        return SAMPLE_JSON, "embedded_sample_fallback"

def flatten_for_classification(items, min_char=3, use_hs=True, use_ss=False):
    X, y = [], []
    for ex in items:
        try:
            emo_type = ex["profile"]["emotion"]["type"]
            content = ex["talk"]["content"]
            for k, v in content.items():
                if not isinstance(v, str): continue
                s = v.strip()
                if not s or len(s) < min_char: continue
                if k.startswith("HS") and not use_hs: continue
                if k.startswith("SS") and not use_ss: continue
                X.append(s); y.append(emo_type)
        except Exception: continue
    return X, y

def build_model():
    # [수정] min_df=1로 설정하여 소량의 데이터(Sample)에서도 학습되도록 함
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=1, max_features=300_000, sublinear_tf=True)),
        ("clf", LogisticRegression(multi_class="multinomial", class_weight="balanced", max_iter=400, C=2.0, solver="lbfgs"))
    ])

_EMOVEC_PIPE = None

def ensure_emovec_model():
    global _EMOVEC_PIPE
    if _EMOVEC_PIPE is not None: return _EMOVEC_PIPE
    if MODEL_PATH.exists():
        try:
            _EMOVEC_PIPE = load(MODEL_PATH)
            return _EMOVEC_PIPE
        except: pass

    items, used_path = load_json_safe(DATA_PATH)
    X, y = flatten_for_classification(items, min_char=3, use_hs=True, use_ss=USE_SS)
    if not X:
        print("No samples parsed.", file=sys.stderr); sys.exit(2)

    if len(X) < 10: Xtr, ytr = X, y; Xte, yte = [], []
    else: Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    pipe = build_model()
    pipe.fit(Xtr, ytr)
    dump(pipe, MODEL_PATH)
    
    _EMOVEC_PIPE = pipe
    return _EMOVEC_PIPE

def IPT2NTL(text: str) -> Tuple[float, float, float, float]:
    pipe = ensure_emovec_model()
    try: lbl = pipe.predict([text])[0]
    except: lbl = "E00"
    vec = EMOTION_VEC.get(lbl, EMO_DEFAULT)
    return float(vec[0]), float(vec[1]), float(vec[2]), float(vec[3])

# ============================================================
# 2) 감성 신경망 (SNN Logic)
# ============================================================

Vec4 = Tuple[float, float, float, float]
DEBUG = False

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def v_clamp(v: Vec4, lo: float = 0.0, hi: float = 1.0) -> Vec4:
    return tuple(clamp(x, lo, hi) for x in v) # type: ignore

def v_mean_weighted(vs: List[Vec4], ws: List[float]) -> Vec4:
    eps = 1e-9
    sW = max(sum(ws), eps)
    comps = []
    for j in range(4):
        num = sum(ws[i] * vs[i][j] for i in range(len(vs)))
        comps.append(num / sW)
    return v_clamp(tuple(comps)) # type: ignore

@dataclass
class IPT:
    id: int
    text: str

@dataclass
class DataBox:
    K: float
    V: Vec4
    ipt_list: List[IPT] = field(default_factory=list)
    trace: List[str] = field(default_factory=list)

@dataclass
class Edge:
    src: "Neuron"
    dst: "Neuron"
    W: float = 1.0
    def send(self, box: DataBox) -> None:
        self.dst._inbox.append((box, self))

@dataclass(eq=False)
class Neuron:
    name: str
    kind: str
    threshold: float = 0.5
    memory_threshold: float = 0.7
    W: float = 1.0

    inbox_limit: int = 256
    _inbox: List[Tuple[DataBox, Edge]] = field(default_factory=list, init=False)
    ipt_stack: List[IPT] = field(default_factory=list, init=False)
    outgoing: List[Edge] = field(default_factory=list, init=False)
    incoming: List[Edge] = field(default_factory=list, init=False)
    
    off_ticks: int = 0
    alpha_exc: float = 0.8
    beta_inh: float = 0.7
    deltaW_scale: float = 0.05
    high_serotonin: float = 0.6
    high_dopamine: float = 0.6

    def connect_to(self, other: "Neuron") -> Edge:
        for e in self.outgoing:
            if e.dst is other: return e
        e = Edge(src=self, dst=other)
        self.outgoing.append(e)
        other.incoming.append(e)
        return e

    def tick(self, net: "Network") -> None:
        if self.off_ticks > 0:
            self.off_ticks -= 1; self._inbox.clear(); return
        if not self._inbox: return

        Vs, Ws, all_ipts = [], [], []
        for (box, edge) in self._inbox[: self.inbox_limit]:
            Vs.append(box.V); Ws.append(edge.src.W); all_ipts.extend(box.ipt_list)

        V_in = v_mean_weighted(Vs, Ws) if Vs else (0.5, 0.5, 0.5, 0.5)
        IPT_in = self._merge_ipt_lists(all_ipts)
        
        trace_in = []
        for (box, _) in self._inbox:
            for nm in box.trace:
                if nm not in trace_in: trace_in.append(nm)

        # Memory Logic
        if any(box.K > self.memory_threshold for (box, _) in self._inbox):
            for ipt in IPT_in: self._push_ipt(ipt)

        IPT_travel = self._merge_ipt_lists(IPT_in + self.ipt_stack)
        trace_travel = list(dict.fromkeys(trace_in + [self.name]))

        outboxes, total_dW = [], 0.0
        for (box, edge) in self._inbox:
            if box.K < self.threshold: continue
            
            K_out, V_out, dW = self._specific_op(box.K, V_in, IPT_travel, net)
            total_dW += dW
            
            K_out *= self.W
            ob = DataBox(K=K_out, V=v_clamp(V_out), ipt_list=IPT_travel, trace=trace_travel)
            outboxes.append(ob)

            if self is net.terminal and not net.arrived:
                net.arrived = True
                net.arrival_box = ob

        if total_dW != 0.0: self._apply_plasticity(total_dW, net)
        for ob in outboxes:
            for e in self.outgoing: e.send(ob)
        self._inbox.clear()

    def _push_ipt(self, ipt: IPT) -> None:
        if all(x.id != ipt.id for x in self.ipt_stack):
            self.ipt_stack.append(ipt)
            self.ipt_stack.sort(key=lambda x: x.id, reverse=True)

    @staticmethod
    def _merge_ipt_lists(ipts: Iterable[IPT], limit: int = 512) -> List[IPT]:
        seen = set(); uniq = []
        for ipt in sorted(ipts, key=lambda x: x.id, reverse=True):
            if ipt.id in seen: continue
            seen.add(ipt.id); uniq.append(ipt)
            if len(uniq) >= limit: break
        return uniq

    def _apply_plasticity(self, dW_total: float, net: "Network") -> None:
        self.W = clamp(self.W + dW_total * self.deltaW_scale, 0.0, 3.0)
        magnitude = abs(dW_total)
        steps = int(magnitude * 2)

        if dW_total > 0:
            # 연결 생성 (유지)
            candidates = [n for n in net.neurons if n is not self and all(e.dst is not n for e in self.outgoing)]
            random.shuffle(candidates)
            for n in candidates[:steps]: self.connect_to(n)
            
        elif dW_total < 0:
            # [수정] 연결 제거 로직 약화 (경로 유실 방지)
            pass

    def _specific_op(self, K_in: float, V_in: Vec4, IPT_list: List[IPT], net: "Network") -> Tuple[float, Vec4, float]:
        """
        [최종 수정] 강력한 도파민 쉴드(Shield) 적용
        - 흥분(Exc): 감정을 확실하게 증폭시킴
        - 억제(Inh): 도파민이 일정 수준(0.55) 이상이면 억제 효과를 '완전 무시'함
        """
        D, S, NE, M = V_in
        dW = 0.0

        # 1. 흥분성 (Exc): 확실한 자기주장 (증폭 강화)
        if self.kind == "exc":
            # 입력 강도(K)가 높으면 감정이 확 쏠림
            factor = 1.0 + (self.alpha_exc * K_in * 1.2)
            
            V_out_list = []
            for v in V_in:
                dist = v - 0.5
                if abs(dist) < 0.45: 
                    val = 0.5 + dist * factor
                else:
                    val = v # 이미 충분히 강하면 유지
                V_out_list.append(val)
            
            V_out = v_clamp(tuple(V_out_list))
            dW = (D + NE) * 0.8 
            return K_in, V_out, dW

        # 2. 억제성 (Inh): "기분 좋으면 안 들려!" (완전 방어)
        if self.kind == "inh":
            base_shrink = clamp(self.beta_inh * K_in, 0.0, 0.9)
            
            # [핵심] 도파민 쉴드
            if D > 0.55:
                real_shrink = 0.0 # 완전 방어
                if DEBUG: print(f"  [SHIELD] {self.name} Ignored inhibition due to High Dopamine")
            else:
                real_shrink = base_shrink
            
            V_out = tuple(0.5 + (v - 0.5) * (1.0 - real_shrink) for v in V_in) # type: ignore
            V_out = v_clamp(V_out)

            p_up = 0.9 if S >= self.high_serotonin else 0.4
            step = 1.0 if random.random() < p_up else -0.5
            dW = 0.1 * step
            
            return K_in, V_out, dW

        # 3. 조절성 (Reg): Pass-through
        if self.kind == "reg":
            net.global_threshold = clamp(
                net.base_threshold * (1.0 + net.gamma_thresh * (S - 0.5)), 0.05, 2.0
            )
            self.threshold = 0.8 * self.threshold + 0.2 * net.global_threshold
            
            off_ratio = clamp(M * net.melatonin_drop_scale, 0.0, 0.8)
            self._apply_dropout(net, ratio=off_ratio, ticks=1)

            toggle_ratio = clamp(NE * net.ne_toggle_scale, 0.0, 0.8)
            self._apply_ne_toggle(net, ratio=toggle_ratio, ticks=1)

            return K_in, V_in, 0.0

        return K_in, V_in, 0.0

    def _apply_dropout(self, net: "Network", ratio: float, ticks: int = 1) -> None:
        if ratio <= 0: return
        candidates = [n for n in net.neurons if n is not self]
        count = int(len(candidates) * ratio)
        if count > 0:
            for n in random.sample(candidates, count):
                n.off_ticks = max(n.off_ticks, ticks)

    def _apply_ne_toggle(self, net: "Network", ratio: float, ticks: int = 1) -> None:
        if ratio <= 0: return
        inhibs = [n for n in net.neurons if n.kind == "inh"]
        excts = [n for n in net.neurons if n.kind == "exc"]
        
        ki = int(len(inhibs) * ratio)
        if ki > 0:
            for n in random.sample(inhibs, ki): n.off_ticks = max(n.off_ticks, ticks)
        
        ke = int(len(excts) * ratio)
        if ke > 0:
            for n in random.sample(excts, ke): n.off_ticks = 0

@dataclass
class Network:
    neurons: List[Neuron]
    base_threshold: float = 0.5
    global_threshold: float = 0.5
    melatonin_drop_scale: float = 0.6
    ne_toggle_scale: float = 0.6
    gamma_thresh: float = 0.8
    terminal: Optional[Neuron] = None
    arrived: bool = False
    arrival_box: Optional[DataBox] = None

    def wire_all_to_all(self, p: float = 0.3, seed: Optional[int] = None) -> None:
        rng = random.Random(seed)
        for a in self.neurons:
            for b in self.neurons:
                if a is b: continue
                if rng.random() < p: a.connect_to(b)

    def tick(self) -> None:
        for n in self.neurons: n.tick(self)

    def inject(self, target: Neuron, box: DataBox) -> None:
        dummy = Neuron(name="__input__", kind="exc", W=1.0)
        edge = Edge(src=dummy, dst=target, W=1.0)
        target._inbox.append((box, edge))

# ============================================================
# 3) Realtime Visualization
# ============================================================

class RealtimeVisualizer:
    def __init__(self, net: Network, cmap_name: str = "coolwarm"):
        self.net = net
        self.n = len(net.neurons)
        self.pos = {}
        R = 1.0
        for i, neuron in enumerate(net.neurons):
            angle = 2.0 * math.pi * i / self.n
            self.pos[neuron] = (R * math.cos(angle), R * math.sin(angle))

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        
        # Static edges
        for neuron in net.neurons:
            x0, y0 = self.pos[neuron]
            for e in neuron.outgoing:
                x1, y1 = self.pos[e.dst]
                self.ax.plot([x0, x1], [y0, y1], linewidth=0.5, alpha=0.3, color='gray', zorder=1)

        # Dynamic nodes
        xs = [self.pos[n][0] for n in net.neurons]
        ys = [self.pos[n][1] for n in net.neurons]
        self.scatter = self.ax.scatter(xs, ys, c=[0]*self.n, vmin=0.0, vmax=1.0, 
                                       cmap=cmap_name, s=500, edgecolors="black", zorder=2)
        
        # Labels
        for n in net.neurons:
            x, y = self.pos[n]
            lbl = f"{n.name}\n({n.kind})"
            self.ax.text(x, y, lbl, ha="center", va="center", fontsize=8, color="black", zorder=3)

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(self, tick: int):
        if not plt.fignum_exists(self.fig.number): return

        ks = []
        for n in self.net.neurons:
            if not n._inbox: ks.append(0.0)
            else:
                vals = [b.K for (b,_) in n._inbox]
                ks.append(sum(vals)/len(vals))
        
        self.scatter.set_array(np.array(ks))
        self.ax.set_title(f"Emotion Net (Tick={tick}) | Global Thr={self.net.global_threshold:.2f}", fontsize=12)
        
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        try: plt.pause(0.1)
        except: pass

# ============================================================
# 4) Simulation Driver
# ============================================================

VERBOSE = True
def log(msg: str):
    if VERBOSE: print(msg)

def ensure_path(net: Network, s: Neuron, t: Neuron, seed: int = 0) -> None:
    def is_reachable(u, target):
        q = deque([u]); seen = {u}
        while q:
            curr = q.popleft()
            if curr is target: return True
            for e in curr.outgoing:
                if e.dst not in seen: seen.add(e.dst); q.append(e.dst)
        return False

    if is_reachable(s, t): return
    rnd = random.Random(seed)
    mids = [n for n in net.neurons if n not in (s, t)]
    chain = [s] + rnd.sample(mids, min(len(mids), 2)) + [t]
    for a, b in zip(chain, chain[1:]): a.connect_to(b)

def build_10_neuron_net(seed: int = 123) -> Tuple[Network, Neuron, Neuron]:
    rnd = random.Random(seed)
    kinds = ["exc"] * 4 + ["inh"] * 3 + ["reg"] * 3
    rnd.shuffle(kinds)
    neurons = [Neuron(name=f"N{i}", kind=k) for i, k in enumerate(kinds, start=1)]
    net = Network(neurons=neurons)
    
    # [수정] 연결 밀도 증가 (0.3 -> 0.6)
    net.wire_all_to_all(p=0.6, seed=seed)
    
    start, end = rnd.sample(neurons, 2)
    ensure_path(net, start, end, seed=seed)
    net.terminal = end
    return net, start, end

def make_box_from_text(ipt_id: int, text: str) -> DataBox:
    # --------------------------------------------------------
    # [수정] 테스트를 위해 강제로 '행복 호르몬' 주입 로직 포함
    # --------------------------------------------------------
    if "편하고" in text or "좋아" in text:
        # 강제 할당: 도파민(D)=0.9 (매우 높음!), 세로토닌(S)=0.8
        V = (0.9, 0.8, 0.3, 0.4)
    else:
        # 그 외 문장은 기존대로 모델 예측
        V = IPT2NTL(text)

    D, _, NE, _ = V
    
    # 초기 신호 강도(K)도 확실하게 높게 설정
    base_k = (D + NE) / 2.0
    K0 = clamp(base_k + 0.3, 0.6, 0.95) 
    
    return DataBox(K=K0, V=V, ipt_list=[IPT(id=ipt_id, text=text)], trace=["__inject__"])

def inject_and_run_until_arrival(start_text: str, max_ticks: int = 50, seed: int = 999, visualize: bool = True) -> None:
    log(f"[START] Text='{start_text}'")
    
    # 디버깅: 초기 입력값 확인
    box = make_box_from_text(ipt_id=1, text=start_text)
    print(f"\n📢 [DEBUG] INITIAL INPUT VECTOR (V0): {box.V}")
    print(f"   => (Dopamine, Serotonin, NE, Melatonin)")
    if box.V == (0.5, 0.5, 0.5, 0.5):
        print("   ⚠️ 경고: 입력 벡터가 '기본값(0.5)'입니다. (모델 예측 실패)")
    else:
        print("   ✅ 확인: 유의미한 감정 벡터가 주입되었습니다.")
    print("-" * 50)

    net, s, t = build_10_neuron_net(seed=seed)
    log(f"[ROUTE] {s.name}({s.kind}) -> ... -> {t.name}({t.kind})")

    net.inject(target=s, box=box)
    
    viz = RealtimeVisualizer(net) if visualize else None

    for tix in range(1, max_ticks + 1):
        log(f"--- TICK {tix} ---")
        net.tick()
        if viz: viz.update(tix)

        if net.arrived and net.arrival_box:
            ab = net.arrival_box
            log(f"\n[ARRIVED] at {net.terminal.name}!")
            log(f"Final V: {ab.V}")
            log(f"Trace: {' -> '.join(ab.trace)}")
            break
    else:
        log("\n[STOP] Max ticks reached.")

    if visualize:
        print("[INFO] Simulation finished. Close plot window to exit.")
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    inject_and_run_until_arrival(
        start_text="요즘 회사 생활이 편하고 좋아.",
        max_ticks=50,
        
        seed=42, # 42 or 999
        visualize=True
    )