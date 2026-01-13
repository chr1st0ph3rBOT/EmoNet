# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Iterable
import json, pathlib, sys
import math, random, itertools
from collections import deque

import numpy as np
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

# ============================================================
# 1) 감정 분류기 + 텍스트 → 4D 감정벡터
# ============================================================

# ------------------- SETTINGS -------------------
DATA_PATH = pathlib.Path("C:/Users/Admin/Documents/GitHub/emotionAI/ver.epsilon/감성대화말뭉치(최종데이터)_Training.json")  # JSON array or JSONL
OUT_DIR   = pathlib.Path("emovec_autorun_cls_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUT_DIR / "emovec_autorun_cls.pkl"
REPORT_PATH = OUT_DIR / "emovec_autorun_cls_report.json"
USE_SS    = False                        # 기본은 HS만 사용(노이즈 감소)

TEST_TEXTS = [
    "일이 왜 이렇게 끝이 없지? 화나.",
    "요즘 회사 생활이 편하고 좋아.",
    "면접에서 갑자기 예상치 못한 질문이 나와서 당황했어.",
    "친구들은 다 취업했는데 나만 못 해서 불안해.",
]

# ------------------- Emotion code → 4D prototype -------------------
EMOTION_VEC: Dict[str, List[float]] = {
  "E10":[0.32,0.28,0.70,0.15],
  "E11":[0.55,0.45,0.40,0.35],
  "E12":[0.45,0.40,0.55,0.25],
  "E15":[0.28,0.22,0.78,0.12],
  "E16":[0.55,0.45,0.50,0.30],
  "E18":[0.30,0.20,0.80,0.10],
  "E19":[0.35,0.30,0.65,0.18],
  "E20":[0.25,0.30,0.55,0.20],
  "E21":[0.40,0.38,0.52,0.28],
  "E22":[0.22,0.28,0.60,0.18],
  "E23":[0.48,0.40,0.58,0.30],
  "E24":[0.42,0.38,0.56,0.26],
  "E25":[0.33,0.32,0.68,0.18],
  "E26":[0.30,0.34,0.58,0.22],
  "E30":[0.38,0.36,0.60,0.24],
  "E32":[0.34,0.34,0.62,0.22],
  "E33":[0.45,0.35,0.55,0.25],
  "E35":[0.40,0.30,0.60,0.20],
  "E36":[0.44,0.42,0.48,0.30],
  "E37":[0.50,0.50,0.45,0.35],
  "E39":[0.46,0.44,0.46,0.34],
  "E40":[0.28,0.30,0.58,0.22],
  "E42":[0.36,0.32,0.62,0.22],
  "E44":[0.30,0.26,0.70,0.16],
  "E47":[0.35,0.33,0.63,0.20],
  "E49":[0.26,0.28,0.57,0.20],
  "E50":[0.40,0.40,0.50,0.30],
  "E51":[0.52,0.46,0.42,0.32],
  "E52":[0.44,0.40,0.54,0.28],
  "E53":[0.40,0.38,0.52,0.30],
  "E54":[0.36,0.34,0.56,0.28],
  "E55":[0.34,0.36,0.56,0.26],
  "E56":[0.48,0.46,0.48,0.32],
  "E57":[0.42,0.40,0.52,0.30],
  "E58":[0.40,0.38,0.50,0.30],
  "E59":[0.43,0.41,0.49,0.31],
  "E60":[0.62,0.58,0.38,0.42],
  "E62":[0.35,0.35,0.65,0.20],
  "E64":[0.70,0.78,0.30,0.48],
  "E65":[0.76,0.70,0.36,0.44],
  "E66":[0.60,0.58,0.42,0.38],
  "E67":[0.68,0.64,0.40,0.42],
  "E68":[0.80,0.78,0.32,0.46],
  "E69":[0.82,0.80,0.35,0.46]
}
EMO_DEFAULT = [0.5, 0.5, 0.5, 0.5]
KEYS = ["dopamine","serotonin","norepinephrine","melatonin"]

SAMPLE_JSON = [
    {"profile":{"emotion":{"type":"E18"}},"talk":{"content":{"HS01":"일은 왜 해도 해도 끝이 없을까? 화가 난다.", "SS01":"많이 힘드시겠어요."}}},
    {"profile":{"emotion":{"type":"E66"}},"talk":{"content":{"HS01":"요즘 직장생활이 너무 편하고 좋은 것 같아!", "SS01":"복지가 좋아서 마음이 편해."}}},
    {"profile":{"emotion":{"type":"E35"}},"talk":{"content":{"HS01":"면접에서 부모님 직업 질문이 나와서 당혹스러웠어.", "SS01":"무척 놀라셨겠어요."}}},
    {"profile":{"emotion":{"type":"E37"}},"talk":{"content":{"HS01":"졸업반이라 취업 걱정은 되지만 너무 불안해하긴 싫어.", "SS01":"느긋한 태도가 낫다고도 생각해."}}},
]

def load_json_any(path: pathlib.Path):
    s = path.read_text(encoding="utf-8").strip()
    if s.startswith("["):
        return json.loads(s)
    rows = []
    for line in s.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows

def flatten_for_classification(items, min_char=3, use_hs=True, use_ss=False):
    X, y = [], []
    for ex in items:
        try:
            emo_type = ex["profile"]["emotion"]["type"]
            content = ex["talk"]["content"]
            for k, v in content.items():
                if not isinstance(v, str):
                    continue
                s = v.strip()
                if not s or len(s) < min_char:
                    continue
                if k.startswith("HS") and not use_hs:
                    continue
                if k.startswith("SS") and not use_ss:
                    continue
                X.append(s); y.append(emo_type)
        except Exception:
            continue
    return X, y

def build_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3,5),
            min_df=5,
            max_features=300_000,
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            multi_class="multinomial",
            class_weight="balanced",
            max_iter=400,
            C=2.0,
            solver="lbfgs"
        ))
    ])

def map_vec(labels):
    return np.asarray([EMOTION_VEC.get(l, EMO_DEFAULT) for l in labels], dtype=float)

_emovec_pipe = None  # lazy global

def ensure_emovec_model():
    global _emovec_pipe
    if _emovec_pipe is not None:
        return _emovec_pipe

    if MODEL_PATH.exists():
        _emovec_pipe = load(MODEL_PATH)
        return _emovec_pipe

    # 없으면 학습
    if DATA_PATH.exists():
        items = load_json_any(DATA_PATH); used_path = str(DATA_PATH)
    else:
        items = SAMPLE_JSON; used_path = "(embedded SAMPLE_JSON)"

    X, y = flatten_for_classification(items, min_char=3, use_hs=True, use_ss=USE_SS)
    if not X:
        print("No samples parsed. Check DATA_PATH or format.", file=sys.stderr); sys.exit(2)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe = build_model()
    pipe.fit(Xtr, ytr)
    pred_lbl = pipe.predict(Xte)

    acc = accuracy_score(yte, pred_lbl)
    f1m = f1_score(yte, pred_lbl, average="macro")
    mae_vec = mean_absolute_error(map_vec(yte), map_vec(pred_lbl))

    dump(pipe, MODEL_PATH)
    report = {
        "mode": "classification->vector_map",
        "data_used": used_path,
        "use_ss": USE_SS,
        "n_train": len(Xtr), "n_test": len(Xte),
        "acc": float(acc), "f1_macro": float(f1m),
        "mae_after_mapping": float(mae_vec),
        "keys": KEYS
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    _emovec_pipe = pipe
    return _emovec_pipe

def text_to_vec(text: str) -> Tuple[float, float, float, float]:
    """
    IPT 문자열 → 감정코드 → 4D 감정벡터(Vec4)
    """
    pipe = ensure_emovec_model()
    lbl = pipe.predict([text])[0]
    vec = EMOTION_VEC.get(lbl, EMO_DEFAULT)
    return float(vec[0]), float(vec[1]), float(vec[2]), float(vec[3])

# ============================================================
# 2) 감정 네트워크
# ============================================================

Vec4 = Tuple[float, float, float, float]  # (dopamine, serotonin, norepinephrine, melatonin)
CHEM_KEYS = ("dopamine", "serotonin", "norepinephrine", "melatonin")

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def v_clamp(v: Vec4, lo: float = 0.0, hi: float = 1.0) -> Vec4:
    return tuple(clamp(x, lo, hi) for x in v)  # type: ignore

def v_mean_weighted(vs: List[Vec4], ws: List[float]) -> Vec4:
    # Weighted mean per component with small epsilon for stability
    eps = 1e-9
    sW = max(sum(ws), eps)
    comps = []
    for j in range(4):
        num = sum(ws[i] * vs[i][j] for i in range(len(vs)))
        comps.append(num / sW)
    return v_clamp(tuple(comps))  # type: ignore

# ---------------- Logging ----------------
DEBUG = True
LOG_TO_FILE = True
LOG_FILE = "emovec_net_log.txt"

def fmt_vec(v: Vec4) -> str:
    return f"D={v[0]:.3f} S={v[1]:.3f} NE={v[2]:.3f} M={v[3]:.3f}"

def log(msg: str):
    print(msg)
    if LOG_TO_FILE:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

# ---------------- Data structures ----------------

@dataclass
class IPT:
    id: int  # 1..n (larger == more recent)
    text: str

@dataclass
class DataBox:
    K: float
    V: Vec4
    ipt_list: List[IPT] = field(default_factory=list)
    trace: List[str] = field(default_factory=list)  # 방문한 뉴런 이름

# ==========================
# Edges & Neurons
# ==========================

@dataclass(eq=False)
class Neuron:
    name: str
    kind: str  # "exc", "inh", "reg"
    threshold: float = 0.5
    memory_threshold: float = 0.7
    W: float = 1.0  # neuron-level weight (used for K update at send)

    # runtime state
    inbox_limit: int = 256
    _inbox: List[Tuple[DataBox, "Edge"]] = field(default_factory=list, init=False)
    ipt_stack: List[IPT] = field(default_factory=list, init=False)  # LIFO
    outgoing: List["Edge"] = field(default_factory=list, init=False)
    incoming: List["Edge"] = field(default_factory=list, init=False)

    # temporary toggles
    off_ticks: int = 0  # if >0, neuron is temporarily off

    # hyperparams for specific ops
    alpha_exc: float = 0.8     # excitatory scaling vs K
    beta_inh: float = 0.7      # inhibitory scaling vs K (balanced)
    deltaW_scale: float = 0.05 # global scale for ΔW magnitude

    # thresholds for "많음" heuristic
    high_serotonin: float = 0.6
    high_dopamine: float = 0.6

    def connect_to(self, other: "Neuron") -> "Edge":
        e = Edge(src=self, dst=other)
        self.outgoing.append(e)
        other.incoming.append(e)
        return e

    # =============== Core Processing Cycle ===============
    def tick(self, net: "Network") -> None:
        if self.off_ticks > 0:
            if DEBUG:
                log(f"[SKIP] {self.name} off_ticks={self.off_ticks}")
            self.off_ticks -= 1
            self._inbox.clear()
            return

        if not self._inbox:
            return

        # --- 1) Aggregate inputs ---
        Vs: List[Vec4] = []
        Ws: List[float] = []
        all_ipts: List[IPT] = []

        for (box, edge) in self._inbox[: self.inbox_limit]:
            Vs.append(box.V)
            Ws.append(edge.src.W)  # sender W
            all_ipts.extend(box.ipt_list)
            if DEBUG:
                log(f"[IN]  {self.name} <= {edge.src.name} | K={box.K:.3f} thr={self.threshold:.3f} "
                    f"| Wsrc={edge.src.W:.3f} | {fmt_vec(box.V)} | IPTs={[i.id for i in box.ipt_list]}")

        V_in = v_mean_weighted(Vs, Ws) if Vs else (0.5, 0.5, 0.5, 0.5)
        IPT_in = self._merge_ipt_lists(all_ipts)

        # trace 병합
        trace_in: List[str] = []
        for (box, _) in self._inbox:
            for n in box.trace:
                if n not in trace_in:
                    trace_in.append(n)
        if DEBUG:
            log(f"[MERGE] {self.name} V_in={fmt_vec(V_in)} IPT_in={[i.id for i in IPT_in]} trace_in={trace_in}")

        # Memory rule
        if any(box.K > self.memory_threshold for (box, _) in self._inbox):
            if DEBUG:
                log(f"[MEM]  {self.name} push to stack: {[i.id for i in IPT_in]}")
            for ipt in IPT_in[:]:
                self._push_ipt(ipt)

        IPT_travel = self._merge_ipt_lists(IPT_in + self.ipt_stack)
        trace_travel = list(dict.fromkeys(trace_in + [self.name]))  # 순서 유지하며 자기 자신 추가

        # --- 2) For each incoming DataBox, decide to fire and process ---
        outboxes: List[DataBox] = []
        total_dW = 0.0

        for (box, edge) in self._inbox:
            if box.K <= self.threshold:
                if DEBUG:
                    log(f"[NOFIRE] {self.name} from {edge.src.name} K={box.K:.3f} <= thr={self.threshold:.3f}")
                continue

            if DEBUG:
                log(f"[FIRE] {self.name} from {edge.src.name} K_in={box.K:.3f} thr={self.threshold:.3f} "
                    f"| V_in={fmt_vec(V_in)}")
            K_out, V_out, dW = self._specific_op(box.K, V_in, IPT_travel, net)
            total_dW += dW
            if DEBUG:
                log(f"       specific_op -> K_mid={K_out:.3f} V_out={fmt_vec(V_out)} dW={dW:.3f} W_before={self.W:.3f}")

            # Final K update (K *= self.W)
            K_out = K_out * self.W
            if DEBUG:
                log(f"       K_final={K_out:.3f} (after *W)")
            outboxes.append(DataBox(K=K_out,
                                    V=v_clamp(V_out),
                                    ipt_list=IPT_travel,
                                    trace=trace_travel))

        # --- 3) Plasticity & structural change ---
        if total_dW != 0.0:
            if DEBUG:
                log(f"[PLAS] {self.name} total dW={total_dW:.3f}")
            self._apply_plasticity(total_dW, net)

        # --- 4) Send ---
        for ob in outboxes:
            for e in self.outgoing:
                e.send(ob)

        self._inbox.clear()

    # =============== Helpers ===============
    def _push_ipt(self, ipt: IPT) -> None:
        if all(x.id != ipt.id for x in self.ipt_stack):
            self.ipt_stack.append(ipt)
            self.ipt_stack.sort(key=lambda x: x.id, reverse=True)

    @staticmethod
    def _merge_ipt_lists(ipts: Iterable[IPT], limit: int = 512) -> List[IPT]:
        seen = set()
        uniq: List[IPT] = []
        for ipt in sorted(ipts, key=lambda x: x.id, reverse=True):
            if ipt.id in seen:
                continue
            seen.add(ipt.id)
            uniq.append(ipt)
            if len(uniq) >= limit:
                break
        return uniq

    def _apply_plasticity(self, dW_total: float, net: "Network") -> None:
        W_before = self.W
        self.W = clamp(self.W + dW_total * self.deltaW_scale, 0.0, 3.0)
        if DEBUG:
            log(f"       W update: {W_before:.3f} -> {self.W:.3f}")

        magnitude = abs(dW_total)
        steps = int(magnitude * 2)

        if dW_total > 0:
            candidates = [n for n in net.neurons
                          if n is not self and all(e.dst is not n for e in self.outgoing)]
            random.shuffle(candidates)
            added = 0
            for n in candidates[:steps]:
                self.connect_to(n)
                added += 1
            if DEBUG:
                log(f"       edges +{added}")
        elif dW_total < 0:
            random.shuffle(self.outgoing)
            removed = 0
            for _ in range(min(steps, len(self.outgoing))):
                e = self.outgoing.pop()
                e.dst.incoming.remove(e)
                removed += 1
            if DEBUG:
                log(f"       edges -{removed}")

    # =============== Specific Ops by kind ===============
    def _specific_op(self, K_in: float, V_in: Vec4, IPT_list: List[IPT],
                     net: "Network") -> Tuple[float, Vec4, float]:
        D, S, NE, M = V_in
        dW = 0.0

        if self.kind == "exc":
            # 감정 극단화
            factor = 1.0 + self.alpha_exc * K_in
            V_out = tuple(0.5 + (v - 0.5) * factor for v in V_in)  # type: ignore
            V_out = v_clamp(V_out)
            # ΔW ∝ (D + NE) - 1
            dW = (D + NE) - 1.0
            return K_in, V_out, dW

        if self.kind == "inh":
            # 감정 평준화
            shrink = clamp(self.beta_inh * K_in, 0.0, 0.99)
            V_out = tuple(0.5 + (v - 0.5) * (1.0 - shrink) for v in V_in)  # type: ignore
            V_out = v_clamp(V_out)

            # 세로토닌/도파민 기반 확률적 가중치 조정
            p_up = 0.5
            if S >= self.high_serotonin:
                p_up = 0.9
            elif D >= self.high_dopamine:
                p_up = 0.7
            step = 1.0 if random.random() < p_up else -1.0
            dW = 0.2 * step
            return K_in, V_out, dW

        if self.kind == "reg":
            # 임계값 조절 (세로토닌)
            if DEBUG:
                log(f"       [REG] pre: base_thr={net.base_threshold:.3f} "
                    f"global_thr={net.global_threshold:.3f} S={S:.3f} NE={NE:.3f} M={M:.3f}")
            net.global_threshold = clamp(
                net.base_threshold * (1.0 + net.gamma_thresh * (S - 0.5)),
                0.05, 2.0
            )
            if DEBUG:
                log(f"       [REG] global_thr -> {net.global_threshold:.3f}")
            self.threshold = 0.8 * self.threshold + 0.2 * net.global_threshold
            if DEBUG:
                log(f"       [REG] {self.name}.thr -> {self.threshold:.3f}")

            # 멜라토닌 → 드롭아웃
            off_ratio = clamp(M * net.melatonin_drop_scale, 0.0, 0.8)
            self._apply_dropout(net, ratio=off_ratio, ticks=1)
            if DEBUG and off_ratio > 0:
                log(f"       [REG] dropout ratio={off_ratio:.2f}")

            # 노르에피네프린 → 억제 off, 흥분 on
            toggle_ratio = clamp(NE * net.ne_toggle_scale, 0.0, 0.8)
            self._apply_ne_toggle(net, ratio=toggle_ratio, ticks=1)
            if DEBUG and toggle_ratio > 0:
                log(f"       [REG] NE toggle ratio={toggle_ratio:.2f}")

            V_out = V_in
            dW = 0.0
            return K_in, V_out, dW

        # default passthrough
        return K_in, V_in, 0.0

    # --- Regulatory helpers ---
    def _apply_dropout(self, net: "Network", ratio: float, ticks: int = 1) -> None:
        if ratio <= 0:
            return
        candidates = [n for n in net.neurons if n is not self]
        random.shuffle(candidates)
        k = int(len(candidates) * ratio)
        for n in candidates[:k]:
            n.off_ticks = max(n.off_ticks, ticks)
            if DEBUG:
                log(f"         [OFF] {n.name} for {n.off_ticks} ticks")

    def _apply_ne_toggle(self, net: "Network", ratio: float, ticks: int = 1) -> None:
        if ratio <= 0:
            return
        inhibs = [n for n in net.neurons if n.kind == "inh"]
        excts = [n for n in net.neurons if n.kind == "exc"]
        random.shuffle(inhibs); random.shuffle(excts)
        ki = int(len(inhibs) * ratio)
        ke = int(len(excts) * ratio)
        for n in inhibs[:ki]:
            n.off_ticks = max(n.off_ticks, ticks)
            if DEBUG:
                log(f"         [TOGGLE] INH off {n.name}")
        for n in excts[:ke]:
            n.off_ticks = 0
            if DEBUG:
                log(f"         [TOGGLE] EXC on  {n.name}")

@dataclass
class Edge:
    src: Neuron
    dst: Neuron
    W: float = 1.0  # synaptic weight

    def send(self, box: DataBox) -> None:
        if DEBUG:
            log(f"  [SEND] {self.src.name} -> {self.dst.name} | K={box.K:.3f} | {fmt_vec(box.V)} "
                f"| IPTs={[i.id for i in box.ipt_list]}")
        self.dst._inbox.append((box, self))

# ==========================
# Network Container
# ==========================

@dataclass
class Network:
    neurons: List[Neuron]
    base_threshold: float = 0.5
    global_threshold: float = 0.5

    melatonin_drop_scale: float = 0.6
    ne_toggle_scale: float = 0.6
    gamma_thresh: float = 0.8

    # runtime route info
    terminal: Optional[Neuron] = None
    arrived: bool = False
    arrival_box: Optional[DataBox] = None

    def wire_all_to_all(self, p: float = 0.3, seed: Optional[int] = None) -> None:
        rng = random.Random(seed)
        for a in self.neurons:
            for b in self.neurons:
                if a is b: continue
                if rng.random() < p:
                    a.connect_to(b)

    def reset_inboxes(self) -> None:
        for n in self.neurons:
            n._inbox.clear()

    def tick(self) -> None:
        for n in self.neurons:
            n.tick(self)

    def inject(self, target: Neuron, box: DataBox) -> None:
        dummy = Neuron(name="__inject__", kind="exc", W=1.0)
        edge = Edge(src=dummy, dst=target, W=1.0)
        target._inbox.append((box, edge))

# ============================================================
# 3) 10-뉴런 랜덤 네트워크 + 경로 보장
# ============================================================

def summarize_topology(net: Network) -> None:
    log("\n--- TOPOLOGY SUMMARY ---")
    log(f"neurons: {len(net.neurons)} (base_thr={net.base_threshold:.2f}, "
        f"global_thr={net.global_threshold:.2f})")
    for n in net.neurons:
        log(f"- {n.name:>3} kind={n.kind} "
            f"deg_out={len(n.outgoing)} deg_in={len(n.incoming)} "
            f"W={n.W:.2f} thr={n.threshold:.2f}")

def adj_list(net: Network):
    return {n: [e.dst for e in n.outgoing] for n in net.neurons}

def reachable(net: Network, s: Neuron, t: Neuron) -> bool:
    g = adj_list(net)
    q = deque([s])
    seen = {s}
    while q:
        u = q.popleft()
        if u is t:
            return True
        for v in g[u]:
            if v not in seen:
                seen.add(v)
                q.append(v)
    return False

def ensure_path(net: Network, s: Neuron, t: Neuron, seed: int = 0) -> None:
    if reachable(net, s, t):
        return
    rnd = random.Random(seed)
    mids = [n for n in net.neurons if n not in (s, t)]
    rnd.shuffle(mids)
    chain = [s] + mids[: rnd.randint(2, max(2, len(mids)))] + [t]
    for a, b in zip(chain, chain[1:]):
        if all(e.dst is not b for e in a.outgoing):
            a.connect_to(b)

def build_10_neuron_net(seed: int = 123) -> Tuple[Network, Neuron, Neuron]:
    rnd = random.Random(seed)
    kinds = ["exc"] * 4 + ["inh"] * 3 + ["reg"] * 3
    rnd.shuffle(kinds)
    neurons = [Neuron(name=f"N{i}", kind=k) for i, k in enumerate(kinds, start=1)]
    net = Network(neurons=neurons)
    net.wire_all_to_all(p=0.3, seed=seed)

    start, end = rnd.sample(neurons, 2)
    ensure_path(net, start, end, seed=seed)
    summarize_topology(net)

    net.terminal = end
    net.arrived = False
    net.arrival_box = None
    return net, start, end

def inject_and_run_until_arrival(start_text: str,
                                 max_ticks: int = 50,
                                 seed: int = 999) -> None:
    """
    start_text: IPT 문장
    - start_text를 감정벡터로 변환해서 시작 뉴런에 주입
    - 10개 뉴런 랜덤 네트워크에서 start→end 도달할 때까지 tick
    """
    # 로그 파일 초기화
    if LOG_TO_FILE:
        pathlib.Path(LOG_FILE).write_text("", encoding="utf-8")

    log(f"[INFO] start_text = {start_text}")
    V0 = text_to_vec(start_text)
    log(f"[INFO] start_vec = {fmt_vec(V0)}  (from emotion classifier)")

    rnd = random.Random(seed)
    net, s, t = build_10_neuron_net(seed=seed)
    log(f"\n[ROUTE] start={s.name} -> end={t.name}")

    box = DataBox(K=0.75, V=V0, ipt_list=[IPT(id=1, text=start_text)], trace=["__inject__"])
    net.inject(target=s, box=box)

    for tix in range(1, max_ticks + 1):
        log(f"\n=== TICK {tix} ===")
        net.tick()
        log(f"global_threshold={net.global_threshold:.3f}")
        for n in net.neurons:
            log(f"{n.name} kind={n.kind} W={n.W:.3f} thr={n.threshold:.3f} "
                f"off={n.off_ticks} inbox={len(n._inbox)}")

        terminal: Neuron = net.terminal  # type: ignore
        if terminal and terminal._inbox:
            net.arrived = True
            net.arrival_box = terminal._inbox[0][0]
            log(f"\n[ARRIVED] at {terminal.name} on tick {tix}")
            log(f"DataBox: K={net.arrival_box.K:.3f}, {fmt_vec(net.arrival_box.V)}, "
                f"IPTs={[ipt.id for ipt in net.arrival_box.ipt_list]}")
            log(f"Trace: {' -> '.join(net.arrival_box.trace)}")
            break
    else:
        log("\n[STOP] max_ticks reached without arrival.")

# ============================================================
# main
# ============================================================

if __name__ == "__main__":
    # 여기서 테스트할 IPT 문장 바꿔가면서 실험하면 됨
    inject_and_run_until_arrival(
        start_text="요즘 회사 생활이 편하고 좋아.",
        max_ticks=50,
        seed=2025
    )
