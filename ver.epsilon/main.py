from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Iterable
import random
from collections import deque

# ==========================
# Core Data Structures
# ==========================

Vec4 = Tuple[float, float, float, float]  # (dopamine, serotonin, norepinephrine, melatonin)
CHEM_KEYS = ("dopamine", "serotonin", "norepinephrine", "melatonin")

DEBUG = True  # 상세 로그 스위치

def fmt_vec(v: Vec4) -> str:
    return f"D={v[0]:.3f} S={v[1]:.3f} NE={v[2]:.3f} M={v[3]:.3f}"

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

@dataclass
class IPT:
    id: int  # 1..n (larger == more recent)
    text: str

@dataclass
class DataBox:
    K: float
    V: Vec4
    ipt_list: List[IPT] = field(default_factory=list)
    trace: List[str] = field(default_factory=list)  # visited neuron names

# ==========================
# Edges & Neurons
# ==========================

@dataclass
class Edge:
    src: "Neuron"
    dst: "Neuron"

    def send(self, box: DataBox) -> None:
        # Edge sends a reference to box (본 예제에서는 immutable처럼 취급)
        if DEBUG:
            print(f"  [SEND] {self.src.name} -> {self.dst.name} | K={box.K:.3f} | {fmt_vec(box.V)} | IPTs={[i.id for i in box.ipt_list]}")
        self.dst._inbox.append((box, self))

@dataclass(eq=False)
class Neuron:
    name: str
    kind: str  # "exc", "inh", "reg"
    threshold: float = 0.5
    memory_threshold: float = 0.7
    W: float = 1.0  # neuron-level weight (used for K update at send)

    # runtime state
    inbox_limit: int = 256
    _inbox: List[Tuple[DataBox, Edge]] = field(default_factory=list, init=False)
    ipt_stack: List[IPT] = field(default_factory=list, init=False)  # LIFO
    outgoing: List[Edge] = field(default_factory=list, init=False)
    incoming: List[Edge] = field(default_factory=list, init=False)

    # temporary toggles
    off_ticks: int = 0  # if >0, neuron is temporarily off

    # hyperparams for specific ops
    alpha_exc: float = 0.8     # excitatory scaling vs K
    beta_inh: float = 0.7      # inhibitory scaling vs K (balanced)
    deltaW_scale: float = 0.05 # global scale for ΔW magnitude

    # thresholds for "많음" heuristic
    high_serotonin: float = 0.6
    high_dopamine: float = 0.6

    def connect_to(self, other: "Neuron") -> Edge:
        e = Edge(src=self, dst=other)
        self.outgoing.append(e)
        other.incoming.append(e)
        return e

    # =============== Core Processing Cycle ===============
    def tick(self, net: "Network") -> None:
        if self.off_ticks > 0:
            if DEBUG:
                print(f"[SKIP] {self.name} off_ticks={self.off_ticks}")
            self.off_ticks -= 1
            self._inbox.clear()
            return

        if not self._inbox:
            return

        # --- 1) Aggregate inputs (spec rules) ---
        Vs: List[Vec4] = []
        Ws: List[float] = []
        all_ipts: List[IPT] = []

        for (box, edge) in self._inbox[: self.inbox_limit]:
            Vs.append(box.V)
            Ws.append(edge.src.W)  # sender's W for vector merge
            all_ipts.extend(box.ipt_list)
            if DEBUG:
                print(f"[IN]  {self.name} <= {edge.src.name} | K={box.K:.3f} thr={self.threshold:.3f} | Wsrc={edge.src.W:.3f} | {fmt_vec(box.V)} | IPTs={[i.id for i in box.ipt_list]}")

        V_in = v_mean_weighted(Vs, Ws) if Vs else (0.5, 0.5, 0.5, 0.5)
        IPT_in = self._merge_ipt_lists(all_ipts)
        # trace merge
        trace_in: List[str] = []
        for (box, _) in self._inbox:
            for n in box.trace:
                if n not in trace_in:
                    trace_in.append(n)
        if DEBUG:
            print(f"[MERGE] {self.name} V_in={fmt_vec(V_in)} IPT_in={[i.id for i in IPT_in]} trace_in={trace_in}")

        # Memory rule
        if any(box.K > self.memory_threshold for (box, _) in self._inbox):
            if DEBUG:
                print(f"[MEM]  {self.name} push to stack: {[i.id for i in IPT_in]}")
            for ipt in IPT_in[:]:
                self._push_ipt(ipt)

        # Merge stored IPT into traveling list
        IPT_travel = self._merge_ipt_lists(IPT_in + self.ipt_stack)
        trace_travel = list(dict.fromkeys(trace_in + [self.name]))

        # --- 2) For each incoming DataBox, process per-input K ---
        outboxes: List[DataBox] = []
        total_dW = 0.0

        for (box, edge) in self._inbox:
            if box.K <= self.threshold:
                if DEBUG:
                    print(f"[NOFIRE] {self.name} from {edge.src.name} K={box.K:.3f} <= thr={self.threshold:.3f}")
                continue

            if DEBUG:
                print(f"[FIRE] {self.name} from {edge.src.name} K_in={box.K:.3f} thr={self.threshold:.3f} | V_in={fmt_vec(V_in)}")
            K_out, V_out, dW = self._specific_op(box.K, V_in, IPT_travel, net)
            total_dW += dW
            if DEBUG:
                print(f"       specific_op -> K_mid={K_out:.3f} V_out={fmt_vec(V_out)} dW={dW:.3f} W_before={self.W:.3f}")

            # Final K update at sender side before sending forward: K *= self.W
            K_out = K_out * self.W
            if DEBUG:
                print(f"       K_final={K_out:.3f} (after *W)")
            outboxes.append(DataBox(K=K_out, V=v_clamp(V_out), ipt_list=IPT_travel, trace=trace_travel))

        # --- 3) Plasticity & structural change ---
        if total_dW != 0.0:
            if DEBUG:
                print(f"[PLAS] {self.name} total dW={total_dW:.3f}")
            self._apply_plasticity(total_dW, net)

        # --- 4) Send ---
        for ob in outboxes:
            for e in self.outgoing:
                e.send(ob)

        # clear inbox
        self._inbox.clear()

    # =============== Helpers ===============
    def _push_ipt(self, ipt: IPT) -> None:
        if all(x.id != ipt.id for x in self.ipt_stack):
            self.ipt_stack.append(ipt)
            self.ipt_stack.sort(key=lambda x: x.id, reverse=True)  # most-recent-first

    @staticmethod
    def _merge_ipt_lists(ipts: Iterable[IPT], limit: int = 512) -> List[IPT]:
        seen = set()
        uniq = []
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
            print(f"       W update: {W_before:.3f} -> {self.W:.3f}")

        # structural change magnitude -> number of edges to add/remove
        magnitude = abs(dW_total)
        steps = int(magnitude * 2)  # heuristic mapping

        if dW_total > 0:
            candidates = [n for n in net.neurons if n is not self and all(e.dst is not n for e in self.outgoing)]
            random.shuffle(candidates)
            added = 0
            for n in candidates[:steps]:
                self.connect_to(n)
                added += 1
            if DEBUG:
                print(f"       edges +{added}")
        elif dW_total < 0:
            random.shuffle(self.outgoing)
            removed = 0
            for _ in range(min(steps, len(self.outgoing))):
                e = self.outgoing.pop()
                e.dst.incoming.remove(e)
                removed += 1
            if DEBUG:
                print(f"       edges -{removed}")

    # =============== Specific Ops by kind ===============
    def _specific_op(self, K_in: float, V_in: Vec4, IPT_list: List[IPT], net: "Network") -> Tuple[float, Vec4, float]:
        D, S, NE, M = V_in
        dW = 0.0

        if self.kind == "exc":
            # Vector extremization away from 0.5 proportional to K
            factor = 1.0 + self.alpha_exc * K_in
            V_out = tuple(0.5 + (v - 0.5) * factor for v in V_in)  # type: ignore
            V_out = v_clamp(V_out)
            # ΔW ∝ (D + NE)
            dW = (D + NE) - 1.0  # centered near 0 when both ~0.5
            return K_in, V_out, dW

        if self.kind == "inh":
            # Vector normalization toward 0.5 proportional to K
            shrink = clamp(self.beta_inh * K_in, 0.0, 0.99)
            V_out = tuple(0.5 + (v - 0.5) * (1.0 - shrink) for v in V_in)  # type: ignore
            V_out = v_clamp(V_out)
            # Probabilistic W adjustment (S high: 90% up / D high: 70% up)
            p_up = 0.5
            if S >= self.high_serotonin:
                p_up = 0.9
            elif D >= self.high_dopamine:
                p_up = 0.7
            step = 1.0 if random.random() < p_up else -1.0
            dW = 0.2 * step
            return K_in, V_out, dW

        if self.kind == "reg":
            if DEBUG:
                print(f"       [REG] pre: base_thr={net.base_threshold:.3f} global_thr={net.global_threshold:.3f} S={S:.3f} NE={NE:.3f} M={M:.3f}")
            # Threshold modulation by serotonin
            net.global_threshold = clamp(net.base_threshold * (1.0 + net.gamma_thresh * (S - 0.5)), 0.05, 2.0)
            if DEBUG:
                print(f"       [REG] global_thr -> {net.global_threshold:.3f}")
            self.threshold = 0.8 * self.threshold + 0.2 * net.global_threshold
            if DEBUG:
                print(f"       [REG] {self.name}.thr -> {self.threshold:.3f}")

            # Melatonin-driven temporary dropout
            off_ratio = clamp(M * net.melatonin_drop_scale, 0.0, 0.8)
            self._apply_dropout(net, ratio=off_ratio, ticks=1)
            if DEBUG and off_ratio > 0:
                print(f"       [REG] dropout ratio={off_ratio:.2f}")

            # NE-driven toggle
            toggle_ratio = clamp(NE * net.ne_toggle_scale, 0.0, 0.8)
            self._apply_ne_toggle(net, ratio=toggle_ratio, ticks=1)
            if DEBUG and toggle_ratio > 0:
                print(f"       [REG] NE toggle ratio={toggle_ratio:.2f}")

            V_out = V_in
            dW = 0.0
            return K_in, V_out, dW

        return K_in, V_in, 0.0

    # --- Regulatory helpers ---
    def _apply_dropout(self, net: "Network", ratio: float, ticks: int = 1) -> None:
        if ratio <= 0: return
        candidates = [n for n in net.neurons if n is not self]
        random.shuffle(candidates)
        k = int(len(candidates) * ratio)
        for n in candidates[:k]:
            n.off_ticks = max(n.off_ticks, ticks)
            if DEBUG:
                print(f"         [OFF] {n.name} for {n.off_ticks} ticks")

    def _apply_ne_toggle(self, net: "Network", ratio: float, ticks: int = 1) -> None:
        if ratio <= 0: return
        inhibs = [n for n in net.neurons if n.kind == "inh"]
        excts = [n for n in net.neurons if n.kind == "exc"]
        random.shuffle(inhibs); random.shuffle(excts)
        ki = int(len(inhibs) * ratio)
        ke = int(len(excts) * ratio)
        for n in inhibs[:ki]:
            n.off_ticks = max(n.off_ticks, ticks)
            if DEBUG:
                print(f"         [TOGGLE] INH off {n.name}")
        for n in excts[:ke]:
            n.off_ticks = 0
            if DEBUG:
                print(f"         [TOGGLE] EXC on  {n.name}")

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

    # for arrival demo
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
        # One processing step for all neurons
        for n in self.neurons:
            n.tick(self)

    def inject(self, target: Neuron, box: DataBox) -> None:
        # 외부에서 들어오는 입력(가상 소스)
        dummy = Neuron(name="__input__", kind="exc", W=1.0)
        edge = Edge(src=dummy, dst=target)
        target._inbox.append((box, edge))

# ==========================
# IPT -> Vec4 Adapter (placeholder)
# ==========================

def ipt_to_vec4_stub(text: str) -> Vec4:
    # TODO: replace with real call to emovec_autorun_cls.IPT2NTL(text)
    h = abs(hash(text))
    rnd = random.Random(h)
    v = (rnd.random(), rnd.random(), rnd.random(), rnd.random())
    return v_clamp(v)

# ==========================
# Topology / Demo helpers
# ==========================

def summarize_topology(net: Network) -> None:
    print("\n--- TOPOLOGY SUMMARY ---")
    print(f"neurons: {len(net.neurons)} (base_thr={net.base_threshold:.2f}, global_thr={net.global_threshold:.2f})")
    for n in net.neurons:
        print(f"- {n.name:>3} kind={n.kind} deg_out={len(n.outgoing)} deg_in={len(n.incoming)} W={n.W:.2f} thr={n.threshold:.2f}")

def adj_list(net: Network):
    return {n: [e.dst for e in n.outgoing] for n in net.neurons}

def reachable(net: Network, s: Neuron, t: Neuron) -> bool:
    g = adj_list(net)
    q = deque([s]); seen = {s}
    while q:
        u = q.popleft()
        if u is t: return True
        for v in g[u]:
            if v not in seen:
                seen.add(v); q.append(v)
    return False

def ensure_path(net: Network, s: Neuron, t: Neuron, seed: int = 0) -> None:
    if reachable(net, s, t): return
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
    net.terminal = end
    net.arrived = False
    net.arrival_box = None
    summarize_topology(net)
    return net, start, end

def inject_and_run_until_arrival(max_ticks: int = 50, seed: int = 2025) -> None:
    rnd = random.Random(seed)
    net, s, t = build_10_neuron_net(seed=seed)
    print(f"\n[ROUTE] start={s.name} -> end={t.name}")

    # initial box
    V = v_clamp((rnd.random(), rnd.random(), rnd.random(), rnd.random()))
    box = DataBox(K=0.75, V=V, ipt_list=[IPT(id=1, text="start")], trace=["__inject__"])
    net.inject(target=s, box=box)

    for tix in range(1, max_ticks + 1):
        print(f"\n=== TICK {tix} ===")
        net.tick()

        terminal: Neuron = net.terminal  # type: ignore
        if terminal and terminal._inbox:
            net.arrived = True
            net.arrival_box = terminal._inbox[0][0]
            print(f"\n[ARRIVED] at {terminal.name} on tick {tix}")
            print(f"DataBox: K={net.arrival_box.K:.3f}, {fmt_vec(net.arrival_box.V)}, IPTs={[ipt.id for ipt in net.arrival_box.ipt_list]}")
            print(f"Trace: {' -> '.join(net.arrival_box.trace)}")
            return

    print("\n[STOP] max_ticks reached without arrival.")

# ==========================
# Main
# ==========================

if __name__ == "__main__":
    # 10-뉴런 랜덤 라우팅 데모 (도착 시 트레이스 출력)
    inject_and_run_until_arrival(max_ticks=50, seed=2025)
