# -*- coding: utf-8 -*-
"""
[Neuro-Chatbot: SPEED & MAZE EDITION - DRAMATIC EMOTION VERSION]

- 감정 변화가 더 크게 일어나도록 튜닝
- LLM 프롬프트에 '더 직설적/거칠게 말해도 됨' 지침 추가
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import sys, random, csv
from collections import deque

import matplotlib.pyplot as plt  # 시각화용

# ============================================================
# [PART 1] Manual Input Bridge
# ============================================================
Vec4 = Tuple[float, float, float, float]

def get_manual_vector(text: str) -> Vec4:
    print("\n" + "="*60)
    print("🤖 [1단계: 분석 요청] 아래를 복사해서 LLM에게 물어보세요:")
    print("-" * 60)
    print("당신은 인간과 대화 중인 '인공 생명체'입니다.")
    print("상대방이 아래와 같은 말을 했을 때, 당신의 뇌에서 어떤 호르몬이 분비될지 0.0~1.0으로 수치화하세요.")
    print("주의: 문장의 감정이 아니라, 그 말을 들은 '당신의 기분 변화'입니다.\n")
    print("1. 도파민 (D): 칭찬, 재미, 기대감, 보상 (기분 좋음)")
    print("2. 세로토닌 (S): 안도, 공감, 이해, 차분함 (편안함)")
    print("3. 노르에피네프린 (NE): 공격, 위협, 짜증, 긴장 (스트레스)")
    print("4. 멜라토닌 (M): 실망, 상처, 무시, 지루함 (우울/회피)\n")
    print(f'상대방 입력: "{text}"\n')
    print("출력 형식: [D, S, NE, M] 숫자 4개만 공백으로 구분해서 출력해.")
    print("-" * 60)
    
    while True:
        try:
            raw = input("📝 [2단계: 수치 주입] (예: 0.9 0.8 0.1 0.1) > ").strip()
            clean_raw = raw.replace(',', ' ').replace('[', ' ').replace(']', ' ')
            parts = clean_raw.split()
            if len(parts) >= 4:
                vec = tuple(float(p) for p in parts[:4])
                vec = tuple(max(0.0, min(1.0, v)) for v in vec)
                return vec  # type: ignore
            print("⚠️ 숫자 4개가 필요합니다.")
        except ValueError:
            print("⚠️ 숫자가 아닙니다.")

# ============================================================
# [PART 2] SNN Core
# ============================================================
def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

@dataclass
class DataBox:
    __slots__ = ['K', 'V', 'trace', 'ipt_list']
    K: float            # 막전위
    V: Vec4             # 감정 벡터 (D, S, NE, M)
    trace: List[str]    # 경로 추적
    ipt_list: List[Tuple[int, str]]  # (id, text) 리스트

@dataclass
class Edge:
    __slots__ = ['src', 'dst', 'W']
    src: "Neuron"
    dst: "Neuron"
    W: float

    def send(self, box: DataBox) -> None:
        self.dst._next_inbox.append((box, self))

def merge_ipt_lists(list_of_lists: List[List[Tuple[int, str]]],
                    max_keep: int = 20) -> List[Tuple[int, str]]:
    merged = {}
    for L in list_of_lists:
        for (id_, txt) in L:
            merged[id_] = txt
    items = sorted(merged.items(), key=lambda x: -x[0])
    return items[:max_keep]

@dataclass(eq=False)
class Neuron:
    __slots__ = [
        'name', 'kind', 'threshold', 'W',
        '_inbox', '_next_inbox', 'outgoing', 'incoming',
        'off_ticks', 'alpha_exc', 'beta_inh', 'fatigue', 'base_threshold',
        'memory_threshold', 'IPT_stack'
    ]
    
    name: str
    kind: str
    threshold: float
    W: float
    _inbox: deque
    _next_inbox: deque
    outgoing: List[Edge]
    incoming: List[Edge]
    off_ticks: int
    alpha_exc: float
    beta_inh: float
    fatigue: float
    base_threshold: float
    memory_threshold: float
    IPT_stack: List[Tuple[int, str]]

    def __init__(self, name: str, kind: str):
        self.name = name
        self.kind = kind
        self.base_threshold = random.uniform(0.1, 0.3)
        self.threshold = self.base_threshold
        self.W = random.uniform(0.8, 1.2)
        self._inbox = deque()
        self._next_inbox = deque()
        self.outgoing = []
        self.incoming = []
        self.off_ticks = 0
        self.alpha_exc = 1.2
        self.beta_inh = 0.5
        self.fatigue = 0.0

        self.memory_threshold = 0.7
        self.IPT_stack = []

    def connect_to(self, other: "Neuron") -> Edge:
        for e in self.outgoing:
            if e.dst is other:
                return e
        e = Edge(src=self, dst=other, W=1.0)
        self.outgoing.append(e)
        other.incoming.append(e)
        return e

    def swap_buffer(self):
        if self._next_inbox:
            self._inbox.extend(self._next_inbox)
            self._next_inbox.clear()

    def tick(self, net: "Network", global_chem: Vec4) -> None:
        g_D, g_S, g_NE, g_M = global_chem

        if self.fatigue > 0:
            self.fatigue -= 0.01

        current_threshold = (
            self.base_threshold
            + (self.fatigue * 0.2)
            + (g_S * 0.1)
            - (g_D * 0.1)
        )
        self.threshold = clamp(current_threshold, 0.05, 0.9)

        if g_M > 0.7 and random.random() < g_M * 0.1:
            self.off_ticks = max(self.off_ticks, 1)

        if self.off_ticks > 0:
            self.off_ticks -= 1
            self._inbox.clear()
            return

        if not self._inbox:
            return

        MAX_INBOX = 16
        if len(self._inbox) > MAX_INBOX:
            sorted_inbox = sorted(self._inbox, key=lambda be: be[0].K, reverse=True)
            self._inbox = deque(sorted_inbox[:MAX_INBOX])

        Vs: List[Vec4] = []
        Ws: List[float] = []
        ipt_lists: List[List[Tuple[int, str]]] = []

        for box, edge in self._inbox:
            Vs.append(box.V)
            Ws.append(edge.src.W)
            ipt_lists.append(box.ipt_list)

        if not Vs:
            V_in: Vec4 = (0.5, 0.5, 0.5, 0.5)
        else:
            sW = sum(Ws) + 1e-9
            V_in = tuple(
                sum(Ws[i] * Vs[i][j] for i in range(len(Vs))) / sW
                for j in range(4)
            )  # type: ignore

        ipt_in = merge_ipt_lists(ipt_lists, max_keep=20)

        if any(box.K > self.memory_threshold for box, _ in self._inbox):
            self.IPT_stack = merge_ipt_lists(
                [ipt_in, self.IPT_stack],
                max_keep=50
            )
            if hasattr(net, "global_ipt_memory"):
                net.global_ipt_memory = merge_ipt_lists(
                    [ipt_in, net.global_ipt_memory],
                    max_keep=80
                )

        ipt_merged = merge_ipt_lists([ipt_in, self.IPT_stack], max_keep=30)

        outboxes: List[DataBox] = []
        total_dW = 0.0
        fired = False

        for box, edge in self._inbox:
            penalty = 0.5 if self.name in box.trace else 1.0
            noise = random.uniform(-0.02, 0.02) * g_NE

            if (box.K * penalty) + noise < self.threshold:
                continue

            trace_travel = box.trace + [self.name]
            if len(trace_travel) > 300:
                continue

            K_out, V_out, dW = self._specific_op(box.K, V_in, global_chem)
            total_dW += dW

            decay = 0.85
            K_out = min(K_out * self.W * penalty * decay, 1.5)

            outboxes.append(
                DataBox(
                    K=K_out,
                    V=tuple(clamp(x) for x in V_out),  # type: ignore
                    trace=trace_travel,
                    ipt_list=ipt_merged
                )
            )
            fired = True

            if self is net.terminal and not net.arrived:
                if len(trace_travel) >= 30:
                    net.arrived = True
                    net.arrival_box = outboxes[-1]

        if total_dW != 0.0:
            self._apply_plasticity(total_dW, net)

        for ob in outboxes:
            for e in self.outgoing:
                e.send(ob)

        self._inbox.clear()

        if fired:
            self.off_ticks = 2
            self.fatigue = min(self.fatigue + 0.05, 1.0)

    def _specific_op(self, K_in: float, V_in: Vec4,
                     g_chem: Vec4) -> Tuple[float, Vec4, float]:
        D, S, NE, M = V_in
        dW = 0.0

        if self.kind == "exc":
            factor = 1.0 + (self.alpha_exc * K_in * 0.2)
            V_out = tuple(0.5 + (v - 0.5) * factor for v in V_in)  # type: ignore
            dW = (D + NE) * 0.3
            return K_in, V_out, dW

        if self.kind == "inh":
            base_shrink = clamp(self.beta_inh * K_in)
            if D > 0.6 and NE < 0.6:
                real_shrink = base_shrink * 0.3
            else:
                real_shrink = base_shrink
            V_out = tuple(0.5 + (v - 0.5) * (1.0 - real_shrink)
                          for v in V_in)  # type: ignore
            dW = 0.1 if S > 0.6 else -0.1
            return K_in, V_out, dW

        if self.kind == "reg":
            self.threshold = clamp(
                self.base_threshold + (S - 0.5) * 0.5,
                0.05, 0.9
            )
            if random.random() < M * 0.6:
                self.off_ticks = max(self.off_ticks, 3)
            K_out = K_in * (1.0 + 0.15 * NE)
            return K_out, V_in, 0.0

        return K_in, V_in, 0.0

    def _apply_plasticity(self, dW_total: float, net: "Network") -> None:
        self.W = clamp(self.W + dW_total * 0.02, 0.6, 1.8)

        if abs(dW_total) < 0.1:
            return

        MAX_OUT_DEG = 20

        if dW_total > 0 and len(self.outgoing) < MAX_OUT_DEG:
            candidates = [
                n for n in net.neurons
                if n is not self and all(e.dst is not n for e in self.outgoing)
            ]
            if candidates:
                n = random.choice(candidates)
                self.connect_to(n)
        elif dW_total < 0 and len(self.outgoing) > 1:
            if random.random() < 0.5:
                e = random.choice(self.outgoing)
                self.outgoing.remove(e)
                if e in e.dst.incoming:
                    e.dst.incoming.remove(e)

@dataclass
class Network:
    neurons: List[Neuron]
    terminal: Optional[Neuron] = None
    arrived: bool = False
    arrival_box: Optional[DataBox] = None
    global_ipt_memory: List[Tuple[int, str]] = field(default_factory=list)

    def wire_randomly(self, p: float = 0.5, seed: int = 42):
        rng = random.Random(seed)
        for a in self.neurons:
            for b in self.neurons:
                if a is b:
                    continue
                if rng.random() < p:
                    a.connect_to(b)

    def tick(self, global_chem: Vec4):
        for n in self.neurons:
            n.swap_buffer()
        for n in self.neurons:
            n.tick(self, global_chem)

    def inject(self, target: Neuron, box: DataBox):
        dummy = Neuron("Input", "exc")
        edge = Edge(src=dummy, dst=target, W=1.0)
        target._next_inbox.append((box, edge))
        self.arrived = False
        self.arrival_box = None

# ============================================================
# [PART 3] Emotion Dynamics
# ============================================================
def setup_brain(n_neurons=160, seed=42):
    rng = random.Random(seed)
    n_exc = n_neurons // 2
    n_inh = n_neurons // 4
    n_reg = n_neurons - n_exc - n_inh
    kinds = ["exc"] * n_exc + ["inh"] * n_inh + ["reg"] * n_reg
    rng.shuffle(kinds)
    neurons = [Neuron(f"N{i}", k) for i, k in enumerate(kinds)]
    net = Network(neurons)
    net.wire_randomly(p=0.05, seed=seed)
    s = neurons[0]
    t = neurons[-1]
    net.terminal = t
    return net, s, t

def mix_emotions(curr: Vec4, prev: Vec4) -> Vec4:
    """
    감정 변화를 더 '드라마틱'하게 만드는 업데이트.
    - curr: 새로 들어온 감정 (target)
    - prev: 이전 기분
    - diff가 클수록 한 번에 더 많이 움직인다.
    """
    mixed: List[float] = []

    max_curr = max(curr)
    global_boost = 1.0 + 0.2 * max(0.0, max_curr - 0.6)  # 강한 자극이면 전체 강화

    for i in range(4):
        diff = curr[i] - prev[i]
        # 차이가 클수록 gain↑ (0.4 ~ 0.9)
        base_gain = 0.4
        intensity = min(1.0, abs(diff) * 1.8 + max(curr[i], prev[i]) * 0.3)
        gain = (base_gain + 0.5 * intensity) * global_boost
        gain = min(gain, 0.95)

        # 복원력은 아주 약하게만
        homeo_gain = 0.02
        homeostasis = (0.5 - prev[i]) * homeo_gain

        val = prev[i] + gain * diff + homeostasis
        mixed.append(clamp(val))

    return tuple(mixed)  # type: ignore

def generate_prompt(user_text: str, vec: Vec4, steps: int) -> str:
    D, S, NE, M = vec
    moods = []

    if D > 0.7:
        moods.append("기대와 즐거움")
    elif D < 0.3:
        moods.append("흥미 저하")

    if S > 0.7:
        moods.append("안정과 신뢰")
    elif S < 0.3:
        moods.append("불안정함")

    if NE > 0.7:
        moods.append("긴장·짜증·방어적 상태")
    elif NE < 0.3:
        moods.append("긴장 완화")

    if M > 0.7:
        moods.append("실망·상처·회피 욕구")
    elif M < 0.3:
        moods.append("가벼움")

    mood_str = ", ".join(moods) if moods else "복합적이고 애매한 상태"
    depth_desc = "깊은 고뇌" if steps > 50 else ("적당한 생각" if steps > 25 else "직관적 반응")

    return f"""
[LLM 페르소나 지시문]
당신은 감정을 느끼는 인공 생명체이다. 
아래의 '뇌 상태 수치'와 '정성 요약'을 최우선으로 반영하여 말투와 단어 선택을 결정하라.

[뇌 상태 수치]
- 도파민 D (보상/기대): {D:.2f}
- 세로토닌 S (안정/신뢰): {S:.2f}
- 노르에피네프린 NE (긴장/짜증): {NE:.2f}
- 멜라토닌 M (실망/우울/회피): {M:.2f}

[정성 요약]
- 전반적인 정서 기조: {mood_str}
- 최근 사고 깊이: {steps}단계의 신경망 경로 → {depth_desc}

[표현 강도 가이드]
1. NE와 M이 높을수록 말투가 더 직설적이고, 차갑고, 거칠어져도 좋다.
2. 상대를 똑바로 찌르는 말, 서운함이 드러나는 말, 상처받은 티가 나는 반응은 허용된다.

[연기 지침]
1. D, S가 높을수록 말투를 따뜻하고 여유 있게, 상대를 존중하는 방향으로 가져갈 것.
2. NE, M이 높을수록 문장은 짧아지고, 방어적이거나 냉담하거나 무기력한 뉘앙스를 섞을 것.
3. 너무 논리적으로 완벽하려고 하지 말고, 감정의 흔들림·모순·머뭇거림이 드러나도 괜찮다.
4. 속마음을 전부 설명하지 말고, 일부는 숨기거나 돌려 말하면서 여운을 남길 것.
5. 인간 대화처럼, 상대의 말에 상처받거나 안도하는 흐름이 자연스럽게 이어지도록 반응할 것.
6. 인간의 대화처럼 자연스럽게 말할것
7. 자신의 상태 수치를 설명하려 하지 말것
8. 부자연스러운 언행은 하지 말것

[입력 문장]
"{user_text}"

위 정보를 모두 반영하여, 당신의 현재 감정 상태에 어울리는 한 번의 답변만 생성하라.
"""

# ============================================================
# [PART 4] Visualization Helpers
# ============================================================
def plot_mood_trajectories(mood_history: List[Vec4],
                           raw_history: List[Vec4],
                           brain_history: List[Vec4]):
    if not mood_history:
        print("⚠️ 시각화할 감정 데이터가 없습니다.")
        return

    turns = list(range(1, len(mood_history) + 1))

    def split_vecs(history: List[Vec4]):
        D = [v[0] for v in history]
        S = [v[1] for v in history]
        NE = [v[2] for v in history]
        M = [v[3] for v in history]
        return D, S, NE, M

    D_m, S_m, NE_m, M_m = split_vecs(mood_history)
    D_r, S_r, NE_r, M_r = split_vecs(raw_history)
    D_b, S_b, NE_b, M_b = split_vecs(brain_history)

    plt.figure(figsize=(10, 8))

    plt.subplot(3, 1, 1)
    plt.plot(turns, D_m, marker='o', label='D (mood)')
    plt.plot(turns, S_m, marker='o', label='S (mood)')
    plt.plot(turns, NE_m, marker='o', label='NE (mood)')
    plt.plot(turns, M_m, marker='o', label='M (mood)')
    plt.ylim(0, 1)
    plt.ylabel("current_mood")
    plt.title("Emotional Trajectory (current_mood)")
    plt.legend(loc='best')

    plt.subplot(3, 1, 2)
    plt.plot(turns, D_r, marker='.', linestyle='--', label='D (raw)')
    plt.plot(turns, S_r, marker='.', linestyle='--', label='S (raw)')
    plt.plot(turns, NE_r, marker='.', linestyle='--', label='NE (raw)')
    plt.plot(turns, M_r, marker='.', linestyle='--', label='M (raw)')
    plt.ylim(0, 1)
    plt.ylabel("raw_vec")
    plt.title("Input Emotion (raw_vec)")
    plt.legend(loc='best')

    plt.subplot(3, 1, 3)
    plt.plot(turns, D_b, marker='.', linestyle='--', label='D (brain)')
    plt.plot(turns, S_b, marker='.', linestyle='--', label='S (brain)')
    plt.plot(turns, NE_b, marker='.', linestyle='--', label='NE (brain)')
    plt.plot(turns, M_b, marker='.', linestyle='--', label='M (brain)')
    plt.ylim(0, 1)
    plt.xlabel("turn")
    plt.ylabel("brain_vec")
    plt.title("SNN Output Emotion (brain_vec)")
    plt.legend(loc='best')

    plt.tight_layout()
    plt.show()

def plot_flow_over_ticks(flow_ticks: List[int],
                         flow_active_counts: List[int]):
    if not flow_ticks:
        print("⚠️ 시각화할 흐름 데이터가 없습니다.")
        return

    plt.figure(figsize=(8, 4))
    plt.plot(flow_ticks, flow_active_counts, marker='o')
    plt.xlabel("Tick")
    plt.ylabel("Active Neurons")
    plt.title("Network Activity Over Ticks (Last Turn)")
    plt.tight_layout()
    plt.show()

def export_logs_to_csv(filename: str,
                       mood_history: List[Vec4],
                       raw_history: List[Vec4],
                       brain_history: List[Vec4],
                       steps_history: List[int],
                       text_history: List[str]):
    if not mood_history:
        print("⚠️ 저장할 데이터가 없습니다.")
        return

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "turn",
            "user_text",
            "raw_D", "raw_S", "raw_NE", "raw_M",
            "brain_D", "brain_S", "brain_NE", "brain_M",
            "mood_D", "mood_S", "mood_NE", "mood_M",
            "steps"
        ])
        for i in range(len(mood_history)):
            rD, rS, rNE, rM = raw_history[i]
            bD, bS, bNE, bM = brain_history[i]
            mD, mS, mNE, mM = mood_history[i]
            writer.writerow([
                i + 1,
                text_history[i],
                f"{rD:.3f}", f"{rS:.3f}", f"{rNE:.3f}", f"{rM:.3f}",
                f"{bD:.3f}", f"{bS:.3f}", f"{bNE:.3f}", f"{bM:.3f}",
                f"{mD:.3f}", f"{mS:.3f}", f"{mNE:.3f}", f"{mM:.3f}",
                steps_history[i],
            ])

    print(f"✅ 로그가 '{filename}' 파일에 저장되었다.")

# ============================================================
# [PART 5] Main Loop
# ============================================================
def main():
    print(f"\n🧠 [Neuro-Chatbot: SPEED & MAZE EDITION]")
    print("   - 160 Neurons / Sparse Connectivity (p=0.05).")
    print("   - No Artificial Delays. Max Speed.\n")
    
    net, s_node, t_node = setup_brain(n_neurons=160, seed=777)
    current_mood: Vec4 = (0.5, 0.5, 0.5, 0.5)
    ipt_id = 0

    mood_history: List[Vec4] = []
    raw_history: List[Vec4] = []
    brain_history: List[Vec4] = []
    steps_history: List[int] = []
    text_history: List[str] = []

    last_flow_ticks: List[int] = []
    last_flow_actives: List[int] = []

    while True:
        print(f"\n[Brain State] D:{current_mood[0]:.2f} S:{current_mood[1]:.2f} NE:{current_mood[2]:.2f} M:{current_mood[3]:.2f}")
        
        user_input = input("👤 You: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            break
        if not user_input:
            continue
        
        raw_vec = get_manual_vector(user_input)

        chem_turn = mix_emotions(raw_vec, current_mood)

        ipt_id += 1
        k_val = 0.95

        if net.global_ipt_memory:
            base_ipt = merge_ipt_lists(
                [net.global_ipt_memory],
                max_keep=20
            )
        else:
            base_ipt = []

        base_ipt = merge_ipt_lists(
            [base_ipt, [(ipt_id, user_input)]],
            max_keep=30
        )

        box = DataBox(
            K=k_val,
            V=chem_turn,
            trace=["Input"],
            ipt_list=base_ipt
        )
        net.inject(s_node, box)
        
        print("\n   🧠 Simulating...", end="")
        brain_vec = current_mood
        steps_taken = 0

        flow_ticks: List[int] = []
        flow_actives: List[int] = []
        
        for i in range(200):
            net.tick(global_chem=chem_turn)
            
            if i % 5 == 0:
                active_neurons = [n for n in net.neurons if n._next_inbox]
                active_count = len(active_neurons)
                flow_ticks.append(i)
                flow_actives.append(active_count)
                preview = ", ".join(n.name for n in active_neurons[:5])
                print(
                    f"\r   🧠 Tick {i:03d}: Active {active_count:03d} "
                    f"{'█' * (active_count // 5)}  ({preview})",
                    end=""
                )
                sys.stdout.flush()
            
            if net.arrived and net.arrival_box:
                steps_taken = len(net.arrival_box.trace)
                brain_vec = net.arrival_box.V
                # soft-normalize (아예 죽이지 않게, 0.5를 향해 살짝만 끌어당김)
                brain_vec = tuple(
                    clamp(0.5 + (v - 0.5) * 0.5) for v in brain_vec
                )  # type: ignore

                print(
                    f"\n   >>> ✅ SIGNAL ARRIVED at Node {net.terminal.name} "
                    f"(Steps: {steps_taken}) <<<"
                )
                break
        
        else:
            print(f"\n   >>> ❌ SIGNAL LOST (Complex thought process) <<<")
            d, s, ne, m = current_mood
            brain_vec = (d * 0.95, s * 0.95, clamp(ne + 0.05), clamp(m + 0.05))
            steps_taken = 200

        last_flow_ticks = flow_ticks
        last_flow_actives = flow_actives

        pos_raw = (raw_vec[0] + raw_vec[1]) * 0.5
        neg_raw = (raw_vec[2] + raw_vec[3]) * 0.5

        # drama 모드: raw_vec 비중을 크게 (0.7~0.9 근처)
        emotion_strength = max(abs(pos_raw - neg_raw), 0.2)
        w_raw = 0.7 + 0.2 * emotion_strength
        w_raw = clamp(w_raw, 0.7, 0.9)
        w_brain = 1.0 - w_raw

        target_vec: Vec4 = tuple(
            w_raw  * raw_vec[i]   +   w_brain * brain_vec[i]
            for i in range(4)
        )  # type: ignore

        current_mood = mix_emotions(target_vec, current_mood)

        raw_history.append(raw_vec)
        brain_history.append(brain_vec)
        mood_history.append(current_mood)
        steps_history.append(steps_taken)
        text_history.append(user_input)

        print("\n" + "="*50)
        print(generate_prompt(user_input, current_mood, steps_taken))
        print("="*50 + "\n")

    if mood_history:
        try:
            export_logs_to_csv(
                "emotion_log.csv",
                mood_history,
                raw_history,
                brain_history,
                steps_history,
                text_history,
            )
        except Exception as e:
            print("CSV 저장 중 오류:", e)

        try:
            plot_mood_trajectories(
                mood_history,
                raw_history,
                brain_history,
            )
        except Exception as e:
            print("감정 그래프 생성 중 오류:", e)

        try:
            plot_flow_over_ticks(
                last_flow_ticks,
                last_flow_actives,
            )
        except Exception as e:
            print("흐름 그래프 생성 중 오류:", e)

if __name__ == "__main__":
    main()
