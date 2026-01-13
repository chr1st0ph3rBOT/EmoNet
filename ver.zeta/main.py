# -*- coding: utf-8 -*-
"""
[Neuro-Chatbot: MANUAL BRIDGE EDITION]
- Input: User manually inputs Emotion Vector (D S NE M) from external LLM.
- Core: Double Buffered SNN + Plasticity + Dopamine Shield.
- No external libraries required (Pure Python).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import sys, time, random
from collections import deque

# ============================================================
# [PART 1] SNN Core (Stable Double-Buffered)
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
        self.threshold = 0.3; self.W = 1.0
        self._inbox = deque(); self._next_inbox = deque()
        self.outgoing = []; self.incoming = []
        self.off_ticks = 0; self.refractory_period = 2
        self.alpha_exc = 1.5; self.beta_inh = 0.6

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
        if self.off_ticks > 0: self.off_ticks -= 1; self._inbox.clear(); return
        if not self._inbox: return

        Vs, Ws = [], []
        for box, edge in self._inbox: Vs.append(box.V); Ws.append(edge.src.W)
        if not Vs: V_in = (0.5,0.5,0.5,0.5)
        else: sW = sum(Ws) + 1e-9; V_in = tuple(sum(Ws[i]*Vs[i][j] for i in range(len(Vs)))/sW for j in range(4)) # type: ignore

        trace_sample = self._inbox[-1][0].trace[-5:] if self._inbox else []
        trace_travel = trace_sample + [self.name]

        if len(self._inbox[-1][0].trace) > 20: self._inbox.clear(); return

        outboxes, total_dW = [], 0.0
        fired = False

        for box, edge in self._inbox:
            penalty = 0.3 if self.name in box.trace else 1.0
            if box.K * penalty < self.threshold: continue
            
            K_out, V_out, dW = self._specific_op(box.K, V_in)
            total_dW += dW
            K_out = min(K_out * self.W * penalty * 0.98, 2.0)
            
            outboxes.append(DataBox(K=K_out, V=tuple(clamp(x) for x in V_out), trace=trace_travel)) # type: ignore
            fired = True

            if self is net.terminal and not net.arrived:
                net.arrived = True; net.arrival_box = outboxes[-1]

        if total_dW != 0.0: self._apply_plasticity(total_dW, net)
        for ob in outboxes:
            for e in self.outgoing: e.send(ob)
        self._inbox.clear()
        
        if fired: self.off_ticks = self.refractory_period

    def _specific_op(self, K_in: float, V_in: Vec4) -> Tuple[float, Vec4, float]:
        D, S, NE, M = V_in; dW = 0.0
        if self.kind == "exc":
            factor = 1.0 + (self.alpha_exc * K_in)
            V_out = tuple(0.5 + (v-0.5)*factor if abs(v-0.5) > 0.05 else v for v in V_in) # type: ignore
            dW = (D + NE) * 0.8; return K_in, V_out, dW
        if self.kind == "inh":
            base_shrink = clamp(self.beta_inh * K_in)
            if D > 0.55 and NE < 0.6: real_shrink = base_shrink * 0.2
            else: real_shrink = base_shrink
            V_out = tuple(0.5 + (v-0.5)*(1.0-real_shrink) for v in V_in) # type: ignore
            dW = 0.1 if S > 0.6 else -0.5; return K_in, V_out, dW
        return K_in, V_in, 0.0

    def _apply_plasticity(self, dW_total: float, net: "Network") -> None:
        self.W = clamp(self.W + dW_total * 0.08)
        steps = int(abs(dW_total) * 2)
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
    neurons: List[Neuron]; terminal: "Neuron" | None = None
    arrived: bool = False; arrival_box: DataBox | None = None
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
        dummy = Neuron("Input", "exc"); edge = Edge(src=dummy, dst=target, W=1.0)
        target._next_inbox.append((box, edge))
        self.arrived = False; self.arrival_box = None

# ============================================================
# [PART 2] Helper & Main
# ============================================================
def setup_brain(n_neurons=50, seed=42):
    rng = random.Random(seed)
    kinds = ["exc"]*25 + ["inh"]*10 + ["reg"]*15
    rng.shuffle(kinds)
    neurons = [Neuron(f"N{i}", k) for i, k in enumerate(kinds)]
    net = Network(neurons)
    net.wire_randomly(p=0.55, seed=seed)
    s, t = rng.sample(neurons, 2)
    net.terminal = t
    return net, s, t

def mix_emotions(curr: Vec4, prev: Vec4, decay: float = 0.3) -> Vec4:
    mixed = []
    for i in range(4):
        val = curr[i]*(1.0-decay) + prev[i]*decay
        mixed.append(val)
    return tuple(mixed) # type: ignore

def get_manual_vector(text: str) -> Vec4:
    """사용자가 직접 LLM 결과값을 입력하는 함수"""
    print("\n" + "="*50)
    print("🤖 [LLM 분석 요청] 아래 프롬프트를 복사해서 ChatGPT/Claude에 붙여넣으세요:")
    print("-" * 50)
    print(f'다음 문장의 감정을 분석해서 4가지 수치(0.0~1.0)만 순서대로 나열해줘 (설명X).')
    print(f'순서: [도파민(쾌락), 세로토닌(안정), 노르에피네프린(스트레스), 멜라토닌(우울)]')
    print(f'문장: "{text}"')
    print("-" * 50)
    
    while True:
        try:
            raw = input("📝 LLM이 알려준 숫자 4개를 입력하세요 (예: 0.1 0.2 0.8 0.1) > ").strip()
            # 콤마 제거 및 공백 기준 분리
            parts = raw.replace(',', ' ').split()
            if len(parts) >= 4:
                vec = tuple(float(p) for p in parts[:4])
                return vec # type: ignore
            else:
                print("⚠️ 숫자 4개가 필요합니다. 다시 입력해주세요.")
        except ValueError:
            print("⚠️ 숫자가 아닙니다. 다시 입력해주세요.")

def generate_persona_prompt(user_text: str, vec: Vec4) -> str:
    D, S, NE, M = vec
    moods = []
    if D>0.7: moods.append("신남/열정")
    elif D<0.3: moods.append("지루함")
    if NE>0.7: moods.append("예민/공격적")
    elif NE<0.3: moods.append("이완됨")
    if M>0.7: moods.append("무기력")
    if S>0.7: moods.append("이성적")
    mood_str = ", ".join(moods) if moods else "미묘함"
    
    return f"""
[LLM 페르소나 지시문]
당신은 '화학적 뇌 시뮬레이션'을 장착한 AI입니다.
현재 뇌 상태: D={D:.2f} S={S:.2f} NE={NE:.2f} M={M:.2f}
결론적 기분: "{mood_str}"
지시: 위 상태에 맞춰 연기하세요. 감정 수치를 직접 언급하지 마세요.
입력: "{user_text}"
"""

def main():
    print(f"\n🧠 [Neuro-Chatbot: MANUAL BRIDGE MODE]")
    print("   - You act as the bridge between LLM and Brain.")
    
    net, s_node, t_node = setup_brain(n_neurons=50, seed=42)
    current_mood = (0.5, 0.5, 0.5, 0.5)

    while True:
        user_input = input("\n👤 You: ").strip()
        if user_input.lower() in ["quit", "exit"]: break
        if not user_input: continue
        
        # 1. 수동 벡터 입력
        raw_vec = get_manual_vector(user_input)
        
        # 2. 기억 혼합
        mixed_vec = mix_emotions(raw_vec, current_mood, decay=0.3)
        k_val = 0.9 
        box = DataBox(K=k_val, V=mixed_vec, trace=["Input"])
        net.inject(s_node, box)
        
        print("   🧠 Thinking", end="", flush=True)
        final_vec = current_mood
        
        # 3. 시뮬레이션
        for i in range(40):
            net.tick()
            if i % 5 == 0: print(".", end="", flush=True)
            
            if net.arrived and net.arrival_box:
                final_vec = net.arrival_box.V
                print(f" DONE! (Step {len(net.arrival_box.trace)})")
                break
            time.sleep(0.005)
        else:
            print(" LOST")
            d, s, ne, m = current_mood
            final_vec = (d*0.9, s*0.9, clamp(ne+0.1), clamp(m+0.1))

        current_mood = final_vec
        
        # 4. 결과 출력
        print("\n" + "="*50)
        print(generate_persona_prompt(user_input, current_mood))
        print("="*50 + "\n")

if __name__ == "__main__":
    main()