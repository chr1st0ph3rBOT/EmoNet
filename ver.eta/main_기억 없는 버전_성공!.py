# -*- coding: utf-8 -*-
"""
[Emotion Brain: Manual Mode]
1. Simulates a 50-neuron biological brain (SNN).
2. Calculates final neurotransmitter levels based on input.
3. Generates a 'System Prompt' for you to copy-paste into ChatGPT/Claude.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import math, random, time

# ============================================================
# 1) SNN Core (뇌세포 및 네트워크 로직)
# ============================================================
Vec4 = Tuple[float, float, float, float] # D, S, NE, M

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

@dataclass
class DataBox:
    K: float
    V: Vec4
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
    kind: str # exc, inh, reg
    threshold: float = 0.5
    W: float = 1.0
    _inbox: List[Tuple[DataBox, Edge]] = field(default_factory=list, init=False)
    outgoing: List[Edge] = field(default_factory=list, init=False)
    incoming: List[Edge] = field(default_factory=list, init=False)
    off_ticks: int = 0
    
    alpha_exc: float = 0.8
    beta_inh: float = 0.7
    deltaW_scale: float = 0.05

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

        # 입력 통합
        Vs, Ws = [], []
        for (box, edge) in self._inbox:
            Vs.append(box.V); Ws.append(edge.src.W)
        
        if not Vs: V_in = (0.5, 0.5, 0.5, 0.5)
        else:
            sW = sum(Ws) + 1e-9
            V_in = tuple(sum(Ws[i]*Vs[i][j] for i in range(len(Vs)))/sW for j in range(4)) # type: ignore

        # 경로 추적
        trace_in = []
        for (box, _) in self._inbox:
            for t in box.trace: 
                if t not in trace_in: trace_in.append(t)
        trace_travel = trace_in + [self.name]

        # 신경 처리
        outboxes, total_dW = [], 0.0
        for (box, edge) in self._inbox:
            if box.K < self.threshold: continue
            
            K_out, V_out, dW = self._specific_op(box.K, V_in)
            total_dW += dW
            
            K_out *= self.W
            outboxes.append(DataBox(K=K_out, V=tuple(clamp(x) for x in V_out), trace=trace_travel)) # type: ignore

            if self is net.terminal and not net.arrived:
                net.arrived = True
                net.arrival_box = outboxes[-1]

        # 가소성
        if total_dW != 0.0: self._apply_plasticity(total_dW, net)
        for ob in outboxes:
            for e in self.outgoing: e.send(ob)
        self._inbox.clear()

    def _specific_op(self, K_in: float, V_in: Vec4) -> Tuple[float, Vec4, float]:
        D, S, NE, M = V_in
        dW = 0.0
        
        # [흥분성] 증폭 + 학습
        if self.kind == "exc":
            factor = 1.0 + (self.alpha_exc * K_in * 1.5)
            V_out_list = []
            for v in V_in:
                dist = v - 0.5
                if abs(dist) < 0.45: val = 0.5 + dist * factor
                else: val = v
                V_out_list.append(val)
            V_out = tuple(V_out_list)
            dW = (D + NE) * 0.8
            return K_in, V_out, dW

        # [억제성] 도파민 쉴드 (기분 좋으면 억제 무시)
        if self.kind == "inh":
            base_shrink = clamp(self.beta_inh * K_in, 0.0, 0.9)
            real_shrink = 0.0 if D > 0.55 else base_shrink
            
            V_out = tuple(0.5 + (v-0.5)*(1.0-real_shrink) for v in V_in) # type: ignore
            dW = 0.1 if S > 0.6 else -0.8 
            return K_in, V_out, dW

        # [조절성] (패스스루)
        return K_in, V_in, 0.0

    def _apply_plasticity(self, dW_total: float, net: "Network") -> None:
        self.W = clamp(self.W + dW_total * self.deltaW_scale, 0.0, 3.0)
        magnitude = abs(dW_total)
        steps = int(magnitude * 2)

        if dW_total > 0: # 연결 생성
            candidates = [n for n in net.neurons if n is not self and all(e.dst is not n for e in self.outgoing)]
            if candidates:
                random.shuffle(candidates)
                for n in candidates[:steps]: self.connect_to(n)

        elif dW_total < 0: # 연결 제거 (Pruning)
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
    
    def wire_randomly(self, p: float = 0.3, seed: int = 42):
        rng = random.Random(seed)
        for a in self.neurons:
            for b in self.neurons:
                if a is b: continue
                if rng.random() < p: a.connect_to(b)

    def tick(self):
        for n in self.neurons: n.tick(self)
    
    def inject(self, target: Neuron, box: DataBox):
        dummy = Neuron("Input", "exc")
        edge = Edge(dummy, target)
        target._inbox.append((box, edge))

# ============================================================
# 2) Helper Functions
# ============================================================

def setup_complex_brain(n_neurons=50, seed=None):
    if seed is None: seed = random.randint(0, 10000)
    rng = random.Random(seed)
    
    kinds = ["exc"]*20 + ["inh"]*15 + ["reg"]*15
    rng.shuffle(kinds)
    neurons = [Neuron(f"N{i}", k) for i, k in enumerate(kinds)]
    
    net = Network(neurons)
    net.wire_randomly(p=0.5, seed=seed) # 연결 밀도 0.5
    
    s, t = rng.sample(neurons, 2)
    net.terminal = t
    return net, s, t

def get_keyword_vector(text: str) -> Vec4:
    """간이 감정 추출기"""
    text = text.replace(" ", "")
    # 긍정/행복
    if any(w in text for w in ["좋아", "행복", "신나", "편해", "사랑", "최고", "감사", "멋져"]):
        return (0.9, 0.8, 0.3, 0.2)
    # 분노/짜증
    elif any(w in text for w in ["화나", "짜증", "미친", "열받아", "싫어", "망했"]):
        return (0.4, 0.2, 0.9, 0.1)
    # 슬픔/우울
    elif any(w in text for w in ["슬퍼", "우울", "힘들", "지쳐", "눈물", "외로"]):
        return (0.2, 0.3, 0.4, 0.8)
    # 공포/불안
    elif any(w in text for w in ["무서", "불안", "걱정", "당황"]):
        return (0.3, 0.2, 0.8, 0.4)
    # 기본
    return (0.5, 0.5, 0.5, 0.5)

def generate_prompt_for_human(user_text: str, emotion_vec: Vec4, trace_len: str) -> str:
    """
    최종 뇌 상태를 바탕으로 LLM에게 줄 '지시문(Prompt)'을 생성합니다.
    """
    D, S, NE, M = emotion_vec
    
    # 1. 상태 해석
    mood_desc = []
    if D > 0.7: mood_desc.append("엄청나게 신나고, 의욕적이며, 보상감을 느끼는 상태")
    elif D < 0.3: mood_desc.append("지루하고, 의욕이 없으며, 무관심한 상태")
    
    if NE > 0.7: mood_desc.append("스트레스가 심하고, 예민하며, 공격적인 상태")
    elif NE < 0.3: mood_desc.append("긴장이 풀려 아주 나른하고 편안한 상태")
    
    if M > 0.7: mood_desc.append("졸리고, 무기력하며, 축 쳐진 상태")
    if S > 0.7: mood_desc.append("마음이 안정되고, 차분하며, 현명한 상태")
    
    if not mood_desc: mood_desc.append("감정적으로 평온하고 중립적인 상태")
    
    full_mood = ", ".join(mood_desc)

    # 2. 페르소나 강도 조절
    intensity = "보통"
    if D > 0.9 or NE > 0.9: intensity = "매우 강함 (과장되게 표현할 것)"
    elif trace_len == "Lost": intensity = "멍함 (맥락을 놓침)"

    # 3. 프롬프트 텍스트 생성
    prompt = f"""
[아래 내용을 LLM에 복사해서 붙여넣으세요]
--------------------------------------------------
당신은 '생물학적 뇌 시뮬레이션'을 장착한 AI입니다.
방금 사용자의 말을 듣고 당신의 뇌 속 신경전달물질이 반응했습니다.

현재 당신의 내부 화학 상태:
- 도파민(쾌락/의욕): {D:.2f}
- 세로토닌(안정): {S:.2f}
- 노르에피네프린(스트레스): {NE:.2f}
- 멜라토닌(피로): {M:.2f}

결론적인 기분: "{full_mood}"
감정 강도: {intensity}

[지시사항]
위의 '화학 상태'와 '기분'을 바탕으로 사용자에게 대답하세요.
- 도파민이 높으면: 느낌표(!)를 많이 쓰고, 이모지를 사용하며, 아주 열정적으로 반응하세요.
- 노르에피네프린이 높으면: 말투가 날카롭거나, 방어적이거나, 단답형으로 반응하세요.
- 멜라토닌이 높으면: 졸린 듯이 말하거나, 귀찮다는 듯이 반응하세요.
- 수치를 직접 언급하지 말고, '연기(Acting)'를 통해 보여주세요.

사용자 입력: "{user_text}"
--------------------------------------------------
"""
    return prompt

# ============================================================
# 3) Main Loop
# ============================================================

def main():
    print(f"\n🧠 [Neuro-Brain: Manual Mode]")
    print("   - 50 Neurons / Plasticity ON / Dopamine Shield ON")
    print("   - 입력을 넣으면 'LLM용 프롬프트'를 생성해줍니다.")
    print("   - Type 'quit' to exit.\n")

    while True:
        user_input = input("\n👤 You: ").strip()
        if user_input.lower() in ["quit", "exit"]: break
        if not user_input: continue

        print("   🧠 Brain is processing... ", end="", flush=True)
        
        # 1. 감정 추출 및 주입
        v_init = get_keyword_vector(user_input)
        net, s, t = setup_complex_brain(n_neurons=50)
        
        # 기분 좋으면 강하게 주입
        k_init = 0.9 if v_init[0] > 0.7 else 0.7
        box = DataBox(K=k_init, V=v_init, trace=["Input"])
        net.inject(s, box)
        
        # 2. 시뮬레이션
        final_vec = (0.5, 0.5, 0.5, 0.5)
        path_str = "Lost"
        
        for _ in range(100):
            net.tick()
            if net.arrived and net.arrival_box:
                final_vec = net.arrival_box.V
                path_str = f"{len(net.arrival_box.trace)} steps"
                break
        
        print("Done!")
        
        # 3. 결과 출력
        print(f"\n   [Neural Result]")
        print(f"   - Input V: {v_init}")
        print(f"   - Final V: ({final_vec[0]:.2f}, {final_vec[1]:.2f}, {final_vec[2]:.2f}, {final_vec[3]:.2f})")
        
        if not net.arrived:
            print("   ⚠️ (생각이 뇌 안에서 길을 잃었습니다. 멍 때리는 중...)")
            final_vec = (0.3, 0.3, 0.3, 0.8) # 멍함 = 멜라토닌 높음
            path_str = "Lost"

        # 4. 프롬프트 생성
        prompt = generate_prompt_for_human(user_input, final_vec, path_str)
        print(prompt)
        print("👉 위 박스 안의 내용을 복사해서 LLM에게 물어보세요!\n")

if __name__ == "__main__":
    main()