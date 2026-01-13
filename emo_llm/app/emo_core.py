# emo_core.py
import re, json, time, numpy as np
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

# ===== 설정 =====
MODEL_NAME   = "openai/gpt-oss-20b"   # HF 경로(또는 로컬 모델 폴더)
TAU_SECONDS  = 45.0                   # 감정 잔향 시간상수
T0, P0       = 0.8, 0.9               # 기본 temperature / top_p

def _clamp(v, a=-1.0, b=1.0) -> float:
    return float(max(a, min(b, v)))

# ===== 1) 감정 벡터 추출 =====
class EmotionExtractor:
    """
    텍스트 → [DA, 5HT, NE, ACh] ∈ [-1,1]
    이미 로드한 토크나이저/모델을 tok/mdl 인자로 주입 가능(중복 로딩 방지)
    """
    def __init__(self, model_name: str = MODEL_NAME, tok: Optional[AutoTokenizer] = None, mdl: Optional[AutoModelForCausalLM] = None):
        if tok is not None and mdl is not None:
            self.tok = tok
            self.model = mdl
        else:
            self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def infer(self, text: str) -> np.ndarray:
        prompt = (
            "[SYSTEM] 문장의 정동을 도파민(DA), 세로토닌(5HT), 노르에피네프린(NE), 아세틸콜린(ACh) "
            "스칼라로 추정해. -1.0~+1.0 범위 JSON만 출력.\n"
            f"[USER] 문장: \"{text}\"\n"
            "{ \"DA\": , \"5HT\": , \"NE\": , \"ACh\": }"
        )
        ids = self.tok(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**ids, max_new_tokens=120)
        s = self.tok.decode(out[0], skip_special_tokens=True)
        m = re.search(r"\{.*\}", s, re.S)
        try:
            d = json.loads(m.group(0)) if m else {}
        except Exception:
            d = {}
        DA  = _clamp(float(d.get("DA",  0.0)))
        HT  = _clamp(float(d.get("5HT", 0.0)))
        NE  = _clamp(float(d.get("NE",  0.0)))
        ACh = _clamp(float(d.get("ACh", 0.0)))
        return np.array([DA, HT, NE, ACh], dtype=float)

# ===== 2) 감정 잔향(감쇠) =====
@dataclass
class EmotionState:
    m: np.ndarray = np.zeros(4)  # 혼합 잔향
    t: float      = time.time()

class Memory:
    """세션별 감정 잔향 관리"""
    def __init__(self):
        self._s = {}

    def mix(self, sid: str, e_now: np.ndarray) -> np.ndarray:
        s = self._s.get(sid, EmotionState())
        now = time.time()
        dt = max(0.0, now - s.t)
        alpha = float(np.exp(-dt / TAU_SECONDS))
        s.m = alpha*s.m + (1-alpha)*e_now
        s.t = now
        self._s[sid] = s
        return 0.6*s.m + 0.4*e_now  # \tilde e_t

# ===== 3) 뉴런군 신경망(클러스터 → 최종 1노드 수렴) =====
class NT(IntEnum):
    E=0  # 흥분(글루탐산)
    I=1  # 억제(GABA)
    M=2  # 조절
    OUT=3

@dataclass
class Node:
    type: NT
    layer: int
    cluster: int

class EmoNet:
    """
    - 여러 층과 군집(클러스터)
    - E/I는 시냅스 연결, M은 게인/학습률 조절
    - 마지막에 단일 OUT 노드로 수렴
    - 규칙:
      * E: DA↑ → 가중치 증가(보상), 5-HT↑ → 임계값↑
      * I: 5-HT(90/10), DA(70/30) 게이트로 가중 강약 조정
      * NE: 전역 게인, ACh: 학습률(발화 큰 노드 가중)
    """
    def __init__(self, L=3, C=2, sizes=(4,3,1), seed=0):
        self.rng = np.random.default_rng(seed)
        self.nodes = []
        for l in range(L):
            for c in range(C):
                E,I,M = sizes
                self.nodes += [Node(NT.E,l,c) for _ in range(E)]
                self.nodes += [Node(NT.I,l,c) for _ in range(I)]
                self.nodes += [Node(NT.M,l,c) for _ in range(M)]
        self.out_idx = len(self.nodes)
        self.nodes.append(Node(NT.OUT, L, 0))
        self.N = len(self.nodes)

        self.idx_E = np.array([i for i,n in enumerate(self.nodes) if n.type==NT.E])
        self.idx_I = np.array([i for i,n in enumerate(self.nodes) if n.type==NT.I])
        self.idx_M = np.array([i for i,n in enumerate(self.nodes) if n.type==NT.M])

        # 시냅스 가중치(양수로 저장; 억제는 계산 시 빼기)
        self.WE = np.zeros((self.N, self.N))  # from E
        self.WI = np.zeros((self.N, self.N))  # from I
        self._wire(L, C)

        self.v = np.zeros(self.N)  # 막전위
        self.r = np.zeros(self.N)  # 발화율 tanh(v)

        # 감정 입력 투사(첫 층 E에)
        self.B = np.zeros((self.N,4))
        for i,n in enumerate(self.nodes):
            if n.layer==0 and n.type==NT.E:
                self.B[i,:] = self.rng.normal(0.15, 0.02, size=4)

        # 하이퍼파라미터 계수
        self.k_E     = 0.35   # E: DA→W_E
        self.k_I     = 0.45   # I: (5-HT,DA)→W_I
        self.k_theta = 0.4    # 5-HT→θ (양수에서만)
        self.k_NE    = 0.4    # NE→gain
        self.k_ACh   = 0.5    # ACh 전역 학습률
        self.k_r     = 0.2    # 발화율 가중
        self.beta_reward = 0.5 # DA 보상항
        self.lmbda   = 0.1    # 누출

    def _wire(self, L, C, p_intra=0.6, p_inter=0.2, p_skip=0.05):
        def group(l,c,t):
            return [i for i,n in enumerate(self.nodes) if (n.layer==l and n.cluster==c and n.type==t)]
        for l in range(L):
            for c in range(C):
                E = group(l,c,NT.E); I = group(l,c,NT.I)
                # 군집 내부(조밀)
                for pre in E:
                    for post in E+I:
                        if self.rng.random()<p_intra:
                            self.WE[post,pre]=self.rng.normal(0.6,0.1)
                for pre in I:
                    for post in E+I:
                        if self.rng.random()<p_intra:
                            self.WI[post,pre]=abs(self.rng.normal(0.5,0.1))
                # 이웃/스킵(희박)
                for tc in range(C):
                    for tl,p in [(min(l+1,L-1),p_inter),(min(l+2,L-1),p_skip)]:
                        E2 = group(tl,tc,NT.E); I2 = group(tl,tc,NT.I)
                        for pre in E:
                            for post in E2+I2:
                                if self.rng.random()<p:
                                    self.WE[post,pre]=self.rng.normal(0.5,0.12)
                        for pre in I:
                            for post in E2+I2:
                                if self.rng.random()<p*0.5:
                                    self.WI[post,pre]=abs(self.rng.normal(0.4,0.12))
        # 마지막 층 → OUT(수렴)
        for i,n in enumerate(self.nodes):
            if n.layer==L-1 and n.type in (NT.E,NT.I):
                (self.WE if n.type==NT.E else self.WI)[self.out_idx, i]=self.rng.normal(0.75,0.1)

    @staticmethod
    def _gate(x: float, hi: float, lo: float) -> float:
        """x>=0 → hi*x, x<0 → lo*x (방출 비율 게이트)"""
        return (hi if x>=0 else lo) * x

    def step(self, ev: np.ndarray, lmbda: float = None, theta0: float = 1.0, kM: float = 0.15):
        """
        ev: 혼합 감정 벡터 [DA, 5HT, NE, ACh]
        반환: (out_value, stats_dict)
        """
        if lmbda is None:
            lmbda = self.lmbda

        DA, HT, NE, ACh = map(float, ev)

        # --- 방출 비율 게이트(5-HT 90/10, DA 70/30) ---
        g5 = (0.9 if HT >= 0 else 0.1)
        gD = (0.7 if DA >= 0 else 0.3)

        # 모듈레이션 계수
        sE    = 1 + self.k_E * (gD * DA)                           # E: DA↑ → 보상/강화
        sI    = 1 + self.k_I * (g5 * HT + gD * DA)                 # I: 5-HT/DA 비율 반영
        theta = theta0 * (1 + self.k_theta * max(HT, 0.0))         # 5-HT 많을 때만 임계값↑
        gain  = 1 + self.k_NE * NE                                 # NE: 전역 게인
        eta_global = 1e-3 * (1 + self.k_ACh * ACh)                 # ACh: 전역 학습률
        eta_node   = eta_global * (1 + self.k_r * np.clip(self.r, 0, 1))  # 발화 큰 노드 가중

        # 군집 M 평균 발화율로 미세 조정(조절성)
        M_mean = np.zeros(self.N)
        buckets = {}
        for i,n in enumerate(self.nodes):
            if n.type==NT.M:
                buckets.setdefault((n.layer,n.cluster), []).append(self.r[i])
        for (l,c), arr in buckets.items():
            mean_m = float(np.mean(arr))
            for i,n in enumerate(self.nodes):
                if n.layer==l and n.cluster==c:
                    M_mean[i]=mean_m

        # 동역학 업데이트
        drive = (sE*(self.WE @ self.r)) - (sI*(self.WI @ self.r))   # 억제는 감산
        inp   = self.B @ ev                                         # 감정 입력 투사
        self.v = (1 - lmbda)*self.v + (gain*(1 + kM*M_mean))*drive + inp - theta
        self.r = np.tanh(self.v)

        # 간단 가소성: ACh 학습률 + DA 보상(양수일 때)
        if DA > 0:
            outer = np.outer(self.r, self.r)                        # post x pre
            dWE = (eta_node[:, None]) * (self.beta_reward * DA) * outer
            # E-발신 열만 업데이트(흥분성만 보상강화)
            self.WE[:, self.idx_E] += dWE[:, self.idx_E]

        # 억제 가중 미세 스케일(5-HT/DA 비율)
        scale_I = 1.0 + (0.02 * (g5*HT + gD*DA))
        self.WI[:, self.idx_I] *= np.clip(scale_I, 0.9, 1.1)

        out = float(self.r[self.out_idx])
        stats = dict(sE=float(sE), sI=float(sI), theta=float(theta),
                     gain=float(gain), eta=float(eta_global), out=out)
        return out, stats

# ===== 4) 감정→디코딩 매핑 =====
def decode_controls(ev: np.ndarray):
    DA, HT, NE, ACh = map(float, ev)
    def clip(x,a,b): return float(max(a, min(b, x)))
    temperature = clip(T0 * (1 - 0.25*HT - 0.15*NE + 0.2*DA), 0.1, 1.3)
    top_p       = clip(P0 * (1 - 0.3*ACh - 0.1*HT), 0.5, 1.0)
    return {"temperature": temperature, "top_p": top_p}

# ===== 5) 파이프라인 =====
class EmotionPipeline:
    """
    .process(text, session_id) -> dict
      - emotion_now: 즉시 추정된 감정 벡터
      - emotion_mix: 잔향 포함 혼합 감정 벡터
      - net_out:     출력 노드 활성
      - net_stats:   sE/sI/theta/gain/eta
      - decode:      temperature/top_p
    """
    def __init__(self, model_name: str = MODEL_NAME, tok: Optional[AutoTokenizer] = None, mdl: Optional[AutoModelForCausalLM] = None):
        self.ex  = EmotionExtractor(model_name, tok=tok, mdl=mdl)
        self.mem = Memory()
        self.net = EmoNet()

    def process(self, text: str, session_id: str="default"):
        e_now = self.ex.infer(text)
        e_mix = self.mem.mix(session_id, e_now)
        out, stats = self.net.step(e_mix)
        dec = decode_controls(e_mix)
        return {
            "emotion_now":  e_now.tolist(),
            "emotion_mix":  e_mix.tolist(),
            "net_out":      out,
            "net_stats":    stats,     # sE/sI/theta/gain/eta
            "decode":       dec        # temperature/top_p
        }
