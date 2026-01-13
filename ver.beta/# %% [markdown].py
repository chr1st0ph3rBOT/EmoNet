# %% [markdown]
# **환경**

# %%
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 안전: 토치 컴파일 끄기
os.environ["TORCH_COMPILE_DISABLE"] = "1"

MODEL_ID = "openai/gpt-oss-20b"

tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map=None,           # <- auto 금지
    low_cpu_mem_usage=False,   # <- 지연 로드 금지 (meta 방지)
)
model.to("cuda")               # <- 즉시 materialize
model.eval()

gen = pipeline("text-generation", model=model, tokenizer=tok)

# 짧게 구동 체크
out = gen([{"role": "user", "content": "안녕! 테스트야."}],
          max_new_tokens=40, do_sample=False)[0]["generated_text"]
print(out[-1]["content"])


# %%
import json, re

SYSTEM = (
    "You are EmotionScore. Respond with JSON ONLY. "
    "Return exactly these four keys as floats in [0,1]: "
    "dopamine, serotonin, norepinephrine, melatonin. "
    "Do not write any other text, no explanation."
)

def emotion_vector(text: str):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Text: {text}"},
        {"role": "assistant", "content": "{"}  # JSON 강제 시작
    ]
    out = gen(messages, max_new_tokens=60, do_sample=False,
              return_full_text=False)[0]["generated_text"]
    content = "{" + out

    m = re.search(r"\{.*\}", content, re.S)
    if not m:
        raise RuntimeError(f"JSON not found in: {content[:200]}")
    data = json.loads(m.group())

    def c(x): 
        try: return max(0.0, min(1.0, float(x)))
        except: return 0.0
    vec = [c(data.get(k)) for k in ("dopamine","serotonin","norepinephrine","melatonin")]
    return vec, data

# 테스트
demo = "우울하다"
vec, raw = emotion_vector(demo)
print("emotion vector [DA,5-HT,NE,MLT] =", vec)
print("raw =", raw)


# %% [markdown]
# **감정 저장소**

# %%
from dataclasses import dataclass
from typing import Dict, Optional
import time

def clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))

@dataclass
class EmotionState:
    DA: float = 0.0   # Dopamine
    HT: float = 0.0   # Serotonin
    NE: float = 0.0   # Noradrenaline
    ACh: float = 0.0  # Acetylcholine

    def clamp_(self):
        self.DA = clamp(self.DA); self.HT = clamp(self.HT)
        self.NE = clamp(self.NE); self.ACh = clamp(self.ACh)
        return self

class EmotionStore:
    def __init__(self, init: Optional[EmotionState] = None, decay_lambda: float = 0.9):
        self.raw = (init or EmotionState()).clamp_()
        self.decay_lambda = decay_lambda
        self.last_update_time = time.time()

    def update(self, delta: Dict[str, float]):
        """델타를 단순히 더해 저장"""
        self.raw.DA  = clamp(self.raw.DA  + delta.get("DA",  0.0))
        self.raw.HT  = clamp(self.raw.HT  + delta.get("HT",  0.0))
        self.raw.NE  = clamp(self.raw.NE  + delta.get("NE",  0.0))
        self.raw.ACh = clamp(self.raw.ACh + delta.get("ACh", 0.0))
        self.last_update_time = time.time()

    def get_state(self) -> EmotionState:
        """잔유 감쇠 계산해서 현재 상태 반환"""
        elapsed = max(1, int(time.time() - self.last_update_time))
        lam = self.decay_lambda ** elapsed
        return EmotionState(
            DA=clamp(self.raw.DA * lam),
            HT=clamp(self.raw.HT * lam),
            NE=clamp(self.raw.NE * lam),
            ACh=clamp(self.raw.ACh * lam),
        )


# %% [markdown]
# **흥분성 뉴런**

# %%
def clamp(x, lo, hi): 
    return max(lo, min(hi, x))

class ExcitatoryNeuron:
    def __init__(self, weight=0.6, threshold=1.0, resting=0.0,
                 lr=0.05, w_bounds=(0.0, 2.0),
                 k_scale=1.2,         # ⬅ 입력 스케일 업
                 ht_coeff=0.2,        # ⬅ HT 임계 기여 약화 (기존 0.5)
                 ne_coeff=0.6):       # ⬅ NE 이득 강화 (기존 0.5)
        self.w = weight
        self.thr = threshold
        self.rest = resting
        self.vm = resting
        self.lr = lr
        self.w_lo, self.w_hi = w_bounds
        self.k_scale = k_scale
        self.ht_coeff = ht_coeff
        self.ne_coeff = ne_coeff

    def reset(self):
        self.vm = self.rest

    def forward_text(self, text: str, store=None):
        vec, _ = emotion_vector(text)          # [DA, 5HT, NE, MLT] in [0,1]
        DA, HT, NE, MLT = vec
        K = (DA + HT + NE + MLT) / 4.0
        x = (store.start_potential(K) if store is not None else K) * self.k_scale

        gain = 1.0 + self.ne_coeff * NE
        thr_eff = self.thr + self.ht_coeff * HT

        self.vm += self.w * x * gain

        fired = self.vm >= thr_eff
        y = 1.0 if fired else 0.0
        if fired:
            self.vm = self.rest

        self.w = clamp(self.w + self.lr * (DA - 0.5) * x, self.w_lo, self.w_hi)

        return y, {
            "w": round(self.w, 4),
            "vm": round(self.vm, 4),
            "fired": bool(fired),
            "gain": round(gain, 3),
            "thr_eff": round(thr_eff, 3)
        }


# %%
# test_excitatory.py
ex = ExcitatoryNeuron()

texts = [
    "나는 기쁘고 행복해",     # 긍정 문장
    "나는 우울하고 슬퍼",     # 부정 문장
    "나는 너가 불안하고 싫어" # 강한 부정 문장
]

print("=== ExcitatoryNeuron Test ===")
for t in texts:
    y, info = ex.forward_text(t)
    print(f"Text: {t}")
    print(f" Output y: {y}")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print("-" * 30)


# %% [markdown]
# **억제성 뉴런**

# %%
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

class InhibitoryNeuron:
    """
    억제성 뉴런 (발화 튜닝 버전)
      - gain을 너무 깎지 않도록 하한↑
      - 세로토닌이 임계를 더 낮추도록 계수↑
      - 입력 스케일/초기 임계 약간 공격적으로
    """
    def __init__(self, weight=0.7, threshold=0.9, resting=0.0,
                 lr=0.04, w_bounds=(0.0, 2.0),
                 inhibit_strength=0.7,
                 k_scale=1.4,       # ↑ 입력 스케일
                 ht_coeff=0.35,     # ↑ 5-HT가 임계 낮추는 힘
                 ne_coeff=0.3,      # ↓ NE로 인한 억제 gain 감소 완화
                 gain_floor=0.6):   # ↑ gain 하한
        self.w = weight
        self.thr = threshold
        self.rest = resting
        self.vm = resting
        self.lr = lr
        self.w_lo, self.w_hi = w_bounds
        self.inhibit_strength = inhibit_strength
        self.k_scale = k_scale
        self.ht_coeff = ht_coeff
        self.ne_coeff = ne_coeff
        self.gain_floor = gain_floor

    def reset(self):
        self.vm = self.rest

    def _learn_w(self, x, DA, HT):
        d = 0.0
        d += (+0.90 if HT > 0.5 else -0.10) * self.lr * x
        d += (+0.70 if DA > 0.5 else -0.30) * self.lr * x
        self.w = clamp(self.w + d, self.w_lo, self.w_hi)

    def forward_text(self, text: str, store=None):
        vec, _ = emotion_vector(text)  # [DA, 5HT, NE, MLT]
        DA, HT, NE, MLT = vec
        K = (DA + HT + NE + MLT) / 4.0
        x = (store.start_potential(K) if store is not None else K) * self.k_scale

        gain = max(self.gain_floor, 1.0 - self.ne_coeff * NE)
        thr_eff = self.thr - self.ht_coeff * HT

        self.vm += self.w * x * gain

        fired = self.vm >= thr_eff
        y = -self.inhibit_strength if fired else 0.0
        if fired:
            self.vm = self.rest

        self._learn_w(x, DA, HT)

        return y, {
            "w": round(self.w, 4),
            "vm": round(self.vm, 4),
            "fired": bool(fired),
            "gain": round(gain, 3),
            "thr_eff": round(thr_eff, 3)
        }


# %%

inh = InhibitoryNeuron()

texts = [
    "나는 편안하고 차분해",     # 안정/평온
    "나는 짜증나고 화나",       # 분노/짜증
    "나는 무섭고 불안해",       # 불안/공포
    "나는 너가 불쾌하고 싫어"   # 강한 부정
]

print("=== InhibitoryNeuron Test ===")
for t in texts:
    y, info = inh.forward_text(t)   # store를 쓰지 않을 경우 None
    print(f"Text: {t}")
    print(f" Output y: {y}")        # 발화 시 음수(-inhibit_strength)
    for k, v in info.items():
        print(f"  {k}: {v}")        # w, vm, fired, gain, thr_eff
    print("-" * 30)


# %% [markdown]
# **조절성 뉴런**

# %%
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

class ModulatoryNeuron:
    """
    조절성 뉴런
      - emotion_vector(text) → [DA, 5HT, NE, MLT] ∈ [0,1]
      - 이 뉴런은 직접 발화 출력 대신, modulation dict 반환
        * dopamine   (DA): 보상/흥분 → 동기부여, 가중치 조절
        * serotonin  (HT): 안정/억제 → 발화 임계값 조절
        * norepineph (NE): 예민/각성 → 전체 gain 조절
        * acetylch.  (ACh): 집중/학습률 조절 (여기서는 MLT 대신 ACh로)
    """
    def __init__(self,
                 base_lr=0.05,
                 base_gain=1.0,
                 base_thr=1.0,
                 base_attn=1.0):
        self.base_lr = base_lr      # 기본 학습률
        self.base_gain = base_gain  # 기본 이득
        self.base_thr = base_thr    # 기본 임계
        self.base_attn = base_attn  # 기본 집중계수

    def forward_text(self, text: str):
        vec, _ = emotion_vector(text)   # [DA, 5HT, NE, MLT]
        DA, HT, NE, MLT = vec

        # 조절 파라미터 계산
        lr_mod   = self.base_lr   * (1.0 + 0.6 * DA)        # DA ↑ → 학습률 ↑
        thr_mod  = self.base_thr  * (1.0 + 0.5 * HT)        # 5-HT ↑ → 임계 ↑
        gain_mod = self.base_gain * (1.0 + 0.6 * NE)        # NE ↑ → 이득 ↑
        attn_mod = self.base_attn * (1.0 + 0.6 * MLT)       # MLT → 집중 ↑ (ACh 자리에 사용)

        mods = {
            "lr": round(lr_mod, 4),
            "thr": round(thr_mod, 4),
            "gain": round(gain_mod, 4),
            "attn": round(attn_mod, 4),
            "DA": round(DA,3), "HT": round(HT,3),
            "NE": round(NE,3), "MLT": round(MLT,3)
        }
        return mods


# %%
# test_modulatory.py
mod = ModulatoryNeuron()

texts = [
    "나는 기쁘고 즐겁다",   # 도파민↑
    "나는 안정되고 편안해", # 세로토닌↑
    "나는 불안하고 예민해", # 노르아드레날린↑
    "나는 집중해야 해"      # 아세틸콜린(집중)↑
]

print("=== ModulatoryNeuron Test ===")
for t in texts:
    mods = mod.forward_text(t)
    print(f"Text: {t}")
    for k, v in mods.items():
        print(f"  {k}: {v}")
    print("-"*30)


# %% [markdown]
# **신경망**

# %%
class OnePassNetwork:
    def __init__(self, store=None):
        self.store = store or EmotionStore()
        self.ex = ExcitatoryNeuron()
        self.inh = InhibitoryNeuron()
        self.mod = ModulatoryNeuron()

    def forward(self, text: str):
        vec, _ = emotion_vector(text)           # [DA, HT, NE, MLT]
        K = sum(vec) / len(vec)                 # 단순 평균
        x0 = self.store.start_potential(K)

        mods = self.mod.forward_text(text)      # lr, thr, gain, attn
        x = x0 * mods["attn"]

        yE, infoE = self.ex.forward_text(text, store=self.store)
        yI, infoI = self.inh.forward_text(text, store=self.store)

        # 델타 산출 (간단 규칙)
        delta = {
            "DA": +0.25*yE - 0.05*abs(yI),
            "HT": +0.20*abs(yI),
            "NE": +0.20*yE - 0.10*abs(yI),
            "ACh": +0.15*(mods["attn"]-1.0)
        }
        self.store.update(delta)

        return {
            "input": text,
            "vec": [round(v,3) for v in vec],
            "Vm0": round(x0,3),
            "mods": mods,
            "excitatory": infoE,
            "inhibitory": infoI,
            "delta": {k: round(v,3) for k,v in delta.items()},
            "state_after_update": self.store.raw.__dict__.copy()
        }


# %%
# controller_all.py
# 요구 전제:
# - gen(messages, ...)         : 1/2패스 생성기
# - emotion_vector(text)       : [DA, 5HT, NE, MLT] ∈ [0,1] 반환
# - OnePassNetwork             : 너의 네트워크 (질문에 제공한 버전)
#
# 사용:
#   python controller_all.py
# 명령:
#   /quit            종료
#   /debug on|off    콘솔 상세 로그 토글
#   /context <text>  1패스 초안에 붙일 추가 컨텍스트
#   /raw             2패스 생략(초안 그대로) 토글
#   /help            도움말

import os, json, time, uuid, traceback
from typing import Any, Dict, Optional

# -------------------- 유틸 --------------------
def now_ts():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def clip01(x):
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0

def pretty(d: Any) -> str:
    return json.dumps(d, ensure_ascii=False, indent=2)

# ---------------- 2-패스 컨트롤 ----------------
CONTENT_SYSTEM = (
    "You are a helpful assistant. Focus ONLY on factual, task-relevant content. "
    "Keep tone neutral. Avoid hedging and emotions."
)

def map_emotion_to_controls(vec):
    """[DA, 5HT, NE, MLT] -> 스타일 파라미터"""
    DA, HT, NE, MLT = [clip01(v) for v in vec]
    warmth      = 0.2 + 0.8*DA - 0.3*NE
    directness  = 0.3 + 0.7*NE - 0.4*HT
    brevity     = 0.3 + 0.4*MLT + 0.2*HT
    formality   = 0.3 + 0.6*HT + 0.1*MLT
    urgency     = 0.2 + 0.8*NE - 0.2*MLT
    positivity  = 0.2 + 0.8*DA - 0.3*HT
    temperature = 0.4 + 0.6*max(0.0, NE - 0.3) - 0.3*(HT + MLT)/2
    top_p       = 0.7 + 0.2*NE - 0.2*HT
    base_tokens = 180 + int(120*(DA - 0.5) - 100*(HT + MLT - 1.0))
    return {
        "warmth":      clip01(warmth),
        "directness":  clip01(directness),
        "brevity":     clip01(brevity),
        "formality":   clip01(formality),
        "urgency":     clip01(urgency),
        "positivity":  clip01(positivity),
        "temperature": max(0.1, min(1.2, temperature)),
        "top_p":       max(0.4, min(1.0, top_p)),
        "max_new_tokens": max(80, min(600, base_tokens)),
    }

def generate_draft(user_text: str, extra_context: str = "") -> str:
    msgs = [
        {"role": "system", "content": CONTENT_SYSTEM},
        {"role": "user", "content": (extra_context + "\n" if extra_context else "") + user_text}
    ]
    out = gen(msgs, max_new_tokens=300, do_sample=False, return_full_text=False)[0]["generated_text"]
    return out

STYLE_SYSTEM_TMPL = (
    "You are a style controller for Korean outputs.\n"
    "Rewrite the user's [Draft] to satisfy the style controls below.\n\n"
    "STYLE CONTROLS:\n"
    "- warmth={warmth}\n- directness={directness}\n- brevity={brevity}\n"
    "- formality={formality}\n- urgency={urgency}\n- positivity={positivity}\n\n"
    "OUTPUT RULES:\n"
    "1) 답변은 반드시 한국어 최종 문장만 출력.\n"
    "2) 절대 분석/설명/사고과정을 출력하지 말 것.\n"
    "3) 초안의 사실과 지시를 보존하되, 스타일만 조정할 것.\n"
    "4) brevity>0.7이면 짧은 문장/불릿 활용.\n"
    "5) directness>0.7이면 완곡어/hedging 지양.\n"
    "6) formality>0.6이면 정중한 비즈니스 한국어.\n"
    "7) warmth>0.6이면 공감 문장 1개 이내로 추가.\n"
    "8) urgency>0.7이면 명확한 행동 요청과 기한 포함.\n"
)

def style_rewrite(draft: str, controls: Dict[str, Any]) -> str:
    style_system = STYLE_SYSTEM_TMPL.format(**controls)
    msgs = [
        {"role": "system", "content": style_system},
        {"role": "user", "content": draft},
        {"role": "assistant", "content": ""}  # 최종문장만 강제
    ]
    out = gen(
        msgs,
        max_new_tokens=controls.get("max_new_tokens", 200),
        do_sample=True,
        temperature=controls.get("temperature", 0.7),
        top_p=controls.get("top_p", 0.9),
        return_full_text=False
    )[0]["generated_text"]
    return out

# --------------- 컨트롤러 ----------------
class ChatController:
    def __init__(self, net: "OnePassNetwork", session_id: Optional[str] = None,
                 log_dir: str = "./chat_logs", debug: bool = True, raw_only: bool = False):
        self.net = net
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.log_dir = log_dir
        ensure_dir(log_dir)
        self.log_path = os.path.join(log_dir, f"session_{self.session_id}_{int(time.time())}.jsonl")
        self.turn = 1
        self.debug = debug
        self.raw_only = raw_only   # True면 2패스 생략(초안만 사용)

    def log_jsonl(self, obj: Dict[str,Any]):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def process(self, user_text: str, extra_context: str = "") -> Dict[str,Any]:
        # 감정 벡터 (2패스 컨트롤용)
        vec, raw_json = emotion_vector(user_text)
        controls = map_emotion_to_controls(vec)

        # 1패스 초안
        draft = generate_draft(user_text, extra_context=extra_context)

        # 2패스 스타일 적용 (옵션)
        if self.raw_only:
            final = draft
        else:
            final = style_rewrite(draft, controls)

        # ★ 전체 신경망 forward
        net_out = self.net.forward(user_text)

        record = {
            "ts": now_ts(),
            "turn": self.turn,
            "user_text": user_text,
            "emotion_vec": raw_json,
            "controls": controls,
            "draft": draft,
            "final": final,
            "network": net_out
        }
        self.log_jsonl(record)
        if self.debug:
            self._print_console(record)
        self.turn += 1
        return record

    def _print_console(self, r: Dict[str,Any]):
        print("\n" + "="*78)
        print(f"[TURN {r['turn']}]  {r['ts']}")
        print("="*78)
        print("USER:")
        print(" ", r["user_text"])
        print("-"*78)
        print("EMOTION VECTOR (raw):")
        print(pretty(r["emotion_vec"]))
        print("-"*78)
        print("CONTROLS:")
        print(pretty(r["controls"]))
        print("-"*78)
        print("DRAFT:\n", r["draft"])
        print("-"*78)
        print("FINAL:\n", r["final"])
        print("-"*78)
        print("NETWORK OUT:")
        print(pretty(r["network"]))
        print("="*78)

# ---------------- 실행부 ----------------
HELP = """
명령:
  /quit            종료
  /debug on|off    디버깅 로그 토글
  /context <text>  1패스 초안에 결합할 추가 컨텍스트
  /raw             2패스 생략(초안 그대로) 토글
  /help            도움말
"""

def main():
    print("=== ChatController (OnePassNetwork) ===")
    print(HELP)

    # OnePassNetwork 준비
    # 주의: EmotionStore에 start_potential(K) 메서드가 있어야 합니다.
    net = OnePassNetwork()

    controller = ChatController(net=net, debug=True, raw_only=False)

    extra_context = ""
    while True:
        try:
            user_text = input("\nUSER > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye."); break

        if not user_text:
            continue
        if user_text.startswith("/quit"):
            print("종료합니다."); break
        if user_text.startswith("/help"):
            print(HELP); continue
        if user_text.startswith("/debug"):
            parts = user_text.split()
            if len(parts) == 2 and parts[1] in ("on","off"):
                controller.debug = (parts[1] == "on")
                print(f"debug = {controller.debug}")
            else:
                print("Usage: /debug on|off")
            continue
        if user_text.startswith("/raw"):
            controller.raw_only = not controller.raw_only
            print(f"raw_only = {controller.raw_only}  (True면 2패스 생략)")
            continue
        if user_text.startswith("/context"):
            extra_context = user_text[len("/context"):].strip()
            print(f"extra_context set: {extra_context!r}")
            continue

        try:
            controller.process(user_text, extra_context=extra_context)
        except Exception as e:
            print("[ERROR]", e)
            traceback.print_exc()
            controller.log_jsonl({
                "ts": now_ts(),
                "turn": controller.turn,
                "user_text": user_text,
                "error": str(e),
                "trace": traceback.format_exc()
            })
            controller.turn += 1

if __name__ == "__main__":
    main()



