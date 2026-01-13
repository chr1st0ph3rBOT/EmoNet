# -*- coding: utf-8 -*-
"""
emovec_autorun_cls.py
- 인자 없이 실행:  python emovec_autorun_cls.py
- DATA_PATH에 있는 JSON/JSONL(샘플과 동일 포맷) 로드
- 감정코드(예: "E18")를 **분류(LogisticRegression, multinomial, class_weight='balanced')**
  로 예측한 뒤, 미리 정의한 EMOTION_VEC 프로토타입으로 4D 벡터로 매핑
- 평가: accuracy / macro-F1 (+ 매핑 후 MAE 참고용)
- 데모 문장 예측 출력
- 모델/리포트 저장

사용 전 수정할 부분:
    DATA_PATH = "YOUR_FULL_DATASET.json"  # 전체 데이터 파일명
    OUT_DIR   = "emovec_autorun_cls_out"  # 산출물 폴더
    USE_SS    = False                     # SS(공감 문장) 포함 여부
"""

from __future__ import annotations
import json, pathlib, sys
from typing import List, Dict
import numpy as np
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

# ------------------- SETTINGS -------------------
DATA_PATH = "감성대화말뭉치(최종데이터)_Training.json"     # JSON array or JSONL
OUT_DIR   = "emovec_autorun_cls_out"
USE_SS    = False                        # 기본은 HS만 사용(노이즈 감소)

TEST_TEXTS = [
    "일이 왜 이렇게 끝이 없지? 화나.",
    "요즘 회사 생활이 편하고 좋아.",
    "면접에서 갑자기 예상치 못한 질문이 나와서 당황했어.",
    "친구들은 다 취업했는데 나만 못 해서 불안해.",
]

# ------------------- Emotion code → 4D prototype -------------------
EMOTION_VEC = {
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

# ------------------- Fallback sample (파일 없을 때) -------------------
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

def IPT2NTL():
    data_path = pathlib.Path(DATA_PATH)
    out_dir = pathlib.Path(OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)

    if data_path.exists():
        items = load_json_any(data_path); used_path = str(data_path)
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

    model_path = out_dir/"emovec_autorun_cls.pkl"
    dump(pipe, model_path)
    report = {
        "mode": "classification->vector_map",
        "data_used": used_path,
        "use_ss": USE_SS,
        "n_train": len(Xtr), "n_test": len(Xte),
        "acc": float(acc), "f1_macro": float(f1m),
        "mae_after_mapping": float(mae_vec),
        "keys": KEYS
    }
    (out_dir/"emovec_autorun_cls_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    demo_lbl = pipe.predict(TEST_TEXTS)
    demo_vec = map_vec(demo_lbl).tolist()
    demo = [{"text": t, "label": l, "vector": v, "keys": KEYS} for t, l, v in zip(TEST_TEXTS, demo_lbl, demo_vec)]

    print(json.dumps({
        "saved_model": str(model_path),
        "report_path": str(out_dir/"emovec_autorun_cls_report.json"),
        "report": report,
        "demo_predictions": demo
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    IPT2NTL()
