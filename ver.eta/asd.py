# -*- coding: utf-8 -*-
import pathlib
import sys
from joblib import load

# 1. 경로 설정 (현재 폴더 기준)
BASE_DIR = pathlib.Path(__file__).parent.absolute()
MODEL_PATH = BASE_DIR / "emovec_real_brain.pkl"

def inspect():
    print(f"🔍 [모델 진단 시작] 경로: {MODEL_PATH}")

    # 1. 파일 존재 여부 확인
    if not MODEL_PATH.exists():
        print("❌ [결과] 모델 파일(.pkl)이 없습니다.")
        print("   -> 해결책: neuro_final_retrain.py를 한 번 실행해서 학습시키세요.")
        return

    # 2. 모델 로드 시도
    try:
        pipe = load(MODEL_PATH)
        print("✅ [결과] 모델 파일 로드 성공!")
    except Exception as e:
        print(f"❌ [결과] 모델 파일이 깨졌습니다. ({e})")
        return

    # 3. 뇌 용량(학습된 단어 수) 확인
    try:
        vocab = pipe['tfidf'].vocabulary_
        vocab_size = len(vocab)
        print(f"🧠 [뇌 용량 체크] 학습된 단어 수: {vocab_size}개")

        if vocab_size < 100:
            print("\n🚨 [치명적 문제 발견] 뇌가 너무 작습니다!")
            print("   - 원인: 데이터 파일 경로가 틀려서 '샘플 데이터(4문장)'만 학습된 상태입니다.")
            print("   - 증상: '싫어', '좋아' 같은 말을 해도 못 알아듣고 0.5만 뱉습니다.")
            print("   - 해결: .pkl 파일을 삭제하고, 데이터 파일 위치를 확인한 뒤 재학습하세요.")
        else:
            print("\n🟢 [정상] 뇌 용량이 충분합니다. (실제 데이터를 학습함)")
    except Exception as e:
        print(f"⚠️ [주의] 모델 구조가 예상과 다릅니다. ({e})")

    # 4. 실전 반응 테스트
    print("\n🧪 [반응 테스트]")
    test_sentences = [
        "나는 네가 정말 싫어", 
        "오늘 너무 행복해서 날아갈 것 같아", 
        "우울하고 죽고 싶어"
    ]
    
    for text in test_sentences:
        try:
            pred = pipe.predict([text])[0]
            # 확률(자신감) 확인 - 로지스틱 회귀일 경우 가능
            if hasattr(pipe['clf'], 'predict_proba'):
                proba = pipe.predict_proba([text]).max()
                confidence = f"(확신: {proba*100:.1f}%)"
            else:
                confidence = ""
            
            print(f"   🗣️ 입력: '{text}' -> 🤖 판단: {pred} {confidence}")
        except:
            print(f"   🗣️ 입력: '{text}' -> ❌ 판단 실패")

if __name__ == "__main__":
    inspect()