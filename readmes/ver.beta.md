# ver.beta 상세 리드미

## 1) 버전 개요
- **핵심 목표**: LLM 기반 감정 벡터 추출 + 뉴런 모델링 + 2패스 응답 제어를 결합.
- **중심 접근**: 감정(DA/5HT/NE/MLT) 벡터를 스타일 파라미터로 변환해 출력 톤 제어.

## 2) 핵심 파일 구성
- `ver.beta/# %% [markdown].py`
  - 노트북 셀을 합친 파이썬 파일.
  - LLM 로딩, 감정 벡터 추출, Excitatory/Inhibitory/Modulatory 뉴런, OnePassNetwork, ChatController(2패스 출력) 포함.
- `ver.beta/main.ipynb`, `ver.beta/main ver2.ipynb`, `ver.beta/main ver3.ipynb`
  - 위 구조를 노트북 기반으로 단계적 실험.

## 3) 핵심 아이디어 정리
- **LLM 감정 벡터화**: JSON 형태로 4축 감정값을 수집.
- **뉴런 모델링**: 흥분/억제/조절 뉴런을 분리해 감정 상태를 동적으로 업데이트.
- **2패스 응답 제어**: 1패스 초안 → 감정 스타일 재작성으로 톤 조정.

## 4) 이전 버전 대비 개선/특징
- alpha의 스파이크 기반 실험과 달리 **LLM 감정 벡터 추출과 출력 제어**가 결합됨.
- 감정 상태를 **EmotionStore**에 축적해 잔향/감쇠를 모델링.

## 5) 실행 방법 (처음 보는 사람용)
- **노트북/스크립트 실행 흐름**
  1. `ver.beta/# %% [markdown].py` 파일을 노트북 셀처럼 순서대로 실행.
  2. LLM 가능한 환경(GPU 포함)에서 다음 순서로 진행:
     - 모델 로딩 → `emotion_vector` 테스트 → 뉴런 테스트 → ChatController.
  3. CLI에서 `/help`로 명령 확인.

## 6) 한계 및 개선 아이디어
- `EmotionStore.start_potential` 같은 **누락 메서드 의존성** 존재.
- LLM 호출 비용/속도가 높으므로 **캐싱/경량 모델 대체** 필요.
- 스타일 파라미터가 실제 출력에 반영되는 정도를 **정량 검증**해야 함.

## 7) 다음 단계 제안
- EmotionStore/뉴런 인터페이스 정리 및 빠진 메서드 보완.
- 벡터-스타일 매핑 효과에 대한 A/B 테스트 설계.
