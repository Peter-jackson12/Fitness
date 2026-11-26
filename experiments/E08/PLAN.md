# E08 최종 Test 평가 계획(실행 전에 작성, 2026-11-26)

- 승인: 사용자 "Test 평가 진행"(2026-11-26). 선정은 D-031.
- 모델: 후보 A(Full FT, letterbox 224, 증강 없음), 가중치 `models/SEL-A-s43_full.pt`
  - SHA-256 `f21a180db8c362efc029a8cf0a5d8c651fdcef56d69ca41e82da2f12e65a1f63`. 다르면 평가하지 않는다.
- 평가: `predict.predict_split("SEL-A-s43", "full", "test", allow_test=True)` → `metrics.summarize`(Macro F1 12클래스 고정, Accuracy, 클래스별 P/R/F1, CM).
- **한 번만** 실행한다. `scripts/run_e08_test.py`는 `experiments/E08/test_summary.json`이 이미 있으면 거부한다.
- Test 결과를 보고 모델·가중치·전처리·지표를 바꾸지 않는다. 결과가 나빠도 그대로 보고한다.
- Test 결과로 추가 튜닝을 하지 않는다. 오류 분석은 "어떤 클래스에서 일반화가 깨졌는가"를 기술하는 데까지만 하고, 그 결과로 모델을 고르지 않는다.
