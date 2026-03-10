# Clean Repro (s1_both_e05_h15_nt50_rt50_seed45)

이 폴더는 `sweep_v3_stage1_stage2_seed45` 로그에서 `F1=0.834`를 기록한 설정만
깔끔하게 재실행하기 위한 최소 구성입니다.

## 파일 구성

- `s1_both_e05_h15_nt50_rt50_seed45.env`
  - 재현에 필요한 핵심 하이퍼파라미터만 포함
- `run_repro.sh`
  - 위 env를 로드해서 기존 `run_train.sh`를 호출
  - 결과 로그/추론 파일명을 타임스탬프로 분리 저장

## 실행

```bash
cd /workspace/ancor/pipeline/anchor-guided-collaborative-grounding/clean_repro_s1_both
./run_repro.sh
```

## 출력 위치

- 로그: `../logs/reproduce_s1_both_e05_h15_nt50_rt50_seed45_<timestamp>.log`
- 테스트 추론 결과: `../runs/test_reproduce_s1_both_e05_h15_nt50_rt50_seed45_<timestamp>.jsonl`

## 참고

- 데이터/경로 체크와 `config.py` 생성은 기존 `../run_train.sh`를 그대로 사용합니다.
- 기존 실험 코드/스크립트는 수정하지 않습니다.
