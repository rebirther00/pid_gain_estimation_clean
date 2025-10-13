# PRD (Product Requirements Document)

## 1. 프로젝트 개요

### 1.1 프로젝트 명
Excavator System Identification and Controller Tuning

### 1.2 목적
굴착기 붐(Boom), 암(Arm), 버킷(Bucket)의 동작 시험 데이터를 분석하여 각 관절의 PID 게인(P, I, D) 및 속도 피드포워드(FF) 게인을 추정합니다.

### 1.3 배경
- 굴착기 각 관절에 일정한 제어 출력(duty 값: 40, 50, 60, 70, 80, 90, 100%)을 인가하여 응답 특성을 측정했습니다
- 단독 동작(Single)과 연동 동작(Couple) 데이터를 통해 상호 간섭을 분석합니다
- 부하 조건(High/Low)에 따른 특성 차이를 파악합니다

## 2. 데이터 구조

### 2.1 폴더 구조
```
data/
├── Arm_Single/
├── Arm_Couple/
├── Boom_Single/
├── Boom_Couple/
├── Bucket_Single/
└── Bucket_Couple/
```

### 2.2 파일 명명 규칙
`(축이름)-(축방향)-(duty값)-(H/L)-(S/C).csv`

**파라미터 설명:**
- **축이름**: A(Arm/Bucket, 폴더명으로 구분), B(Boom)
- **축방향**: 
  - Arm/Bucket: `in`, `out`
  - Boom: `up`, `dn` (down)
- **duty값**: 제어 출력 상수값 (40, 50, 60, 70, 80, 90, 100)
- **H/L**: High load(H) / Low load(L)
- **S/C**: Single(S) / Couple(C)

**파일 예시:**
- `A-in-40-H-S.csv` (Arm Single 폴더 내, duty 40%)
- `B-up-70-L-C.csv` (Boom Couple 폴더 내, duty 70%)

### 2.3 CSV 데이터 포맷
**샘플링 레이트**: 10ms (100Hz)

**컬럼 구조** (레거시 코드 분석 필요):
- `time`: 시간 (초)
- `control_input`: 제어 입력 (duty %)
- `Bkt_ang`: 버킷 각도 (도)
- `Arm_ang`: 암 각도 (도)
- `Boom_ang`: 붐 각도 (도)
- `feedback_current`: 피드백 전류 (선택적)
- `pilot_pressure`: 파일럿 압력 (선택적)
- `cylinder_pressure`: 실린더 압력 (선택적)

### 2.4 각도 범위
```python
angle_limits = {
    'Bkt_ang': {'min': -125.0, 'max': 40.0},   # 165도 범위
    'Arm_ang': {'min': -150.0, 'max': -40.0},  # 110도 범위
    'Boom_ang': {'min': 0.0, 'max': 55.0}      # 55도 범위
}
```

### 2.5 Duty 값 사양
- **범위**: 0~100%
- **제어 로직**:
  - 제어기 출력: -100 ~ +100
  - 음수(-100~0): In/Down 밸브에 역변환하여 인가 (100~0%)
  - 양수(0~100): Out/Up 밸브에 그대로 인가 (0~100%)
- **시험 범위**: 40, 50, 60, 70, 80, 90, 100% (dead zone 및 안전 고려)

### 2.6 데이터 유효성
- 실린더 끝단 충격으로 인한 노이즈 제거 필요
- **유효 데이터 범위**: 최종 각도 값 - 3도까지만 사용

## 3. 기능 요구사항

### 3.1 데이터 파싱 (Phase 1)
- [ ] 레거시 코드(`Function_test_csv_analysis` 폴더) 분석
  - CSV 컬럼 구조 파악
  - 데이터 로딩 로직 추출
- [ ] CSV 파일 파싱 구현
  - pandas 기반 데이터 로딩
  - 컬럼명 표준화
- [ ] 파일명 파싱을 통한 메타데이터 추출
  - 축, 방향, duty, 부하, Single/Couple 정보
- [ ] 유효 데이터 필터링
  - 각 축별 angle_limits 적용
  - 최종 각도 - 3도 규칙 적용
- [ ] 파싱된 데이터 저장 (pickle 형식)

### 3.2 데이터 전처리 (Phase 2)
- [ ] 이상치 제거
  - 각도 범위 초과값 제거
  - 급격한 변화 감지 (outlier detection)
- [ ] 노이즈 필터링
  - Savitzky-Golay 필터 또는 이동평균
  - 필터 파라미터는 config.yaml에서 관리
- [ ] 속도 계산
  - 중앙 차분법으로 각속도 계산
  - 필터링 적용
- [ ] 가속도 계산 (선택적, PID D 게인 추정용)
- [ ] 정상 상태 구간 식별
  - 속도 변화율 기준
  - 마지막 N% 구간의 속도 분산 확인
- [ ] 데이터 품질 검증
  - 샘플링 레이트 확인
  - 결측치 확인 및 처리
  - 데이터 재현성 확인 (동일 조건 반복 시험 비교)

### 3.3 시스템 식별 (Phase 3)
- [ ] 계단 응답 특성 추출
  - 상승 시간, 정착 시간, 오버슈트
- [ ] 1차 시스템 모델 피팅
  ```
  G(s) = K / (τs + 1)
  ```
  - DC gain (K) 추정
  - 시정수 (τ) 추정
  - 시간 지연 (delay) 고려
- [ ] 2차 시스템 모델 피팅 (필요 시)
  ```
  G(s) = ωn² / (s² + 2ζωns + ωn²)
  ```
- [ ] 모델 적합도 평가
  - R² (결정계수)
  - RMSE (Root Mean Square Error)
  - 시각적 비교
- [ ] Duty별 모델 파라미터 비교
  - 선형성 확인
  - 포화 특성 파악
- [ ] Single vs Couple 비교 분석
  - 차이율 계산
  - 통계적 유의성 검증
- [ ] 커플링 효과 정량화
  - 5% 이상 차이 시 유의미하다고 판단 (기준은 조정 가능)

### 3.4 게인 추정 (Phase 4)

#### 3.4.1 속도 FF 게인 추정
**방법 1: 정상 상태 속도 기반 직접 계산**
```python
# 정상 상태 구간 추출
steady_state_idx = int(len(velocity) * 0.8)
v_ss = np.mean(velocity[steady_state_idx:])
K_ff = duty / v_ss  # [%/(deg/s)]
```

**방법 2: 1차 시스템 DC gain 활용**
```python
# 피팅된 DC gain으로부터
K_ff = 1 / K_dc  # K_dc는 시스템 식별에서 추정된 값
```

- [ ] 두 방법 결과 비교
- [ ] Duty별 FF 게인 선형성 확인
- [ ] 방향별(In/Out, Up/Down) FF 게인 비교
- [ ] 부하 조건(H/L)별 FF 게인 비교

#### 3.4.2 PID 게인 추정

**Ziegler-Nichols 방법 (1차 시스템 + 시간 지연 모델)**
```python
# 1차 시스템: G(s) = K * e^(-Ls) / (τs + 1)
Kp = τ / (K * L)
Ki = Kp / (2 * L)
Kd = Kp * L / 2
```

**Cohen-Coon 방법**
```python
# 더 나은 외란 제거 성능
Kp = (τ / (K * L)) * (1 + L / (3 * τ))
Ki = Kp / (L * (1 + L / (4 * τ)))
Kd = Kp * L / (1 + L / (4 * τ))
```

**IMC (Internal Model Control) 튜닝**
```python
# 원하는 폐루프 시정수 λ 선택
# λ = 1~3 * τ (보수적), λ = 0.3~1 * τ (공격적)
lambda_c = 1.5 * tau  # 중간 값
Kp = τ / (K * lambda_c)
Ki = 1 / (K * lambda_c)
Kd = 0  # 1차 시스템에서는 D 게인 불필요
```

- [ ] 세 가지 방법으로 PID 게인 추정
- [ ] 안정성 마진 계산
  - Phase margin
  - Gain margin
  - 보드 선도 분석
- [ ] 시뮬레이션 검증
  - 계단 응답 시뮬레이션
  - 추종 성능 평가
- [ ] 보수적 게인 추천 (안전 계수 적용)

### 3.5 결과 분석 및 검증 (Phase 5)

#### 3.5.1 게인 정리
- [ ] 각 축별, 방향별 최종 게인 선정
  - Arm In: PID + FF
  - Arm Out: PID + FF
  - Boom Up: PID + FF
  - Boom Down: PID + FF
  - Bucket In: PID + FF
  - Bucket Out: PID + FF
- [ ] Duty별 게인 변화 분석
- [ ] 부하 조건별 게인 비교
- [ ] Single vs Couple 게인 차이 분석

#### 3.5.2 룩업 테이블 검토
- [ ] 각도 범위별 게인 변화 확인
- [ ] 필요 시 각도 구간 분할 제안
- [ ] 룩업 테이블의 불연속성 검토
  - 구간 경계에서의 게인 차이
  - 스무딩 방법 제안

#### 3.5.3 오차 분석
- [ ] 모델 예측 오차 계산
  - RMSE, MAE
  - 최대 오차
- [ ] 게인 추정 신뢰 구간 계산
  - 부트스트랩 방법
  - 여러 duty 값에서의 일관성
- [ ] 실제 적용 시 예상 오차
  - 시뮬레이션 기반 추정

#### 3.5.4 실행 가능성 판단
- [ ] 데이터 충분성 평가
  - duty 값 개수 (7개: 40~100%)
  - 반복 시험 유무 확인
- [ ] 게인 추정 가능 여부 판단 기준
  - 모델 적합도 R² > 0.9
  - 추정 오차 < 20%
  - 안정성 확보
- [ ] 불가능 시 명확한 사유 출력
  - 데이터 품질 문제
  - 모델 부적합
  - 비선형성 과다
- [ ] 추가 시험 계획 제안
  - PRBS (Pseudo-Random Binary Sequence) 입력
  - Sine sweep (주파수 응답 분석)
  - 다른 각도 범위에서의 시험
  - 더 많은 duty 값 시험

### 3.6 결과 출력 (Phase 6)

#### 3.6.1 게인 값 출력
**JSON 형식**
```json
{
  "arm": {
    "in": {
      "pid": {"kp": 1.5, "ki": 0.3, "kd": 0.05},
      "ff": {"kv": 2.1},
      "method": "IMC",
      "confidence": 0.92
    },
    "out": {...}
  },
  "boom": {...},
  "bucket": {...}
}
```

**YAML 형식 (ROS2 파라미터 파일)**
```yaml
arm_controller:
  in:
    pid: {p: 1.5, i: 0.3, d: 0.05}
    feedforward: {kv: 2.1}
  out:
    pid: {p: 1.8, i: 0.4, d: 0.06}
    feedforward: {kv: 2.3}
```

#### 3.6.2 분석 리포트 생성
**Markdown 형식**
- [ ] Executive Summary
  - 주요 결과 요약
  - 권장 게인 값
- [ ] 데이터 품질 분석
  - 파싱 통계
  - 유효 데이터 비율
- [ ] 시스템 식별 결과
  - 모델 파라미터
  - 적합도 지표
- [ ] 게인 추정 결과
  - 각 방법별 결과
  - 최종 선정 게인
- [ ] 오차 분석
  - 예상 성능
  - 제한 사항
- [ ] 추가 시험 계획 (필요 시)

**PDF 형식** (선택적)
- 전문적인 리포트 형식

#### 3.6.3 시각화
**필수 그래프** (한글 깨짐 방지):
```python
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
# 또는 영어로 레이블 작성
```

- [ ] 계단 응답 그래프
  - 원본 데이터
  - 유효 데이터 (3도 제거)
  - Duty별 비교
- [ ] 모델 피팅 결과
  - 실측 vs 모델 예측
  - 잔차(residual) 플롯
- [ ] 속도 프로파일
  - 각속도 vs 시간
  - 정상 상태 구간 표시
- [ ] 게인 분포도
  - Duty별 FF 게인 변화
  - 방향별 게인 비교
  - 부하별 게인 비교
- [ ] 커플링 분석 결과
  - Single vs Couple 차이
  - 상호 간섭 정량화
- [ ] 주파수 응답 (선택적)
  - 보드 선도 (Bode plot)
  - 안정성 마진 표시

## 4. 프로그램 구조

### 4.1 폴더 구조
```
system_identification_tuning/
├── config/
│   ├── config.yaml              # 메인 설정 파일
│   └── angle_limits.yaml        # 각도 범위 설정
├── data/
│   ├── raw/                     # 원본 CSV 데이터
│   │   ├── Arm_Single/
│   │   ├── Arm_Couple/
│   │   ├── Boom_Single/
│   │   ├── Boom_Couple/
│   │   ├── Bucket_Single/
│   │   └── Bucket_Couple/
│   └── processed/               # 전처리된 데이터
│       ├── arm_in_processed.pkl
│       ├── arm_out_processed.pkl
│       └── ...
├── legacy_analysis/
│   └── Function_test_csv_analysis/  # 레거시 코드
├── src/
│   ├── __init__.py
│   ├── parser/
│   │   ├── __init__.py
│   │   ├── filename_parser.py   # 파일명 파싱
│   │   ├── csv_parser.py        # CSV 데이터 파싱
│   │   └── data_validator.py    # 데이터 검증
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── noise_filter.py      # 노이즈 제거
│   │   ├── velocity_estimator.py # 속도 계산
│   │   ├── outlier_detector.py  # 이상치 탐지
│   │   └── steady_state_detector.py # 정상 상태 감지
│   ├── identification/
│   │   ├── __init__.py
│   │   ├── step_response.py     # 계단 응답 분석
│   │   ├── first_order_model.py # 1차 시스템 모델
│   │   ├── second_order_model.py # 2차 시스템 모델 (선택)
│   │   ├── model_fitting.py     # 시스템 모델 피팅
│   │   └── coupling_analysis.py # 커플링 분석
│   ├── tuning/
│   │   ├── __init__.py
│   │   ├── ff_tuner.py          # FF 게인 추정
│   │   ├── pid_tuner_zn.py      # Ziegler-Nichols
│   │   ├── pid_tuner_cc.py      # Cohen-Coon
│   │   ├── pid_tuner_imc.py     # IMC
│   │   └── lookup_table.py      # 룩업 테이블 생성
│   ├── validation/
│   │   ├── __init__.py
│   │   ├── stability_analysis.py # 안정성 분석
│   │   ├── error_analysis.py    # 오차 분석
│   │   ├── simulator.py         # 게인 검증 시뮬레이터
│   │   └── feasibility_checker.py # 실행 가능성 판단
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── step_response_plotter.py
│   │   ├── model_fit_plotter.py
│   │   ├── gain_distribution_plotter.py
│   │   ├── coupling_plotter.py
│   │   └── report_generator.py  # 리포트 생성
│   └── utils/
│       ├── __init__.py
│       ├── file_utils.py
│       ├── math_utils.py
│       ├── constants.py
│       └── logger.py            # 로깅 설정
├── scripts/
│   ├── run_arm_in.py            # Arm In 방향 분석
│   ├── run_arm_out.py           # Arm Out 방향 분석
│   ├── run_boom_up.py           # Boom Up 방향 분석
│   ├── run_boom_dn.py           # Boom Down 방향 분석
│   ├── run_bucket_in.py         # Bucket In 방향 분석
│   ├── run_bucket_out.py        # Bucket Out 방향 분석
│   ├── run_all.py               # 전체 분석
│   └── generate_ros2_params.py  # ROS2 파라미터 파일 생성
├── results/
│   ├── gains/                   # 추정 게인 값
│   │   ├── gains.json
│   │   ├── gains.yaml
│   │   └── ros2_params.yaml
│   ├── plots/                   # 그래프
│   │   ├── step_response/
│   │   ├── model_fit/
│   │   ├── gain_distribution/
│   │   └── coupling/
│   ├── reports/                 # 분석 리포트
│   │   ├── analysis_report.md
│   │   └── analysis_report.pdf
│   └── test_plans/              # 추가 시험 계획서
│       └── additional_test_plan.md
├── tests/
│   ├── __init__.py
│   ├── test_parser.py
│   ├── test_preprocessing.py
│   ├── test_identification.py
│   └── test_tuning.py
├── logs/                        # 로그 파일
│   └── analysis_YYYYMMDD_HHMMSS.log
├── requirements.txt
├── README.md
└── setup.py
```

### 4.2 실행 스크립트 분리
각 축 및 방향별로 독립적인 실행 스크립트:

```bash
# 특정 축/방향 분석
python scripts/run_arm_in.py

# 전체 분석
python scripts/run_all.py

# ROS2 파라미터 파일 생성
python scripts/generate_ros2_params.py
```

### 4.3 설정 파일 예시

**config/config.yaml**
```yaml
# 데이터 경로
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  
# 파싱 설정
parsing:
  valid_data_margin: 3.0  # 최종 각도 - 3도
  
# 전처리 설정
preprocessing:
  filter:
    type: "savgol"  # 'savgol', 'moving_average', 'butter'
    window_length: 11
    polyorder: 3
  velocity:
    method: "central_diff"  # 'central_diff', 'forward_diff'
  steady_state:
    threshold: 0.01  # 속도 변화율 임계값
    min_duration: 1.0  # 최소 지속 시간 (초)
    
# 시스템 식별 설정
identification:
  model_type: "first_order"  # 'first_order', 'second_order'
  fitting:
    method: "least_squares"
    max_iterations: 1000
  goodness_of_fit:
    min_r_squared: 0.9
    
# 게인 추정 설정
tuning:
  ff:
    method: ["steady_state", "dc_gain"]  # 둘 다 수행
  pid:
    methods: ["ziegler_nichols", "cohen_coon", "imc"]
    imc_lambda_factor: 1.5  # λ = factor * τ
  safety_factor: 0.8  # 보수적 게인 조정
  
# 커플링 분석 설정
coupling:
  significance_threshold: 0.05  # 5% 차이 이상
  
# 시각화 설정
visualization:
  font_family: "DejaVu Sans"  # 한글 깨짐 방지
  dpi: 300
  file_format: "png"
  
# 로깅 설정
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  file: "logs/analysis_{timestamp}.log"
```

**config/angle_limits.yaml**
```yaml
Bkt_ang:
  min: -125.0
  max: 40.0
Arm_ang:
  min: -150.0
  max: -40.0
Boom_ang:
  min: 0.0
  max: 55.0
```

## 5. 기술 스택

### 5.1 필수 라이브러리
```
python>=3.8
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
matplotlib>=3.7.0
control>=0.9.4        # 제어 시스템 분석
pyyaml>=6.0
```

### 5.2 선택 라이브러리
```
plotly>=5.0           # 인터랙티브 시각화
rich>=13.0            # CLI 출력 개선
tqdm>=4.65            # 진행률 표시
scikit-learn>=1.3     # 모델 검증 도구
seaborn>=0.12         # 통계 시각화
```

## 6. 제약사항 및 고려사항

### 6.1 데이터 관련
- ✅ 실린더 끝단 도달 시 노이즈 발생 (3도 제거)
- ✅ 부하 조건이 정량적이지 않음 (High/Low 구분만 존재)
- ✅ Duty 범위: 40~100% (dead zone 고려)
- ✅ 샘플링 레이트: 10ms (100Hz)
- ⚠️ 데이터 재현성 확인 필요 (반복 시험 유무)

### 6.2 모델링 관련
- ✅ 유압 시스템의 비선형성 (마찰, 백래시, 압력 포화)
- ✅ Duty별 선형성 가정 검증 필요
- ✅ 밸브 방향별 특성 차이 (In/Out, Up/Down)
- ✅ 시간 지연(delay) 고려
- ⚠️ 1차 vs 2차 시스템 모델 적합성 비교 필요

### 6.3 게인 추정 관련
- ✅ 목표 성능 지표 명확하지 않음 → 여러 방법 비교
- ✅ 안정성 마진 요구사항 없음 → 보수적 접근
- ✅ P, I, D, FF 게인 모두 추정
- ⚠️ 실제 적용 시 미세 조정 필요할 수 있음

### 6.4 프로그램 설계 관련
- ✅ 모듈화 및 재사용성
- ✅ 로깅 및 디버깅
- ✅ 재현성 보장 (설정 저장)
- ✅ 한글 깨짐 방지 (영어 레이블 또는 폰트 설정)

## 7. 추가 고려사항

### 7.1 데이터 품질 관리
- 📌 결측치 처리: 선형 보간 또는 제거
- 📌 이상치 탐지: IQR 방법 또는 Z-score
- 📌 데이터 재현성: 동일 조건 반복 시험 비교

### 7.2 모델 검증
- 📌 훈련/검증 분리: Duty별로 일부 데이터 홀드아웃
- 📌 교차 검증: K-fold (가능한 경우)
- 📌 적합도 지표: R², RMSE, MAE

### 7.3 안전성 및 실용성
- 📌 안정성 검증: Phase/Gain margin 계산
- 📌 보수적 게인: 안전 계수 0.8 적용
- 📌 ROS2 통합: 파라미터 파일 자동 생성

### 7.4 추가 시험 계획
- 📌 PRBS 입력 시험 (시스템 식별 개선)
- 📌 Sine sweep (주파수 응답 분석)
- 📌 다양한 각도 범위 시험
- 📌 더 조밀한 duty 간격 시험

### 7.5 결과 활용
- 📌 ROS2 파라미터 파일 생성
- 📌 게인 적용 가이드 문서
- 📌 미세 조정(fine-tuning) 가이드라인

## 8. 성공 기준

### 8.1 필수 기준
- [ ] 모든 시험 데이터 파싱 성공 (손실률 < 5%)
- [ ] 각 축/방향별 게인 추정 완료 (6개 조합)
- [ ] 커플링 효과 정량화 완료
- [ ] 예상 오차 분석 완료