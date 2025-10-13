# 최종 스크립트 가이드

## 날짜: 2025-10-13

---

## 🎯 최종 분석 워크플로우

```
1. V3 분석 실행 (데이터 전처리 + 파일 통계)
   └─> scripts/run_integrated_analysis_v3.py
        ↓
2. V3 결과 후처리 (PID/FF 게인 계산)
   └─> scripts/post_process_v3_results.py
        ↓
3. 개별 게인 통합 (Excel 생성)
   └─> scripts/combine_individual_gains.py
```

---

## 📁 최종 스크립트 파일 목록

### 🟢 **필수 실행 스크립트 (3개)**

#### 1. `scripts/run_integrated_analysis_v3.py` ⭐⭐⭐
**역할**: V3 통합 분석 실행 (1단계)
- 모든 CSV 파일 파싱
- 정상 상태 속도 추출
- 파일별 통계 생성
- 디버그 플롯/CSV 생성

**실행 방법**:
```bash
python scripts/run_integrated_analysis_v3.py
```

**출력**:
- `output/integrated_v3/file_statistics.csv` ⭐ (가장 중요!)
- `output/integrated_v3/debug/plots/*.png` (디버그용)
- `output/integrated_v3/debug/parsed_data/*.csv` (검증용)

---

#### 2. `scripts/post_process_v3_results.py` ⭐⭐⭐
**역할**: V3 결과 후처리 (2단계)
- `file_statistics.csv` 읽어서 각 샘플별 PID/FF 계산
- 통계적 이상치 제거 (IQR)
- 최종 대표 게인 산출 (중앙값)

**실행 방법**:
```bash
python scripts/post_process_v3_results.py
```

**출력**:
- `output/post_process_v3/final_gains.json` ⭐ (최종 게인!)
- `output/post_process_v3/Arm_In_individual_gains.csv`
- `output/post_process_v3/Arm_Out_individual_gains.csv`
- `output/post_process_v3/Boom_Up_individual_gains.csv`
- `output/post_process_v3/Boom_Down_individual_gains.csv`
- `output/post_process_v3/Bucket_In_individual_gains.csv`
- `output/post_process_v3/Bucket_Out_individual_gains.csv`

---

#### 3. `scripts/combine_individual_gains.py` ⭐⭐
**역할**: 개별 게인 파일 통합 (3단계, 선택)
- 6개 축의 individual gains를 하나로 통합
- Excel 파일 생성 (축별 시트)

**실행 방법**:
```bash
python scripts/combine_individual_gains.py
```

**출력**:
- `output/post_process_v3/all_individual_gains.csv` (전체 165개 샘플)
- `output/post_process_v3/all_individual_gains.xlsx` ⭐ (Excel, 8개 시트)

---

## 🔧 **핵심 모듈 파일**

### 데이터 파싱
- `src/parser/batch_parser.py` - 파일 그룹핑
- `src/parser/filename_parser.py` - 파일명 파싱 (Boom/Bucket 구분 로직 포함!)
- `src/parser/csv_parser.py` - CSV 로딩
- `src/parser/data_validator.py` - 데이터 검증, 각도 마진 적용

### 분석 엔진
- `src/identification/integrated_analyzer_v3.py` ⭐ - V3 통합 분석 엔진
- `src/identification/model_fitting.py` ⭐ - 1차 모델 피팅 (K >= 0 제약 제거됨!)

### 튜닝
- `src/tuning/pid_tuner_imc.py` - IMC PID 튜닝
- `src/tuning/ff_tuner.py` - FF 게인 추정

---

## 🚀 **전체 실행 순서**

### 처음부터 끝까지 실행

```bash
# 1단계: V3 분석 (약 2-3분)
python scripts/run_integrated_analysis_v3.py

# 2단계: 후처리 (약 30초)
python scripts/post_process_v3_results.py

# 3단계: 통합 (약 10초)
python scripts/combine_individual_gains.py
```

### 또는 배치 파일 사용 (Windows)

**새로 생성**: `run_all_final.bat`
```batch
@echo off
echo ================================================
echo 최종 PID/FF 게인 추정 파이프라인
echo ================================================

echo.
echo [1/3] V3 통합 분석 실행 중...
python scripts/run_integrated_analysis_v3.py
if %errorlevel% neq 0 (
    echo 오류: V3 분석 실패!
    pause
    exit /b 1
)

echo.
echo [2/3] 결과 후처리 중...
python scripts/post_process_v3_results.py
if %errorlevel% neq 0 (
    echo 오류: 후처리 실패!
    pause
    exit /b 1
)

echo.
echo [3/3] 개별 게인 통합 중...
python scripts/combine_individual_gains.py
if %errorlevel% neq 0 (
    echo 오류: 통합 실패!
    pause
    exit /b 1
)

echo.
echo ================================================
echo 완료! 결과 확인:
echo   - output/post_process_v3/final_gains.json
echo   - output/post_process_v3/all_individual_gains.xlsx
echo ================================================
pause
```

---

## 📂 **최종 출력 파일 구조**

```
output/
├── integrated_v3/                          # V3 분석 결과
│   ├── file_statistics.csv                ⭐ (165개 파일 통계)
│   └── debug/
│       ├── plots/                         (정상/비정상 플롯)
│       │   ├── Arm_In_D40_Single_High_0.png
│       │   ├── ABNORMAL_Bucket_In_D40_Single_High_100.png
│       │   └── ...
│       └── parsed_data/                   (파싱된 데이터)
│           ├── Arm_In_D40_Single_High_0.csv
│           └── ...
│
└── post_process_v3/                       # 후처리 결과
    ├── final_gains.json                   ⭐⭐⭐ (최종 게인!)
    ├── all_individual_gains.csv           ⭐⭐ (165개 샘플)
    ├── all_individual_gains.xlsx          ⭐⭐⭐ (Excel)
    ├── Arm_In_individual_gains.csv        (28개)
    ├── Arm_Out_individual_gains.csv       (28개)
    ├── Boom_Up_individual_gains.csv       (28개)
    ├── Boom_Down_individual_gains.csv     (27개)
    ├── Bucket_In_individual_gains.csv     (28개)
    └── Bucket_Out_individual_gains.csv    (26개)
```

---

## 🔍 **스크립트별 상세 설명**

### `run_integrated_analysis_v3.py`

**주요 기능**:
1. 데이터 폴더 스캔
2. 파일 그룹핑 (Arm_In, Arm_Out, ...)
3. 각 파일에서 정상 상태 속도 추출
4. 파일 상태 분류:
   - `OK`: 정상
   - `Too Short`: 데이터 부족
   - `Abnormal Velocity`: 속도 이상
   - `Error`: 처리 오류

**핵심 파라미터** (config.yaml):
```yaml
preprocessing:
  angle_margin: 3.0          # 각도 마진 (도)
  velocity_threshold: 0.5     # 속도 임계값 (deg/s)
  outlier_threshold: 3.0      # 이상치 임계값 (σ)
```

---

### `post_process_v3_results.py`

**주요 기능**:
1. `file_statistics.csv` 로드
2. 각 샘플별 1차 시스템 모델 피팅:
   ```
   G(s) = K / (τs + 1)
   ```
3. IMC 방법으로 PID 계산:
   ```python
   Kp = τ / (K × λ)
   Ki = 1 / (K × λ)
   Kd = 0
   ```
4. IQR 이상치 제거
5. 중앙값으로 최종 게인 산출
6. FF 게인 선형 피팅:
   ```
   duty = Kv × velocity + K_offset
   ```

**핵심 변경사항**:
- ✅ `ModelFitter`의 K 범위: `[-∞, ∞]` (음수 K 허용!)
- ✅ Duty 부호 처리: 감소 방향은 음수 duty
- ✅ 각도 마진: 방향 고려

---

### `combine_individual_gains.py`

**주요 기능**:
1. 6개 축별 individual gains CSV 읽기
2. 하나의 DataFrame으로 통합
3. Excel 파일 생성:
   - `All_Samples`: 전체 165개
   - `Arm_In` ~ `Bucket_Out`: 축별 시트
   - `Summary_Statistics`: 통계 요약

---

## ⚠️ **사용하지 않는 스크립트 (참고용)**

### 🔴 V1 (초기 버전, 사용 안 함)
- `scripts/run_integrated_analysis.py`
- `src/identification/integrated_analyzer.py`

### 🔴 V2 (버그 수정 버전, V3로 대체)
- `scripts/run_integrated_analysis_v2.py`
- `src/identification/integrated_analyzer_v2.py`

### 🔴 V4 (Method 1 + Method 2, 실패)
- `scripts/run_integrated_analysis_v4.py`
- `scripts/run_integrated_analysis_v4_simple.py`
- `src/identification/integrated_analyzer_v4.py`
- `src/identification/integrated_analyzer_v4_simple.py`
- `src/identification/multi_sample_gain_estimator.py`

### 🔴 디버그용 (일회성)
- `scripts/check_bucket_data.py`
- `scripts/debug_batch_parser.py`
- `scripts/analyze_quality.py`

---

## 📊 **결과 파일 우선순위**

### ⭐⭐⭐ 가장 중요 (제어기 구현용)
1. `output/post_process_v3/final_gains.json`
   - 6개 축의 최종 PID/FF 게인
   - 통계 정보 포함

2. `FINAL_GAINS_SUMMARY.md`
   - 표 형식 요약
   - 적용 가이드

3. `CONTROLLER_IMPLEMENTATION_GUIDE.md`
   - 제어기 구현 예제 코드
   - Kd=0, K_offset 설명

### ⭐⭐ 중요 (분석/검증용)
4. `output/post_process_v3/all_individual_gains.xlsx`
   - 전체 165개 샘플
   - 축별 시트
   - 통계 분석

5. `output/integrated_v3/file_statistics.csv`
   - 원본 속도 데이터
   - 파일별 상태

### ⭐ 참고용
6. `SUCCESS_FINAL_RESULTS.md` - 최종 성공 보고서
7. `FINAL_ROOT_CAUSE_ANALYSIS.md` - K=0 문제 근본 원인
8. `output/integrated_v3/debug/plots/*.png` - 시각화

---

## 🛠️ **유지보수 및 재실행**

### 데이터 추가 시

1. 새 CSV를 `data/` 폴더에 추가
2. 전체 재실행:
   ```bash
   python scripts/run_integrated_analysis_v3.py
   python scripts/post_process_v3_results.py
   python scripts/combine_individual_gains.py
   ```

### 파라미터 튜닝 시

**`config/config.yaml` 수정**:
```yaml
preprocessing:
  angle_margin: 3.0       # 조정 가능
  velocity_threshold: 0.5  # 조정 가능

tuning:
  imc:
    lambda_factor: 1.0    # 1.0 ~ 3.0 (보수적 ~ 공격적)
```

### 코드 수정 시

**핵심 파일만 수정**:
1. `src/identification/integrated_analyzer_v3.py` - 분석 로직
2. `scripts/post_process_v3_results.py` - PID 계산
3. `src/identification/model_fitting.py` - 모델 피팅

---

## 📝 **Quick Reference**

### 빠른 실행 (3줄)
```bash
python scripts/run_integrated_analysis_v3.py
python scripts/post_process_v3_results.py
python scripts/combine_individual_gains.py
```

### 결과 확인 (2개 파일)
```
output/post_process_v3/final_gains.json
output/post_process_v3/all_individual_gains.xlsx
```

### 문서 확인 (3개 파일)
```
FINAL_GAINS_SUMMARY.md
CONTROLLER_IMPLEMENTATION_GUIDE.md
SUCCESS_FINAL_RESULTS.md
```

---

## 🎯 **최종 정리**

### 실행 스크립트 (필수 3개)
1. ✅ `scripts/run_integrated_analysis_v3.py` - V3 분석
2. ✅ `scripts/post_process_v3_results.py` - 후처리
3. ✅ `scripts/combine_individual_gains.py` - 통합

### 핵심 모듈 (수정 가능)
- `src/identification/integrated_analyzer_v3.py` - 분석 엔진
- `src/identification/model_fitting.py` - 모델 피팅 (K 범위!)
- `scripts/post_process_v3_results.py` - PID 계산 (Duty 부호!)

### 최종 결과
- `output/post_process_v3/final_gains.json` - 게인
- `output/post_process_v3/all_individual_gains.xlsx` - 샘플
- `FINAL_GAINS_SUMMARY.md` - 요약표

---

**모든 것이 준비되었습니다! 🎉**

실행만 하면 됩니다:
```bash
python scripts/run_integrated_analysis_v3.py && \
python scripts/post_process_v3_results.py && \
python scripts/combine_individual_gains.py
```

