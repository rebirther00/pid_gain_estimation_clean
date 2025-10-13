# PID 게인 추정 방법론

## 결정 방법 (V3 최종 채택)

**날짜**: 2025-10-13  
**상태**: ✅ 검증 완료

### 개요

단일 샘플 기반 게인 추정은 유압 굴착기처럼 비선형성이 크고 노이즈가 많은 시스템에서는 부적절합니다.
여러 대안 방법을 검토한 결과, **방법 1: 다중 샘플 통계 기반 접근**을 최종 채택하였습니다.

### 최종 채택 방법: 다중 샘플 통계 기반 접근 (V3)

#### 전체 프로세스

```
[Step 1] V3 통합 분석
  ↓
[Step 2] 개별 샘플 PID 계산 (후처리)
  ↓
[Step 3] 통계적 필터링 & 대표값 산출
  ↓
[결과] 6개 축별 최종 PID/FF 게인
```

#### Step 1: V3 통합 분석 (데이터 전처리)

**목적**: 모든 CSV 파일에서 유효한 정상 상태 속도 데이터 추출

**처리 과정**:
```python
# 1. 파일 그룹핑
file_groups = {
    'Arm_In': [A-in-40-H-S.csv, ..., A-in-100-L-C.csv],  # 28개
    'Arm_Out': [...],  # 28개
    'Boom_Up': [...],   # 28개
    'Boom_Down': [...], # 27개
    'Bucket_In': [...], # 28개
    'Bucket_Out': [...] # 26개
}

# 2. 각 파일별 처리
for file in group_files:
    # 2.1 데이터 로딩 및 검증
    data = load_csv(file)
    data_valid = apply_angle_margin(data, margin=3.0)  # 각도 마진 적용
    
    # 2.2 노이즈 필터링
    angle_filtered = savgol_filter(data_valid['angle'])
    
    # 2.3 속도 계산
    velocity = calculate_velocity(angle_filtered)
    
    # 2.4 정상 상태 속도 추출
    steady_velocity = extract_steady_state_velocity(velocity)
    
    # 2.5 상태 분류 및 저장
    status = classify_data(data_valid, steady_velocity)
    # status: OK, Too Short, Abnormal Velocity, Error
    
    save_to_statistics_csv(file, duty, steady_velocity, status)

# 출력: file_statistics.csv (165개 파일 정보)
```

**출력 파일**: `output/integrated_v3/file_statistics.csv`
- 파일명, duty, 정상 상태 속도, 상태 등

**핵심 특징**:
- ✅ 각도 마진 3도 적용 (방향 고려)
- ✅ Savitzky-Golay 필터로 노이즈 제거
- ✅ 정상 상태 자동 검출
- ✅ 파일별 상태 분류 및 디버깅 정보 저장

---

#### Step 2: 개별 샘플 PID 계산 (후처리)

**목적**: `file_statistics.csv`의 각 샘플에서 개별 PID 게인 계산

**처리 과정**:
```python
# file_statistics.csv 로드
for each_sample in file_statistics:
    if sample.status != 'OK':
        continue
    
    # 1. 1차 시스템 모델 피팅
    # G(s) = K / (τs + 1)
    
    # 1.1 Duty 부호 처리 (핵심!)
    if angle_change < 0:  # In/Down 방향
        effective_duty = -abs(duty)  # 음수!
    else:  # Out/Up 방향
        effective_duty = abs(duty)
    
    # 1.2 초기값 설정
    K_init = abs(angle_change / effective_duty)
    tau_init = 1.0
    delay_init = 0.5
    
    # 1.3 Curve fitting (scipy.optimize.curve_fit)
    # 중요: K는 음수 허용!
    bounds = ([-np.inf, 0.01, 0], [np.inf, 100, 1.0])
    params, _ = curve_fit(first_order_model, time, angle,
                          p0=[K_init, tau_init, delay_init],
                          bounds=bounds)
    K, tau, delay = params
    
    # 2. IMC 방법으로 PID 계산
    lambda_c = 1.0 * tau  # 폐루프 시정수
    Kp = tau / (abs(K) * lambda_c)
    Ki = 1 / (abs(K) * lambda_c)
    Kd = 0  # 1차 시스템
    
    # 3. 품질 지표 계산
    y_pred = simulate_model(time, K, tau, delay)
    r_squared = calculate_r_squared(angle, y_pred)
    
    # 4. 개별 게인 저장
    save_individual_gain(axis, file, Kp, Ki, Kd, K, tau, r_squared)

# 출력: {axis}_individual_gains.csv (각 축마다)
```

**출력 파일**: 
- `Arm_In_individual_gains.csv` (28개 샘플)
- `Arm_Out_individual_gains.csv` (28개 샘플)
- `Boom_Up_individual_gains.csv` (28개 샘플)
- `Boom_Down_individual_gains.csv` (27개 샘플)
- `Bucket_In_individual_gains.csv` (28개 샘플)
- `Bucket_Out_individual_gains.csv` (26개 샘플)

**핵심 개선사항**:
1. ✅ **Duty 부호 처리**: In/Down 방향은 음수 duty
2. ✅ **K 음수 허용**: `bounds = ([-np.inf, ...], [np.inf, ...])`
3. ✅ **각도 마진 적용**: 방향에 따라 ±3도

---

#### Step 3: 통계적 필터링 & 대표값 산출

**목적**: 개별 게인들을 통계적으로 처리하여 최종 대표 게인 산출

**처리 과정**:
```python
# 각 축별로
for axis in ['Arm_In', 'Arm_Out', 'Boom_Up', 'Boom_Down', 'Bucket_In', 'Bucket_Out']:
    # 1. 개별 게인 로드
    gains_df = load_individual_gains(f"{axis}_individual_gains.csv")
    
    # 2. 품질 필터링
    # 2.1 R² 기반 (이미 계산 시 필터링됨)
    high_quality = gains_df[gains_df['r_squared'] > 0.7]
    
    # 3. IQR 이상치 제거
    Q1 = high_quality['Kp'].quantile(0.25)
    Q3 = high_quality['Kp'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    gains_filtered = high_quality[
        (high_quality['Kp'] >= lower_bound) & 
        (high_quality['Kp'] <= upper_bound)
    ]
    
    # 4. 중앙값으로 최종 게인 산출 (robust!)
    final_Kp = gains_filtered['Kp'].median()
    final_Ki = gains_filtered['Ki'].median()
    final_Kd = gains_filtered['Kd'].median()  # 항상 0
    
    # 5. FF 게인 계산 (선형 피팅)
    # duty = Kv * velocity + K_offset
    duties = valid_samples['duty']
    velocities = valid_samples['velocity']
    params, _ = curve_fit(linear_func, velocities, duties)
    Kv, K_offset = params
    
    # 6. 통계 정보 저장
    save_final_gains(axis, final_Kp, final_Ki, final_Kd, 
                     Kv, K_offset, statistics)

# 출력: final_gains.json
```

**출력 파일**: `output/post_process_v3/final_gains.json`

**최종 결과**:
```json
{
  "Arm_In": {
    "final_gains": {"Kp": 3.74, "Ki": 0.037, "Kd": 0.0},
    "ff_gain": {"Kv": 1.29, "K_offset": 41.1, "r_squared": 0.82},
    "n_samples_total": 28,
    "n_samples_valid": 24,
    "n_outliers": 4
  },
  ...
}
```

---

### 핵심 성공 요인

#### 1. 물리적 제약 고려
```python
# In/Down 방향: duty 양수 → 각도 감소 → K 음수
bounds = ([-np.inf, 0.01, 0], [np.inf, 100, 1.0])
```

#### 2. Robust 통계 방법
- **중앙값(Median)** 사용: 이상치에 강건
- **IQR 필터링**: 극단값 제거
- **품질 기반 필터링**: R² < 0.7 샘플 제외

#### 3. 방향별 비대칭 처리
- Arm In vs Out: 28% 차이
- Boom Up vs Down: 91% 차이 (거의 2배!)
- Bucket In vs Out: 12% 차이

#### 4. 충분한 샘플 수
- 각 축당 22~24개 유효 샘플
- 통계적 신뢰도 확보

---

### 최종 성과

| 항목 | 결과 |
|:---|:---|
| **총 샘플** | 165개 |
| **평균 성공률** | 82.7% |
| **평균 R² (PID)** | 0.92 (매우 우수!) |
| **평균 R² (FF)** | 0.71 (좋음) |
| **Kd** | 모두 0 (1차 시스템) |
| **실행 시간** | 3-4분 |

---

### V3 vs 다른 방법 비교

| 방법 | V3 (채택) | V4 (Method 1+2) | 단일 샘플 |
|:---|:---:|:---:|:---:|
| **계산 속도** | 빠름 (3분) | 느림 (10분+) | 매우 빠름 |
| **데이터 품질 요구** | 중간 | 높음 | 낮음 |
| **Robust성** | 높음 | 낮음 | 매우 낮음 |
| **구현 복잡도** | 낮음 | 높음 | 매우 낮음 |
| **성공률** | 82.7% | 0% (실패) | 33% |
| **결과 신뢰도** | 높음 | - | 낮음 |

**결론**: V3가 가장 실용적이고 효과적!

---

## 제안하는 대안 방법들 (참고용)

### **방법 1: 다중 샘플 통계 기반 접근 (추천)**

모든 시험 데이터(duty 40~100%, H/L, S/C)를 활용하여 게인을 추정합니다.

#### 접근 방식:
```python
# 1단계: 각 시험 케이스마다 모델 파라미터 추정
for each_test_case in all_test_data:
    K, tau, delay = fit_first_order_model(each_test_case)
    model_params.append({'K': K, 'tau': tau, 'delay': delay, 
                         'duty': duty, 'angle_range': angle_range})

# 2단계: 통계적 분석
# - duty에 따른 K, tau 변화 패턴 파악
# - 각도 범위에 따른 변화 파악
# - 부하(H/L)에 따른 변화 파악

# 3단계: 대표 모델 선정
# 방법 A: 중앙값(median) 사용 - 이상치에 강건
K_representative = np.median([p['K'] for p in model_params])
tau_representative = np.median([p['tau'] for p in model_params])

# 방법 B: 가중평균 사용 - 모델 적합도(R²)를 가중치로
weights = [p['r_squared'] for p in model_params]
K_representative = np.average([p['K'] for p in model_params], weights=weights)

# 4단계: 대표 모델로 PID 게인 계산
Kp, Ki, Kd = calculate_pid(K_representative, tau_representative)

# 5단계: 모든 샘플에 대해 검증
for each_test_case in all_test_data:
    error = simulate_with_gains(each_test_case, Kp, Ki, Kd, Kff)
    errors.append(error)
    
# 평균 오차, 최대 오차, 표준편차 계산
```

**장점:**
- 노이즈에 강건합니다
- 전체 동작 범위를 고려합니다
- 통계적 신뢰도를 제공합니다

---

### **방법 2: 최적화 기반 접근**

모든 시험 데이터를 동시에 고려하여 오차를 최소화하는 게인을 직접 찾습니다.

#### 접근 방식:
```python
from scipy.optimize import minimize

# 목적 함수: 모든 샘플에서의 평균 추종 오차
def objective_function(gains):
    Kp, Ki, Kd, Kff = gains
    total_error = 0
    
    for test_data in all_test_data:
        # 각 시험 데이터에 대해 시뮬레이션
        simulated_response = simulate_pid_controller(
            test_data, Kp, Ki, Kd, Kff
        )
        # 실제 응답과 비교
        error = calculate_error(simulated_response, test_data.actual)
        total_error += error
    
    return total_error / len(all_test_data)

# 초기값: Ziegler-Nichols 등으로 구한 값
initial_gains = [Kp_init, Ki_init, Kd_init, Kff_init]

# 최적화
result = minimize(
    objective_function, 
    initial_gains,
    method='Nelder-Mead',  # 또는 'SLSQP'
    bounds=[(0, 10), (0, 5), (0, 1), (0, 10)]  # 게인 범위 제한
)

optimal_gains = result.x
```

**장점:**
- 전체 데이터셋에 대해 최적화된 게인을 얻습니다
- 비선형성을 암묵적으로 고려합니다
- 실제 성능 지표를 직접 최적화할 수 있습니다

**단점:**
- 계산 비용이 높습니다
- 로컬 최적해에 빠질 수 있습니다

---

### **방법 3: 조건별 게인 스케줄링 (Gain Scheduling)**

조건(각도, duty, 부하)에 따라 다른 게인을 사용합니다.

#### 접근 방식:
```python
# 1단계: 조건별 그룹화
grouped_data = group_by_conditions(all_test_data, 
                                   angle_bins=[(-150, -100), (-100, -50), (-50, -40)],
                                   duty_bins=[(40, 60), (60, 80), (80, 100)])

# 2단계: 각 그룹별로 게인 추정 (방법 1 적용)
gain_table = {}
for condition, data_group in grouped_data.items():
    gains = estimate_gains_from_multiple_samples(data_group)
    gain_table[condition] = gains

# 3단계: 룩업 테이블 생성
# angle: -150 to -100, duty: 40-60 -> Kp=1.5, Ki=0.3, ...
# angle: -100 to -50, duty: 40-60 -> Kp=1.7, Ki=0.35, ...

# 4단계: 보간법으로 부드럽게 전환
def get_interpolated_gains(current_angle, current_duty):
    # 주변 조건의 게인들을 보간
    return interpolate_gains(gain_table, current_angle, current_duty)
```

**장점:**
- 비선형성을 명시적으로 다룹니다
- 각 동작 영역에서 최적 성능을 얻습니다

**단점:**
- 구현이 복잡합니다
- 불연속성 문제를 해결해야 합니다
- 많은 데이터가 필요합니다

---

### **방법 4: 데이터 기반 모델 프리 접근 (Model-Free RL)**

PID 게인을 직접 추정하지 않고, 데이터로부터 제어 정책을 학습합니다.

#### 접근 방식:
```python
# 강화학습 또는 지도학습 기반
# 입력: 현재 각도, 목표 각도, 각속도, 각가속도, 이전 오차 등
# 출력: 제어 입력 (duty)

# 시험 데이터를 학습 데이터로 변환
X_train = []  # [current_angle, target, velocity, error, integral_error]
y_train = []  # [control_output]

for test_data in all_test_data:
    # 데이터 변환
    ...

# 회귀 모델 학습 (예: Neural Network, Random Forest)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, y_train)

# 제어기로 사용
def controller(state):
    return model.predict([state])[0]
```

**장점:**
- 모델링 오차 없음
- 비선형성을 자연스럽게 처리

**단점:**
- PID 형태가 아니므로 기존 제어 프레임워크와 통합이 어려울 수 있습니다
- 해석 가능성이 떨어집니다

---

## **추천: 방법 1 + 방법 2 조합**

제 추천은 다음과 같습니다:

### Phase 1: 통계 기반 초기 게인 추정
1. 모든 시험 데이터에서 모델 파라미터 추정
2. 중앙값 또는 가중평균으로 대표 모델 선정
3. Ziegler-Nichols/Cohen-Coon/IMC로 초기 게인 계산

### Phase 2: 최적화 기반 미세 조정
1. Phase 1의 게인을 초기값으로 사용
2. 모든 샘플에서의 평균 오차를 최소화하는 게인 탐색
3. 제약 조건: 안정성 마진 확보

### Phase 3: 검증 및 리포트
1. 각 시험 케이스별 오차 분석
2. 최악의 경우(worst-case) 성능 평가
3. 신뢰 구간 제공

---

## 구현 예시 코드 구조

```python
class MultiSampleGainEstimator:
    def __init__(self, all_test_data):
        self.data = all_test_data
        self.model_params = []
        
    def estimate_all_models(self):
        """모든 샘플에서 모델 파라미터 추정"""
        for test in self.data:
            params = self.fit_model(test)
            if params['r_squared'] > 0.85:  # 품질 필터
                self.model_params.append(params)
    
    def get_representative_model(self, method='median'):
        """대표 모델 선정"""
        if method == 'median':
            K = np.median([p['K'] for p in self.model_params])
            tau = np.median([p['tau'] for p in self.model_params])
        elif method == 'weighted_avg':
            weights = [p['r_squared'] for p in self.model_params]
            K = np.average([p['K'] for p in self.model_params], weights=weights)
            tau = np.average([p['tau'] for p in self.model_params], weights=weights)
        
        return K, tau
    
    def estimate_initial_gains(self, method='imc'):
        """초기 게인 추정"""
        K, tau = self.get_representative_model()
        if method == 'imc':
            lambda_c = 1.5 * tau
            Kp = tau / (K * lambda_c)
            Ki = 1 / (K * lambda_c)
            Kd = 0
        # ... 다른 방법들
        return Kp, Ki, Kd
    
    def optimize_gains(self, initial_gains):
        """최적화 기반 미세 조정"""
        def objective(gains):
            errors = []
            for test in self.data:
                sim_response = self.simulate(test, gains)
                error = self.calculate_error(sim_response, test.actual)
                errors.append(error)
            return np.mean(errors)
        
        result = minimize(objective, initial_gains, 
                         bounds=[(0, 10), (0, 5), (0, 1), (0, 10)])
        return result.x
    
    def validate_gains(self, gains):
        """전체 샘플에 대해 검증"""
        errors = []
        for test in self.data:
            sim_response = self.simulate(test, gains)
            error = self.calculate_error(sim_response, test.actual)
            errors.append(error)
        
        return {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'max_error': np.max(errors),
            'min_error': np.min(errors),
            'confidence_95': np.percentile(errors, 95)
        }
```

