# 속도 기반 모델링: 굴착기의 올바른 접근법

## 🎯 사용자의 핵심 통찰

> "DC 게인을 속도 측면에서 정의해야 하지 않을까? 위치 측면에서 정의할 수 없는 이유는 굴착기의 경우 duty를 주면 실린더 끝단에 도달할 때까지 물리적 한계까지 움직이다 멈출 뿐이다."

**→ 완전히 맞습니다! 이것이 정답입니다!**

---

## 📚 이론적 배경

### 위치 제어 시스템 vs 속도 제어 시스템

#### 🚫 위치 제어 시스템 (이 프로젝트의 잘못된 가정)

```
시스템 특성:
- 목표 위치가 있음
- 도달하면 자연스럽게 정지
- 정상상태 = 위치 일정, 속도 0

전달함수 (위치 출력):
           K
G(s) = ─────────
       τs + 1

입력: Duty (스텝)
출력: 위치 (최종값으로 수렴)

예시: 서보 모터, 로봇 관절
```

#### ✅ 속도 제어 시스템 (굴착기의 실제 특성!)

```
시스템 특성:
- 목표 위치 없음 (물리적 한계만 존재)
- Duty 유지 → 계속 움직임 (등속도)
- 정상상태 = 속도 일정 ≠ 0

전달함수 (속도 출력):
           Kv
V(s) = ─────────
       τs + 1

입력: Duty (스텝)
출력: 속도 (일정 속도 유지)

예시: 자동차, 컨베이어 벨트, 굴착기 유압 실린더 ✅
```

---

## 🔍 굴착기 유압 시스템의 실제 동작

### 물리적 원리

```
유압 실린더 동작:

Duty → 유압 밸브 개도 → 유압 유량 → 실린더 속도

관계:
  Duty ∝ 밸브 개도
  밸브 개도 ∝ 유량
  유량 ∝ 속도

∴ Duty ∝ 속도  ✅

위치는?
  위치 = ∫ 속도 dt
  → Duty와 직접 관계 없음!
  → 물리적 한계까지 계속 움직임
```

### 실제 동작 예시

```
실험 상황:
1. Duty 70% 인가
2. 유압 밸브 70% 개방
3. 유량 발생 → 속도 약 25 deg/s (일정!)
4. 계속 움직임... 움직임... 움직임...
5. 물리적 한계(-134°) 도달
6. 기계적으로 멈춤 (속도 = 0)
7. 사용자가 Duty = 0으로 설정

문제:
- 5단계 이전: 속도 25 deg/s (정상상태!)
- 5단계 이후: 속도 0 (물리적 강제 정지)
→ 위치 기반 DC Gain은 의미 없음!
```

---

## 📊 데이터 재분석: 속도 관점

### Bucket In 데이터 (속도 기반)

```
Duty 40%:  평균 속도 -6.2 deg/s  → Kv = -6.2/40 = -0.155
Duty 70%:  평균 속도 -24.9 deg/s → Kv = -24.9/70 = -0.356
Duty 100%:평균 속도 -23.8 deg/s → Kv = -23.8/100 = -0.238

평균 Kv = -0.250 deg/s/%
변동계수 = 35%  (여전히 큼, 비선형성 존재)
```

**vs 위치 기반 (잘못된 접근):**
```
Duty 40%:  위치 변화 -177° → K = -4.42 (의미 없음!)
Duty 70%:  위치 변화 -177° → K = -2.53 (의미 없음!)
Duty 100%: 위치 변화 -177° → K = -1.77 (의미 없음!)
```

---

## ✅ 올바른 시스템 모델

### 1. 속도 기반 1차 시스템

```
입력: u(t) = Duty
출력: v(t) = 속도

전달함수:
           Kv
V(s) = ───────── × U(s)
       τv·s + 1

시간 영역:
v(t) = Kv × u × (1 - e^(-t/τv))

정상상태:
v(∞) = Kv × u  ✅

예시:
Duty 70%:
v(∞) = Kv × 70% = 0.356 × 70 = 24.9 deg/s  ✅
```

### 2. 위치는 속도의 적분

```
위치: θ(t) = ∫v(t)dt

θ(t) = Kv × u × [t - τv(1 - e^(-t/τv))]

특성:
- t → ∞ 일 때 θ → ∞ (무한정 증가!)
- 정상상태 없음 (물리적 한계가 없다면)
- DC Gain 정의 불가 ❌

실제:
- 물리적 한계에서 강제 정지
- 위치 = 한계값 (Duty와 무관)
```

---

## 🎯 V3 결과 재해석

### 현재 V3가 하는 것 (이미 올바름!)

```python
# scripts/run_integrated_analysis_v3.py

def extract_steady_state_velocities(self, data, angle_col):
    """정상상태 속도 추출"""
    # 속도 계산
    velocity = np.gradient(angles, time)
    
    # 중간 70% 구간 (정상상태)
    steady_start = int(len(velocity) * 0.15)
    steady_end = int(len(velocity) * 0.85)
    steady_velocity = np.median(velocity[steady_start:steady_end])
    
    return steady_velocity  ✅

def estimate_ff_gain(self, duty_velocities):
    """FF 게인 추정 (속도 기반!)"""
    # Duty = Kv × Velocity + Offset
    params, _ = curve_fit(linear_func, velocities, duties)
    Kv, offset = params
    return Kv  ✅
```

**→ V3는 이미 속도 기반 모델을 사용하고 있습니다!**

### 문제는 PID 게인 추정 부분

```python
# 잘못된 부분: 위치 기반 모델 피팅
def estimate_pid_gains_from_model():
    # 위치 데이터로 1차 시스템 피팅
    model_fitter.fit_first_order(time, angle)  ❌
    
    # DC Gain (위치/Duty)
    K = (final_angle - initial_angle) / duty  ❌
    
    # PID 계산
    Kp = τ / (K × λ)  ❌
```

**문제:**
- 위치로 피팅하면 K가 불안정
- 물리적 한계 때문에 K 정의 안 됨
- 모델 피팅 실패

---

## 🔧 올바른 PID 게인 추정 방법

### 방법 1: 속도 기반 모델 피팅 (권장)

```python
def estimate_pid_from_velocity_model(time, angle, duty):
    """
    속도 기반 모델로 PID 추정
    """
    # 1. 속도 계산
    velocity = np.gradient(angle, time)
    
    # 2. 속도로 1차 시스템 피팅
    def velocity_model(t, Kv, tau):
        return Kv * duty * (1 - np.exp(-t/tau))
    
    params, _ = curve_fit(velocity_model, time, velocity)
    Kv, tau = params
    
    # 3. 위치 제어를 위한 등가 변환
    # 위치 제어기 = 속도 제어기 + 적분기
    
    # 속도 루프 PID (내부)
    lambda_v = 2.0 * tau
    Kp_v = tau / (Kv * lambda_v)      # 속도 비례 게인
    Ki_v = 1 / (Kv * lambda_v)        # 속도 적분 게인
    
    # 위치 루프 PID (외부, cascaded)
    Kp_pos = Kp_v * Kv                # 위치 비례 게인
    Ki_pos = Ki_v * Kv                # 위치 적분 게인
    Kd_pos = 0
    
    return {
        'Kp': Kp_pos,
        'Ki': Ki_pos, 
        'Kd': Kd_pos,
        'Kv': Kv,
        'tau': tau
    }
```

### 방법 2: Cascaded 제어 구조

```
         ┌──────────────┐     ┌──────────────┐
목표위치 │ 위치 제어기   │ 목표 │ 속도 제어기   │ Duty
  θ_ref─┤  (P 또는 PI) ├─→v_ref┤  (PI)        ├─→ Valve
         │ Kp_pos       │     │ Kv, τ        │
         └──────────────┘     └──────────────┘
              ↑                      ↑
              θ (피드백)            v (피드백)

구조:
1. 외부 루프: 위치 제어 (느림, P 게인)
2. 내부 루프: 속도 제어 (빠름, PI 게인)

장점:
- 속도 제어는 선형적 (물리적 한계 영향 적음)
- 위치 제어는 단순 (비례만 사용)
- 안정성 높음
```

### 방법 3: FF + FB 구조 (이미 구현됨!)

```python
# 현재 V3의 접근법 (이미 올바름!)

# Feedforward (속도 기반)
u_ff = Kv × v_desired + K_offset  ✅

# Feedback (위치 오차)
u_fb = Kp × (θ_ref - θ) + Ki × ∫(θ_ref - θ)dt  

# 최종 제어
u_total = u_ff + u_fb

특징:
- FF가 주된 제어 (속도 명령 추종)
- FB는 보조 (오차 보정)
- FF가 정확하면 FB 게인 작아도 됨
```

---

## 📊 데이터 재분석 제안

### 스크립트 수정 제안

```python
# 새로운 분석 방법
def analyze_velocity_based_model():
    """
    속도 기반 시스템 식별
    """
    
    # 1. 각 Duty에서 정상상태 속도 추출 (이미 V3에서 함)
    steady_velocities = extract_steady_state_velocities()
    
    # 2. Kv 추정 (이미 V3에서 함)
    Kv, offset = estimate_ff_gain(steady_velocities)
    
    # 3. 과도 응답으로 τ 추정 (새로운 부분)
    tau = estimate_time_constant_from_velocity(velocity_data)
    
    # 4. 속도 루프 PID 계산
    lambda_v = 2.0 * tau
    Kp_v = tau / (Kv * lambda_v)
    Ki_v = 1 / (Kv * lambda_v)
    
    # 5. 위치 루프 게인 변환
    # 간단한 방법: 위치 P 게인만 사용
    Kp_pos = 1.0 / Kv  # 단위: (%·s/deg)
    Ki_pos = 0.1 * Kp_pos
    
    return {
        'Kv': Kv,           # 속도 DC 게인
        'tau': tau,         # 속도 시정수
        'Kp': Kp_pos,      # 위치 비례 게인
        'Ki': Ki_pos       # 위치 적분 게인
    }
```

---

## 🎯 실무적 접근법

### 현재 상황 평가

```
V3가 이미 하고 있는 것 (올바름):
✅ 속도 추출
✅ FF 게인 (Kv) 계산
✅ 통계적 분석

V3가 잘못하고 있는 것:
❌ 위치 기반 모델 피팅
❌ 위치 DC Gain 사용
❌ 물리적 한계 무시
```

### 권장 수정사항

**Option 1: 간단한 방법 (즉시 가능)**

```python
# FF 게인은 그대로 사용 (이미 올바름)
Kv = V3_result['Kv']  # deg/s/%

# 위치 PID는 경험적 공식
Kp = 1.0 / Kv         # %·s/deg (단위 변환)
Ki = 0.1 * Kp
Kd = 0

# 예시:
Kv = 0.356 deg/s/%
Kp = 1/0.356 = 2.81 %·s/deg
Ki = 0.281 %·s²/deg
```

**Option 2: Cascaded 제어 (더 정확)**

```python
# 내부 루프: 속도 제어
Kv = V3_result['Kv']
tau = estimate_from_transient()  # 과도 응답에서 추정

# 속도 PI 게인
Kp_v = tau / (Kv × 2*tau)
Ki_v = 1 / (Kv × 2*tau)

# 외부 루프: 위치 P 제어
Kp_pos = 0.5  # 경험값
Ki_pos = 0.0  # 속도 루프가 이미 적분기 역할
```

**Option 3: 현재 게인 재해석 (타협안)**

```python
# 현재 V3 결과를 그대로 사용하되
# 물리적 의미를 속도 관점으로 재해석

# Out/Up 방향 (성공한 케이스):
# → 충분한 범위, 포화 영향 적음
# → 위치 기반 게인도 어느 정도 유효
Arm_Out_PID = use_as_is()

# In/Down 방향 (실패한 케이스):
# → FF 게인으로 재계산
Kv = V3_result['Kv']
Kp = 1.0 / Kv
Ki = 0.1 * Kp
```

---

## 📈 검증 방법

### 시뮬레이션 테스트

```python
def validate_velocity_based_model():
    """
    속도 기반 모델 검증
    """
    
    # 1. 실제 데이터 로드
    time, angle, duty = load_test_data()
    velocity_measured = np.gradient(angle, time)
    
    # 2. 모델 시뮬레이션
    velocity_model = simulate_velocity_model(time, duty, Kv, tau)
    
    # 3. 비교
    error = velocity_measured - velocity_model
    rmse = np.sqrt(np.mean(error**2))
    r_squared = calculate_r_squared(velocity_measured, velocity_model)
    
    print(f"속도 모델 적합도: R² = {r_squared:.4f}")
    
    # 4. 위치 적분
    position_model = cumtrapz(velocity_model, time)
    position_measured = angle - angle[0]
    
    # 물리적 한계 전까지만 비교
    valid_range = position_model < saturation_limit
    
    return r_squared, rmse
```

---

## 🎯 최종 권장사항

### 즉시 적용 가능한 방법

```python
# 1. V3 FF 게인 사용 (이미 올바름)
FF_gains = load_v3_ff_gains()  ✅

# 2. 위치 PID는 FF 게인에서 유도
for axis in all_axes:
    Kv = FF_gains[axis]['Kv']
    
    # 간단한 공식
    Kp = 1.0 / Kv  # 속도→위치 변환
    Ki = 0.1 * Kp
    Kd = 0
    
    PID_gains[axis] = {'Kp': Kp, 'Ki': Ki, 'Kd': Kd}

# 3. 안전계수 적용
PID_gains = apply_safety_factor(PID_gains, factor=0.7)

# 4. 현장 미세 조정
```

### 장기적 개선 방향

1. **속도 기반 모델 피팅 구현**
2. **Cascaded 제어 구조 고려**
3. **비선형성 보상 (포화, 마찰)**
4. **적응 제어 (부하 변화 대응)**

---

## 📚 결론

### 핵심 통찰

**사용자의 지적이 100% 정확합니다:**

1. ✅ **굴착기는 속도 시스템이다**
   - Duty → 속도 (직접 관계)
   - 위치 = 속도의 적분 (간접 관계)

2. ✅ **위치 DC Gain은 정의 불가**
   - 물리적 한계까지 계속 움직임
   - 정상상태 위치 없음
   - Duty와 최종 위치 비례 안 함

3. ✅ **속도 DC Gain(Kv)이 올바른 접근**
   - 정상상태 속도 존재
   - Duty와 속도 비례
   - FF 게인으로 이미 추정됨

4. ✅ **V3는 이미 올바른 방향**
   - FF 게인 = 속도 기반 ✅
   - 문제는 PID 추정 부분만

### 실무적 해결책

```
단기 (즉시):
- V3 FF 게인 사용
- 위치 PID = 1/Kv (경험적 공식)
- 안전계수 0.7

중기 (1주):
- 속도 기반 모델 피팅 구현
- 과도 응답에서 τ 추정
- 더 정확한 PID 계산

장기 (1개월):
- Cascaded 제어 구조
- 비선형 보상
- 적응 제어
```

**이론과 실무가 만나는 지점: 시스템의 본질을 이해하는 것이 가장 중요합니다!** 🎯

