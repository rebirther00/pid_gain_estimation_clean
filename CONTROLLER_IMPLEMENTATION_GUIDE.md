# 제어기 구현 가이드

## 날짜: 2025-10-13

---

## 📋 목차

1. [Kd가 0인 이유](#kd가-0인-이유)
2. [FF 게인의 K_offset과 r_squared 활용법](#ff-게인의-k_offset과-r_squared-활용법)
3. [제어기 구현 예제](#제어기 구현-예제)
4. [실전 튜닝 팁](#실전-튜닝-팁)

---

## 🤔 Kd가 0인 이유

### 1. IMC 튜닝 방법의 특성

**코드 위치**: `src/tuning/pid_tuner_imc.py:46`

```python
def tune(self, K: float, tau: float, lambda_c: float = None):
    # IMC 공식 (1차 시스템)
    Kp = tau / (K * lambda_c)
    Ki = 1 / (K * lambda_c)
    Kd = 0  # ← 1차 시스템에서는 D 게인 불필요
```

### 2. 1차 시스템의 특성

굴삭기 유압 시스템을 **1차 지연 시스템**으로 모델링했습니다:

```
G(s) = K / (τs + 1)
```

**1차 시스템에서 Kd=0인 이유**:
- 시스템에 관성(2차 항)이 명시적으로 모델링되지 않음
- 오버슈트가 거의 없는 시스템
- D 게인은 주로 2차 시스템의 오버슈트 억제에 사용

### 3. 실측 데이터 검증

모든 축에서 **R² > 0.85**로 1차 모델이 실제 시스템을 충분히 잘 설명합니다:

| 축 | R² | 모델 적합도 |
|---|----|----|
| Arm_In | 0.93 | 매우 좋음 |
| Arm_Out | 0.94 | 매우 좋음 |
| Boom_Up | 0.88 | 좋음 |
| Boom_Down | 0.86 | 좋음 |
| Bucket_In | 0.90 | 매우 좋음 |
| Bucket_Out | 0.95 | 매우 좋음 |

### 4. Kd를 추가해야 하는 경우

다음 상황에서만 Kd를 고려하세요:

1. ❌ **현재는 불필요**: 1차 모델로 충분히 설명됨
2. ✅ **추후 필요시**: 
   - 실제 제어에서 오버슈트 발생
   - 고주파 노이즈 없는 경우
   - 빠른 응답이 필요한 경우

**권장 초기값** (필요시):
```python
Kd = 0.1 * Kp * tau  # 보수적 시작
```

---

## 🎯 FF 게인의 K_offset과 r_squared 활용법

### 1. FF 게인 구성 요소

**코드 위치**: `scripts/post_process_v3_results.py:263-278`

```python
def linear_func(velocity, kv, offset):
    return kv * velocity + offset

# 피팅 결과:
# duty = Kv * velocity + K_offset
```

**출력 예시** (Arm_In):
```json
"ff_gain": {
  "Kv": 1.289,           // 속도 게인
  "K_offset": 41.137,    // 기저 duty
  "r_squared": 0.816     // 모델 품질
}
```

### 2. 각 값의 물리적 의미

#### Kv (속도 게인)
- **단위**: `%/(deg/s)` 또는 `PWM/(rad/s)`
- **의미**: 목표 각속도를 달성하기 위한 duty 변화율
- **제어식**: `duty_ff = Kv * desired_velocity`

**예시** (Arm_In):
```
Kv = 1.289 %/(deg/s)
목표 속도 = 10 deg/s
→ duty_ff = 1.289 * 10 = 12.89 %
```

#### K_offset (기저 duty)
- **단위**: `%` 또는 `PWM`
- **의미**: 시스템을 움직이기 시작하는 최소 duty
- **원인**: 
  - 정지 마찰력 (static friction)
  - 중력 보상
  - 밸브 데드존
  - 유압 누설

**예시** (Arm_In):
```
K_offset = 41.137 %
→ 41% 이하에서는 거의 움직이지 않음
→ 속도 0이어도 41% duty 필요 (중력/마찰 보상)
```

#### r_squared (모델 품질)
- **범위**: 0 ~ 1
- **의미**: 선형 모델의 설명력
- **해석**:
  - **r² > 0.8**: 선형 관계 강함 → FF 신뢰 가능
  - **r² < 0.5**: 비선형성 강함 → FF 보정 필요

**실측 결과**:
| 축 | r² | 평가 | 신뢰도 |
|---|----|----|--------|
| Arm_In | 0.82 | 좋음 | 높음 |
| Arm_Out | 0.67 | 보통 | 중간 |
| Boom_Up | 0.80 | 좋음 | 높음 |
| Boom_Down | 0.79 | 좋음 | 높음 |
| Bucket_In | 0.64 | 보통 | 중간 |
| Bucket_Out | 0.64 | 보통 | 중간 |

### 3. 제어기 반영 방법

#### ✅ 방법 1: Offset 포함 (권장!)

```python
# Feedforward 계산
duty_ff = Kv * desired_velocity + K_offset

# PID 계산
error = desired_angle - current_angle
duty_pid = Kp * error + Ki * integral + Kd * derivative

# 최종 제어 입력
duty_total = duty_ff + duty_pid

# 제한
duty_output = np.clip(duty_total, 0, 100)  # 0~100%
```

**장점**:
- 정지 마찰, 중력 자동 보상
- 빠른 초기 응답
- PID 부담 감소

**단점**:
- 정지 시 불필요한 출력 (안전 로직 필요)

#### ⚠️ 방법 2: Offset 제외

```python
# Feedforward (offset 없이)
duty_ff = Kv * desired_velocity  # K_offset 제외

# 나머지는 동일
duty_total = duty_ff + duty_pid
```

**장점**:
- 정지 시 출력 없음 (안전)

**단점**:
- 초기 응답 느림
- PID가 정지 마찰/중력 극복 필요
- Integral windup 위험

#### 🎯 방법 3: 조건부 Offset (추천!)

```python
# 임계 속도 설정
velocity_threshold = 0.5  # deg/s

if abs(desired_velocity) > velocity_threshold:
    # 움직일 때만 offset 적용
    duty_ff = Kv * desired_velocity + K_offset
else:
    # 정지 시 offset 제외
    duty_ff = 0

duty_total = duty_ff + duty_pid
```

**장점**:
- 움직일 때 빠른 응답
- 정지 시 안전
- 최적의 균형

### 4. r_squared 활용

#### 높은 r² (> 0.8): 직접 사용

```python
# Boom Up/Down (r² ~ 0.8): 그대로 사용
duty_ff = Kv * velocity + K_offset
```

#### 낮은 r² (< 0.7): 보정 적용

```python
# Bucket (r² ~ 0.64): 보수적 사용
confidence = r_squared  # 0.64
duty_ff = confidence * (Kv * velocity + K_offset)
```

또는 **FF 제한**:

```python
# FF 기여도 제한
duty_ff_raw = Kv * velocity + K_offset
duty_ff = np.clip(duty_ff_raw, 0, 50)  # 최대 50%로 제한
```

---

## 🛠️ 제어기 구현 예제

### 기본 PID + FF 제어기 (C++)

```cpp
class HydraulicAxisController {
private:
    // PID 게인
    double Kp, Ki, Kd;
    
    // FF 게인
    double Kv, K_offset;
    
    // PID 상태
    double integral = 0.0;
    double prev_error = 0.0;
    
    // 제한
    double duty_min = 0.0;
    double duty_max = 100.0;
    double integral_max = 10.0;  // Anti-windup
    
public:
    HydraulicAxisController(double kp, double ki, double kd, 
                           double kv, double k_offset) 
        : Kp(kp), Ki(ki), Kd(kd), Kv(kv), K_offset(k_offset) {}
    
    double compute(double desired_angle, double current_angle,
                   double desired_velocity, double dt) {
        // 1. Feedforward 계산
        double duty_ff = 0.0;
        if (abs(desired_velocity) > 0.5) {  // 임계 속도
            duty_ff = Kv * desired_velocity + K_offset;
        }
        
        // 2. PID 계산
        double error = desired_angle - current_angle;
        
        // Proportional
        double P = Kp * error;
        
        // Integral (anti-windup)
        integral += error * dt;
        integral = std::clamp(integral, -integral_max, integral_max);
        double I = Ki * integral;
        
        // Derivative
        double derivative = (error - prev_error) / dt;
        double D = Kd * derivative;
        
        prev_error = error;
        
        double duty_pid = P + I + D;
        
        // 3. 최종 출력
        double duty_total = duty_ff + duty_pid;
        
        // 4. 제한
        return std::clamp(duty_total, duty_min, duty_max);
    }
    
    void reset() {
        integral = 0.0;
        prev_error = 0.0;
    }
};
```

### 사용 예시

```cpp
// 초기화 (Arm In)
HydraulicAxisController arm_in_ctrl(
    3.87,    // Kp
    0.039,   // Ki
    0.0,     // Kd
    1.289,   // Kv
    41.137   // K_offset
);

// 제어 루프 (예: 100Hz)
double dt = 0.01;  // 10ms

while (control_active) {
    // 센서 읽기
    double current_angle = read_angle_sensor();
    
    // 목표 계산
    double desired_angle = trajectory.get_angle(time);
    double desired_velocity = trajectory.get_velocity(time);
    
    // 제어 계산
    double duty = arm_in_ctrl.compute(
        desired_angle, 
        current_angle,
        desired_velocity,
        dt
    );
    
    // 출력
    set_duty(duty);
    
    time += dt;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
}
```

### Python 예시 (ROS2)

```python
import numpy as np

class HydraulicAxisController:
    def __init__(self, kp, ki, kd, kv, k_offset):
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd
        self.Kv = kv
        self.K_offset = k_offset
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.integral_max = 10.0
        
    def compute(self, desired_angle, current_angle, 
                desired_velocity, dt):
        """
        제어 계산
        
        Args:
            desired_angle: 목표 각도 [deg]
            current_angle: 현재 각도 [deg]
            desired_velocity: 목표 각속도 [deg/s]
            dt: 샘플링 시간 [s]
            
        Returns:
            duty: 제어 출력 [%]
        """
        # Feedforward
        duty_ff = 0.0
        if abs(desired_velocity) > 0.5:
            duty_ff = self.Kv * desired_velocity + self.K_offset
        
        # PID
        error = desired_angle - current_angle
        
        # P
        P = self.Kp * error
        
        # I (anti-windup)
        self.integral += error * dt
        self.integral = np.clip(self.integral, 
                               -self.integral_max, 
                               self.integral_max)
        I = self.Ki * self.integral
        
        # D
        derivative = (error - self.prev_error) / dt
        D = self.Kd * derivative
        self.prev_error = error
        
        duty_pid = P + I + D
        
        # 총 출력
        duty_total = duty_ff + duty_pid
        
        # 제한
        return np.clip(duty_total, 0, 100)
    
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

# ROS2 노드에서 사용
class ExcavatorController(Node):
    def __init__(self):
        super().__init__('excavator_controller')
        
        # 각 축별 제어기 초기화
        self.arm_in = HydraulicAxisController(
            kp=3.87, ki=0.039, kd=0.0,
            kv=1.289, k_offset=41.137
        )
        
        self.arm_out = HydraulicAxisController(
            kp=2.92, ki=0.029, kd=0.0,
            kv=1.389, k_offset=40.164
        )
        
        # ... 다른 축들도 동일
        
        self.timer = self.create_timer(0.01, self.control_callback)
    
    def control_callback(self):
        # 제어 로직
        pass
```

---

## 🎓 실전 튜닝 팁

### 1. 초기 테스트

#### Step 1: FF만 테스트
```python
# PID 끄고 FF만 테스트
duty = Kv * desired_velocity + K_offset
```

**확인사항**:
- 정지 마찰 극복 여부
- 목표 속도 달성 여부
- 오버슈트 여부

#### Step 2: FF + P만 테스트
```python
duty = (Kv * desired_velocity + K_offset) + Kp * error
```

**확인사항**:
- 정상 상태 오차
- 응답 속도
- 진동 여부

#### Step 3: FF + PI 테스트
```python
duty = (Kv * velocity + K_offset) + Kp * error + Ki * integral
```

**확인사항**:
- 정상 상태 오차 제거
- Integral windup
- 응답 속도 vs 안정성

### 2. 게인 보정

#### Kp 보정 (응답 속도)

```python
# 너무 느리면
Kp = Kp * 1.2  # 20% 증가

# 진동하면
Kp = Kp * 0.8  # 20% 감소
```

#### Ki 보정 (정상 상태 오차)

```python
# 정상 상태 오차 있으면
Ki = Ki * 1.5  # 50% 증가

# Overshoot 크면
Ki = Ki * 0.5  # 50% 감소
```

#### Kv 보정 (속도 추종)

```python
# 목표 속도보다 느리면
Kv = Kv * 1.1  # 10% 증가

# 목표 속도보다 빠르면
Kv = Kv * 0.9  # 10% 감소
```

### 3. 안전 계수

**보수적 시작** (첫 테스트):
```yaml
Kp: Kp_calculated * 0.5  # 50%
Ki: Ki_calculated * 0.5  # 50%
Kv: Kv_calculated * 0.7  # 70%
K_offset: K_offset_calculated * 0.8  # 80%
```

**점진적 증가**:
1. 테스트 → 평가
2. 안전하면 10% 증가
3. 목표 성능 달성 시 중단

### 4. 방향별 차이 고려

**비대칭 제어**가 필요합니다!

```python
class ArmController:
    def __init__(self):
        # In 방향 (Kp = 3.87)
        self.ctrl_in = HydraulicAxisController(
            kp=3.87, ki=0.039, kd=0.0,
            kv=1.289, k_offset=41.137
        )
        
        # Out 방향 (Kp = 2.92, -24%)
        self.ctrl_out = HydraulicAxisController(
            kp=2.92, ki=0.029, kd=0.0,
            kv=1.389, k_offset=40.164
        )
    
    def compute(self, desired_angle, current_angle, 
                desired_velocity, dt):
        # 방향 판단
        if desired_velocity > 0:
            return self.ctrl_out.compute(...)
        else:
            return self.ctrl_in.compute(...)
```

### 5. 모니터링

**실시간 확인사항**:
```python
# 로깅
log_data = {
    'time': time,
    'desired_angle': desired_angle,
    'current_angle': current_angle,
    'error': error,
    'duty_ff': duty_ff,
    'duty_pid': duty_pid,
    'duty_total': duty_total,
    'P': P, 'I': I, 'D': D
}
```

**성능 지표**:
- **추종 오차**: `mean(abs(error))`
- **정상 상태 오차**: `error[-100:]` 평균
- **오버슈트**: `max(angle) - desired_angle`
- **정착 시간**: 오차 < 5% 도달 시간

---

## 📊 최종 권장 설정

### 실제 적용 (안전 계수 0.8 적용)

```yaml
Controllers:
  Arm:
    In:
      PID: {Kp: 3.10, Ki: 0.031, Kd: 0.0}
      FF: {Kv: 1.03, K_offset: 32.9}
      confidence: high  # r² = 0.82
    
    Out:
      PID: {Kp: 2.34, Ki: 0.023, Kd: 0.0}
      FF: {Kv: 1.11, K_offset: 32.1}
      confidence: medium  # r² = 0.67
  
  Boom:
    Up:
      PID: {Kp: 8.54, Ki: 0.086, Kd: 0.0}
      FF: {Kv: 2.27, K_offset: 32.7}
      confidence: high  # r² = 0.80
    
    Down:
      PID: {Kp: 4.48, Ki: 0.045, Kd: 0.0}
      FF: {Kv: 2.20, K_offset: 0.0}  # offset 없음
      confidence: high  # r² = 0.79
  
  Bucket:
    In:
      PID: {Kp: 1.38, Ki: 0.014, Kd: 0.0}
      FF: {Kv: 0.40, K_offset: 0.0}
      confidence: medium  # r² = 0.64
    
    Out:
      PID: {Kp: 1.22, Ki: 0.014, Kd: 0.0}
      FF: {Kv: 0.68, K_offset: 32.5}
      confidence: medium  # r² = 0.64
```

---

## ⚠️ 주의사항

1. **안전 제한**
   ```cpp
   // 항상 duty 제한
   duty = std::clamp(duty, 0.0, 100.0);
   
   // 급격한 변화 제한
   double duty_rate_max = 20.0;  // %/s
   double duty_change = duty - prev_duty;
   duty_change = std::clamp(duty_change, 
                            -duty_rate_max * dt, 
                            duty_rate_max * dt);
   ```

2. **Anti-windup**
   ```cpp
   // Integral 제한 (필수!)
   integral = std::clamp(integral, -10.0, 10.0);
   ```

3. **방향 전환 시 적분 리셋**
   ```cpp
   if (direction_changed) {
       controller.reset();
   }
   ```

4. **비상 정지**
   ```cpp
   if (emergency_stop) {
       duty = 0;
       controller.reset();
   }
   ```

---

## 📝 요약

### Kd = 0인 이유
- ✅ 1차 시스템 모델 (R² > 0.85)
- ✅ IMC 튜닝 방법의 특성
- ❌ 2차 항(관성) 없음
- 💡 필요시 `Kd = 0.1 * Kp * tau`로 시작

### K_offset과 r_squared 활용
- **K_offset**: 정지 마찰/중력 보상용 기저 duty
  - 사용: `duty_ff = Kv * velocity + K_offset`
  - 조건부 사용 권장 (움직일 때만)
  
- **r_squared**: FF 모델 신뢰도
  - r² > 0.8: 그대로 사용
  - r² < 0.7: 보수적 사용 (0.8배 등)

### 제어 구조
```
duty_total = duty_feedforward + duty_pid
           = (Kv * velocity + K_offset) + (Kp * e + Ki * ∫e + Kd * de/dt)
```

---

**성공적인 제어를 위한 핵심**: 
안전 계수로 시작 → 점진적 증가 → 실시간 모니터링! 🎯

