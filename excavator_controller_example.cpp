/**
 * @file excavator_controller_example.cpp
 * @brief 굴착기 PID/FF 제어기 구현 예제 (ROS2)
 * @date 2025-10-14
 * 
 * 속도 기반 모델링을 사용한 PID/FF 제어기
 * YAML 파일: excavator_pid_ff_gains.yaml
 */

#include <rclcpp/rclcpp.hpp>
#include <algorithm>
#include <cmath>

class ExcavatorController : public rclcpp::Node
{
public:
    ExcavatorController() : Node("excavator_controller")
    {
        // YAML 파일에서 파라미터 로드
        loadParameters();
        
        // 타이머 설정 (100Hz)
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10),
            std::bind(&ExcavatorController::controlLoop, this));
        
        RCLCPP_INFO(this->get_logger(), "Excavator Controller initialized");
    }

private:
    // PID 게인 구조체
    struct PIDGains {
        double kp;
        double ki;
        double kd;
    };
    
    // FF 게인 구조체
    struct FFGains {
        double kv;
        double k_offset;
    };
    
    // 제어 상태 구조체
    struct ControlState {
        double integral = 0.0;
        double prev_error = 0.0;
        double prev_time = 0.0;
    };
    
    // 축별 게인
    PIDGains pid_arm_in_, pid_arm_out_;
    PIDGains pid_boom_up_, pid_boom_down_;
    PIDGains pid_bucket_in_, pid_bucket_out_;
    
    FFGains ff_arm_in_, ff_arm_out_;
    FFGains ff_boom_up_, ff_boom_down_;
    FFGains ff_bucket_in_, ff_bucket_out_;
    
    // 축별 제어 상태
    ControlState state_arm_, state_boom_, state_bucket_;
    
    // 제어 파라미터
    double integral_limit_ = 10.0;
    double sample_time_ = 0.01;
    bool use_safe_gains_ = true;  // 초기에는 안전 게인 사용
    
    rclcpp::TimerBase::SharedPtr timer_;
    
    void loadParameters()
    {
        // Full Gains (나중에 안전 게인에서 전환)
        pid_arm_in_ = {3.691, 1.846, 0.0};
        pid_arm_out_ = {3.597, 1.799, 0.0};
        pid_boom_up_ = {11.431, 5.716, 0.0};
        pid_boom_down_ = {6.382, 3.191, 0.0};
        pid_bucket_in_ = {1.654, 0.827, 0.0};
        pid_bucket_out_ = {1.569, 0.784, 0.0};
        
        // 초기에는 안전 게인 사용 (Ki 50%)
        if (use_safe_gains_) {
            pid_arm_in_.ki = 0.923;
            pid_arm_out_.ki = 0.899;
            pid_boom_up_.ki = 2.858;
            pid_boom_down_.ki = 1.595;
            pid_bucket_in_.ki = 0.414;
            pid_bucket_out_.ki = 0.392;
        }
        
        // FF 게인
        ff_arm_in_ = {2.709, 36.0};
        ff_arm_out_ = {1.390, 40.2};
        ff_boom_up_ = {4.374, 35.9};
        ff_boom_down_ = {3.134, 35.4};
        ff_bucket_in_ = {6.045, 0.0};  // K_offset=0 (원래 -22.6)
        ff_bucket_out_ = {0.850, 40.6};
        
        RCLCPP_INFO(this->get_logger(), "Parameters loaded (Safe mode: %s)", 
                    use_safe_gains_ ? "ON" : "OFF");
    }
    
    /**
     * @brief PID 제어 계산
     * @param gains PID 게인
     * @param error 위치 오차 (deg)
     * @param state 제어 상태 (적분, 이전 오차 등)
     * @param dt 샘플 시간 (s)
     * @return PID 출력 (%)
     */
    double calculatePID(const PIDGains& gains, 
                       double error, 
                       ControlState& state, 
                       double dt)
    {
        // Proportional
        double p_term = gains.kp * error;
        
        // Integral (with anti-windup)
        state.integral += error * dt;
        state.integral = std::clamp(state.integral, -integral_limit_, integral_limit_);
        double i_term = gains.ki * state.integral;
        
        // Derivative
        double derivative = (error - state.prev_error) / dt;
        double d_term = gains.kd * derivative;
        
        state.prev_error = error;
        
        return p_term + i_term + d_term;
    }
    
    /**
     * @brief FF 제어 계산
     * @param gains FF 게인
     * @param target_velocity 목표 속도 (deg/s)
     * @return FF 출력 (%)
     */
    double calculateFF(const FFGains& gains, double target_velocity)
    {
        // u_ff = kv * velocity + k_offset
        return gains.kv * std::abs(target_velocity) + gains.k_offset;
    }
    
    /**
     * @brief 축 제어 (PID + FF)
     * @param target_pos 목표 위치 (deg)
     * @param current_pos 현재 위치 (deg)
     * @param gains_pos 양방향 PID 게인 (In/Up)
     * @param gains_neg 음방향 PID 게인 (Out/Down)
     * @param ff_pos 양방향 FF 게인
     * @param ff_neg 음방향 FF 게인
     * @param state 제어 상태
     * @param valve_out 밸브 방향 출력 (true: In/Up, false: Out/Down)
     * @return Duty 출력 (0~100%)
     */
    double controlAxis(double target_pos,
                      double current_pos,
                      const PIDGains& gains_pos,
                      const PIDGains& gains_neg,
                      const FFGains& ff_pos,
                      const FFGains& ff_neg,
                      ControlState& state,
                      bool& valve_out)
    {
        double dt = sample_time_;
        
        // 1. 목표 속도 계산
        double error = target_pos - current_pos;
        double target_velocity = error / dt;
        
        // 2. 방향 결정
        bool is_positive = (error > 0);
        valve_out = is_positive;
        
        // 3. FF 계산
        double u_ff = 0.0;
        if (is_positive) {
            u_ff = calculateFF(ff_pos, target_velocity);
        } else {
            u_ff = calculateFF(ff_neg, target_velocity);
        }
        
        // 4. PID 계산
        double u_pid = 0.0;
        if (is_positive) {
            u_pid = calculatePID(gains_pos, error, state, dt);
        } else {
            u_pid = calculatePID(gains_neg, std::abs(error), state, dt);
        }
        
        // 5. 총 제어 입력
        double u_total = u_ff + u_pid;
        
        // 6. Duty saturation (0~100%)
        double duty = std::clamp(std::abs(u_total), 0.0, 100.0);
        
        return duty;
    }
    
    void controlLoop()
    {
        // 예제: 목표 위치 설정 (실제로는 ROS topic에서 받아옴)
        double target_arm = 45.0;      // deg
        double target_boom = 30.0;     // deg
        double target_bucket = 20.0;   // deg
        
        // 현재 위치 (실제로는 센서에서 받아옴)
        double current_arm = 40.0;     // deg
        double current_boom = 25.0;    // deg
        double current_bucket = 18.0;  // deg
        
        bool valve_arm, valve_boom, valve_bucket;
        
        // Arm 제어
        double duty_arm = controlAxis(
            target_arm, current_arm,
            pid_arm_in_, pid_arm_out_,
            ff_arm_in_, ff_arm_out_,
            state_arm_, valve_arm
        );
        
        // Boom 제어
        double duty_boom = controlAxis(
            target_boom, current_boom,
            pid_boom_up_, pid_boom_down_,
            ff_boom_up_, ff_boom_down_,
            state_boom_, valve_boom
        );
        
        // Bucket 제어
        double duty_bucket = controlAxis(
            target_bucket, current_bucket,
            pid_bucket_in_, pid_bucket_out_,
            ff_bucket_in_, ff_bucket_out_,
            state_bucket_, valve_bucket
        );
        
        // 로그 출력 (1초마다)
        static int counter = 0;
        if (++counter >= 100) {
            RCLCPP_INFO(this->get_logger(), 
                "Arm: %.1f%% (%s), Boom: %.1f%% (%s), Bucket: %.1f%% (%s)",
                duty_arm, valve_arm ? "IN" : "OUT",
                duty_boom, valve_boom ? "UP" : "DOWN",
                duty_bucket, valve_bucket ? "IN" : "OUT");
            counter = 0;
        }
        
        // 실제로는 여기서 밸브 명령 publish
        // publishValveCommands(duty_arm, valve_arm, duty_boom, valve_boom, ...);
    }
    
public:
    /**
     * @brief 안전 게인에서 Full 게인으로 전환
     * @param ki_ratio Ki 비율 (0.5~1.0)
     */
    void setKiRatio(double ki_ratio)
    {
        ki_ratio = std::clamp(ki_ratio, 0.5, 1.0);
        
        pid_arm_in_.ki = 1.846 * ki_ratio;
        pid_arm_out_.ki = 1.799 * ki_ratio;
        pid_boom_up_.ki = 5.716 * ki_ratio;
        pid_boom_down_.ki = 3.191 * ki_ratio;
        pid_bucket_in_.ki = 0.827 * ki_ratio;
        pid_bucket_out_.ki = 0.784 * ki_ratio;
        
        RCLCPP_INFO(this->get_logger(), "Ki ratio updated to %.1f%%", ki_ratio * 100);
    }
    
    /**
     * @brief 적분기 리셋 (비상 정지 또는 큰 오차 시)
     */
    void resetIntegrators()
    {
        state_arm_.integral = 0.0;
        state_boom_.integral = 0.0;
        state_bucket_.integral = 0.0;
        
        RCLCPP_WARN(this->get_logger(), "All integrators reset");
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ExcavatorController>();
    
    // 예제: 5초 후 Ki 60%로 증가
    std::thread([node]() {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        node->setKiRatio(0.6);
        
        // 10초 후 Ki 80%
        std::this_thread::sleep_for(std::chrono::seconds(5));
        node->setKiRatio(0.8);
        
        // 15초 후 Ki 100%
        std::this_thread::sleep_for(std::chrono::seconds(5));
        node->setKiRatio(1.0);
    }).detach();
    
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}


