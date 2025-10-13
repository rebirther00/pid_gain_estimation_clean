"""
통합 분석 모듈 V3 (디버깅/검증 기능 추가)
- 파일별 상세 로그 출력
- 샘플 데이터 시각화 (10개당 1개)
- 파싱 결과 CSV 출력
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.optimize import curve_fit
import logging
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경에서 사용
import matplotlib.pyplot as plt
from pathlib import Path

from ..preprocessing.velocity_estimator import VelocityEstimator
from .model_fitting import ModelFitter

logger = logging.getLogger(__name__)


class IntegratedAnalyzerV3:
    """통합 분석 클래스 V3 - 디버깅/검증 기능 추가"""
    
    def __init__(self, output_dir: str = 'output/integrated_v3', clean_debug: bool = True):
        self.velocity_estimator = VelocityEstimator(method='central_diff')
        self.model_fitter = ModelFitter()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 디버깅용 디렉토리
        self.debug_dir = self.output_dir / 'debug'
        
        # 기존 debug 폴더 삭제 (새로 시작)
        if clean_debug and self.debug_dir.exists():
            import shutil
            shutil.rmtree(self.debug_dir)
            logger.info(f"기존 debug 폴더 삭제: {self.debug_dir}")
        
        self.debug_dir.mkdir(exist_ok=True)
        self.plots_dir = self.debug_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)
        self.csv_dir = self.debug_dir / 'parsed_data'
        self.csv_dir.mkdir(exist_ok=True)
        
        # 파일 처리 통계
        self.file_stats = []
        self.sample_count = 0  # 샘플링 카운터
        self.normal_count = 0  # 정상 데이터 카운터
        self.abnormal_count = 0  # 비정상 데이터 카운터
    
    def extract_steady_state_velocities(self, parsed_group: Dict) -> Dict:
        """
        각 duty별로 정상 상태 속도 추출 (상세 로그 추가)
        """
        group_name = parsed_group['group_name']
        logger.info(f"\n[{group_name}] 정상 상태 속도 추출")
        
        result = {
            'duty_values': [],
            'velocities': {
                'Single_High': [],
                'Single_Low': [],
                'Couple_High': [],
                'Couple_Low': []
            },
            'all_velocities': [],
            'parsed_data_samples': []  # CSV 출력용
        }
        
        duties = sorted(parsed_group['data_by_duty'].keys())
        
        for duty in duties:
            duty_data = parsed_group['data_by_duty'][duty]
            velocities_for_duty = []
            
            for mode in ['Single', 'Couple']:
                for load in ['High', 'Low']:
                    data_info = duty_data[mode][load]
                    
                    # 파일 정보 로깅
                    file_path = data_info.get('file', 'Unknown') if data_info else 'None'
                    file_name = Path(file_path).name if file_path != 'Unknown' and file_path != 'None' else file_path
                    
                    if data_info is None:
                        logger.debug(f"  [{file_name}] Duty {duty}% {mode} {load}: 파일 없음")
                        result['velocities'][f'{mode}_{load}'].append(np.nan)
                        
                        # 통계 기록
                        self.file_stats.append({
                            'group': group_name,
                            'duty': duty,
                            'mode': mode,
                            'load': load,
                            'file': file_name,
                            'status': 'No File',
                            'samples': 0,
                            'velocity': np.nan
                        })
                        self.abnormal_count += 1
                        continue
                    
                    try:
                        data = data_info['data']
                        angle_col = data_info['angle_column']
                        
                        # 데이터 길이 체크
                        if len(data) < 10:
                            logger.warning(f"  [{file_name}] Duty {duty}% {mode} {load}: 데이터가 너무 짧음 ({len(data)} 샘플)")
                            result['velocities'][f'{mode}_{load}'].append(np.nan)
                            
                            # 통계 기록
                            self.file_stats.append({
                                'group': group_name,
                                'duty': duty,
                                'mode': mode,
                                'load': load,
                                'file': file_name,
                                'status': 'Too Short',
                                'samples': len(data),
                                'velocity': np.nan
                            })
                            self.abnormal_count += 1
                            
                            # 비정상 데이터는 무조건 플롯/CSV 저장
                            if len(data) > 0:
                                try:
                                    angle_data = data[angle_col]
                                    velocity = np.full(len(data), np.nan)  # 속도 계산 불가
                                    
                                    # 플롯 저장
                                    self._plot_sample_data(
                                        group_name, duty, mode, load, file_name,
                                        data['time(s)'].values, angle_data.values, velocity,
                                        is_abnormal=True
                                    )
                                    
                                    # CSV 저장
                                    self._save_parsed_csv(
                                        group_name, duty, mode, load, file_name,
                                        data['time(s)'].tolist(), angle_data.tolist(), velocity.tolist(),
                                        np.nan, is_abnormal=True
                                    )
                                except Exception as e:
                                    logger.error(f"  비정상 데이터 저장 실패: {str(e)}")
                            
                            continue
                        
                        # 간단한 노이즈 제거
                        if len(data) >= 20:
                            window = min(5, len(data) // 4)
                            angle_data = data[angle_col].rolling(window=window, center=True).mean().fillna(data[angle_col])
                            filter_status = f'Filtered (window={window})'
                        else:
                            angle_data = data[angle_col]
                            filter_status = 'No Filter (short data)'
                        
                        # 속도 계산
                        velocity = self.velocity_estimator.estimate_velocity(
                            angle_data,
                            data['time(s)'],
                            apply_filter=False
                        )
                        
                        # 정상 상태 속도 추출
                        steady_start_idx = int(len(velocity) * 0.7)
                        if steady_start_idx >= len(velocity) - 1:
                            steady_start_idx = max(0, len(velocity) - 5)
                        
                        steady_velocities = velocity[steady_start_idx:]
                        
                        # 이상치 제거
                        if len(steady_velocities) > 3:
                            median_v = np.median(steady_velocities)
                            std_v = np.std(steady_velocities)
                            if std_v > 0:
                                mask = np.abs(steady_velocities - median_v) <= 3 * std_v
                                steady_velocities_clean = steady_velocities[mask]
                                if len(steady_velocities_clean) > 0:
                                    steady_velocities = steady_velocities_clean
                        
                        if len(steady_velocities) > 0:
                            steady_velocity = np.mean(steady_velocities)
                        else:
                            steady_velocity = np.mean(velocity[steady_start_idx:])
                        
                        steady_velocity = abs(steady_velocity)
                        
                        # 이상한 값 체크
                        if steady_velocity < 0.1 or steady_velocity > 200:
                            logger.warning(f"  [{file_name}] Duty {duty}% {mode} {load}: 비정상 속도 {steady_velocity:.2f} deg/s")
                            result['velocities'][f'{mode}_{load}'].append(np.nan)
                            
                            # 통계 기록
                            self.file_stats.append({
                                'group': group_name,
                                'duty': duty,
                                'mode': mode,
                                'load': load,
                                'file': file_name,
                                'status': 'Abnormal Velocity',
                                'samples': len(data),
                                'velocity': steady_velocity,
                                'filter': filter_status
                            })
                            self.abnormal_count += 1
                            
                            # 비정상 데이터는 무조건 플롯/CSV 저장
                            self._plot_sample_data(
                                group_name, duty, mode, load, file_name,
                                data['time(s)'].values, angle_data.values, velocity,
                                is_abnormal=True
                            )
                            self._save_parsed_csv(
                                group_name, duty, mode, load, file_name,
                                data['time(s)'].tolist(), angle_data.tolist(), velocity.tolist(),
                                steady_velocity, is_abnormal=True
                            )
                            
                            continue
                        
                        result['velocities'][f'{mode}_{load}'].append(steady_velocity)
                        velocities_for_duty.append(steady_velocity)
                        
                        logger.info(f"  [{file_name}] Duty {duty}% {mode} {load}: {steady_velocity:.2f} deg/s ({len(data)} 샘플, {filter_status})")
                        
                        # 통계 기록
                        self.file_stats.append({
                            'group': group_name,
                            'duty': duty,
                            'mode': mode,
                            'load': load,
                            'file': file_name,
                            'status': 'OK',
                            'samples': len(data),
                            'velocity': steady_velocity,
                            'filter': filter_status
                        })
                        self.normal_count += 1
                        
                        # CSV용 데이터 저장
                        result['parsed_data_samples'].append({
                            'group': group_name,
                            'duty': duty,
                            'mode': mode,
                            'load': load,
                            'file': file_name,
                            'time': data['time(s)'].tolist(),
                            'angle': angle_data.tolist(),
                            'velocity': velocity.tolist(),
                            'steady_velocity': steady_velocity
                        })
                        
                        # 10개당 1개씩 그래프 생성 (정상 데이터만)
                        self.sample_count += 1
                        if self.sample_count % 10 == 0:
                            self._plot_sample_data(
                                group_name, duty, mode, load, file_name,
                                data['time(s)'].values, angle_data.values, velocity,
                                is_abnormal=False
                            )
                            self._save_parsed_csv(
                                group_name, duty, mode, load, file_name,
                                data['time(s)'].tolist(), angle_data.tolist(), velocity.tolist(),
                                steady_velocity, is_abnormal=False
                            )
                        
                    except Exception as e:
                        logger.error(f"  [{file_name}] Duty {duty}% {mode} {load} 오류: {str(e)}")
                        result['velocities'][f'{mode}_{load}'].append(np.nan)
                        
                        # 통계 기록
                        self.file_stats.append({
                            'group': group_name,
                            'duty': duty,
                            'mode': mode,
                            'load': load,
                            'file': file_name,
                            'status': f'Error: {str(e)}',
                            'samples': len(data) if 'data' in locals() else 0,
                            'velocity': np.nan
                        })
                        self.abnormal_count += 1
            
            result['duty_values'].append(duty)
            
            # 평균 속도 계산
            valid_velocities = [v for v in velocities_for_duty if not np.isnan(v) and v > 0]
            if valid_velocities:
                avg_velocity = np.mean(valid_velocities)
                result['all_velocities'].append(avg_velocity)
                logger.info(f"  Duty {duty}%: 평균 {avg_velocity:.2f} deg/s ({len(valid_velocities)}개 데이터)")
            else:
                result['all_velocities'].append(np.nan)
                logger.warning(f"  Duty {duty}%: 유효한 데이터 없음")
        
        return result
    
    def _plot_sample_data(self, group_name: str, duty: int, mode: str, load: str, 
                          file_name: str, time: np.ndarray, angle: np.ndarray, velocity: np.ndarray,
                          is_abnormal: bool = False):
        """샘플 데이터 시각화"""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # 제목 색상 (비정상 데이터는 빨간색)
            title_color = 'red' if is_abnormal else 'black'
            title_prefix = '[ABNORMAL] ' if is_abnormal else ''
            
            # 각도 그래프
            ax1.plot(time, angle, 'b-', linewidth=1.5, label='Angle')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Angle (deg)')
            ax1.set_title(f'{title_prefix}{group_name} - Duty {duty}% {mode} {load}\n{file_name}', 
                         color=title_color, fontweight='bold' if is_abnormal else 'normal')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # 속도 그래프
            if not np.all(np.isnan(velocity)):
                ax2.plot(time[:len(velocity)], velocity, 'r-', linewidth=1.5, label='Velocity')
                
                # 정상 상태 구간 표시 (정상 데이터만)
                if not is_abnormal and len(velocity) > 0:
                    steady_start_idx = int(len(velocity) * 0.7)
                    if steady_start_idx < len(time):
                        ax2.axvline(time[steady_start_idx], color='g', linestyle='--', label='Steady State Start')
            else:
                ax2.text(0.5, 0.5, 'Velocity data unavailable', 
                        ha='center', va='center', transform=ax2.transAxes,
                        fontsize=14, color='red')
            
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Velocity (deg/s)')
            ax2.set_title('Angular Velocity', color=title_color)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            
            # 파일명 생성 (비정상 데이터는 ABNORMAL_ 접두사)
            prefix = 'ABNORMAL_' if is_abnormal else ''
            counter = self.abnormal_count if is_abnormal else self.sample_count
            safe_filename = f"{prefix}{group_name}_D{duty}_{mode}_{load}_{counter}.png"
            output_path = self.plots_dir / safe_filename
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            logger.debug(f"  그래프 저장: {safe_filename}")
            
        except Exception as e:
            logger.error(f"  그래프 생성 실패: {str(e)}")
            plt.close()
    
    def _save_parsed_csv(self, group_name: str, duty: int, mode: str, load: str,
                        file_name: str, time: List, angle: List, velocity: List,
                        steady_velocity: float, is_abnormal: bool = False):
        """파싱된 데이터를 CSV로 저장"""
        try:
            df = pd.DataFrame({
                'time': time,
                'angle': angle,
                'velocity': velocity
            })
            
            # 파일명 생성 (비정상 데이터는 ABNORMAL_ 접두사)
            prefix = 'ABNORMAL_' if is_abnormal else ''
            counter = self.abnormal_count if is_abnormal else self.sample_count
            filename = f"{prefix}{group_name}_D{duty}_{mode}_{load}_{counter}.csv"
            output_path = self.csv_dir / filename
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            
            # 메타데이터 파일
            meta_path = output_path.with_suffix('.meta')
            with open(meta_path, 'w', encoding='utf-8') as f:
                f.write(f"Group: {group_name}\n")
                f.write(f"Duty: {duty}%\n")
                f.write(f"Mode: {mode}\n")
                f.write(f"Load: {load}\n")
                f.write(f"Source File: {file_name}\n")
                if not np.isnan(steady_velocity):
                    f.write(f"Steady State Velocity: {steady_velocity:.4f} deg/s\n")
                else:
                    f.write(f"Steady State Velocity: N/A (abnormal data)\n")
                if is_abnormal:
                    f.write(f"Status: ABNORMAL\n")
            
            logger.debug(f"  CSV 저장: {filename}")
            
        except Exception as e:
            logger.error(f"  CSV 저장 실패: {str(e)}")
    
    def estimate_ff_gain(self, duty_values: List[float], velocities: List[float]) -> Dict:
        """FF 게인 추정"""
        logger.info("\nFF 게인 추정")
        
        valid_indices = [i for i, v in enumerate(velocities) if not np.isnan(v) and v > 0]
        
        if len(valid_indices) < 3:
            logger.error(f"유효한 데이터가 부족합니다 ({len(valid_indices)}개, 최소 3개 필요)")
            return {'kv': 0, 'k_offset': 0, 'r_squared': 0, 'method': 'failed'}
        
        duty_array = np.array([duty_values[i] for i in valid_indices])
        velocity_array = np.array([velocities[i] for i in valid_indices])
        
        def linear_func(velocity, kv, offset):
            return kv * velocity + offset
        
        try:
            params, _ = curve_fit(linear_func, velocity_array, duty_array, p0=[1.0, 0.0])
            kv, k_offset = params
            
            duty_pred = linear_func(velocity_array, kv, k_offset)
            ss_res = np.sum((duty_array - duty_pred) ** 2)
            ss_tot = np.sum((duty_array - np.mean(duty_array)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            logger.info(f"  FF 게인 (Kv): {kv:.4f} %/(deg/s)")
            logger.info(f"  오프셋: {k_offset:.4f} %")
            logger.info(f"  R²: {r_squared:.4f}")
            
            logger.info("  검증:")
            for d, v in zip(duty_array, velocity_array):
                pred = linear_func(v, kv, k_offset)
                error = abs(d - pred)
                logger.info(f"    Duty {d:.0f}%: 예측 {pred:.1f}%, 오차 {error:.1f}%")
            
            return {
                'kv': float(kv),
                'k_offset': float(k_offset),
                'r_squared': float(r_squared),
                'method': 'linear_fit',
                'data_points': len(valid_indices)
            }
            
        except Exception as e:
            logger.error(f"FF 게인 피팅 실패: {str(e)}")
            kv_simple = np.mean(duty_array / velocity_array)
            logger.info(f"  Fallback - 단순 평균 FF 게인: {kv_simple:.4f} %/(deg/s)")
            
            return {
                'kv': float(kv_simple),
                'k_offset': 0.0,
                'r_squared': 0.0,
                'method': 'simple_average'
            }
    
    def estimate_pid_gains_from_model(self, parsed_group: Dict, representative_duty: int = 70) -> Dict:
        """PID 게인 추정"""
        logger.info(f"\nPID 게인 추정 (대표 Duty: {representative_duty}%)")
        
        available_duties = sorted(parsed_group['data_by_duty'].keys())
        if representative_duty not in available_duties:
            representative_duty = available_duties[len(available_duties) // 2]
            logger.info(f"  대표 Duty 변경: {representative_duty}%")
        
        duty_data = parsed_group['data_by_duty'][representative_duty]
        
        data_info = None
        for mode in ['Single', 'Couple']:
            for load in ['High', 'Low']:
                candidate = duty_data[mode][load]
                if candidate is not None:
                    if data_info is None or len(candidate['data']) > len(data_info['data']):
                        data_info = candidate
        
        if data_info is None:
            logger.error("  대표 데이터를 찾을 수 없습니다.")
            return {'model_params': {}, 'pid_gains': {}}
        
        try:
            data = data_info['data']
            angle_col = data_info['angle_column']
            
            logger.info(f"  데이터 길이: {len(data)} 샘플")
            
            if len(data) >= 20:
                window = min(5, len(data) // 4)
                angle_filtered = data[angle_col].rolling(window=window, center=True).mean().fillna(data[angle_col])
            else:
                angle_filtered = data[angle_col]
            
            model = self.model_fitter.fit_first_order(
                data['time(s)'].values,
                angle_filtered.values,
                initial_guess={'K': 10.0, 'tau': 0.5, 'delay': 0.05}
            )
            
            fitted_params = self.model_fitter.get_fitted_params()
            goodness = self.model_fitter.get_goodness_of_fit()
            
            logger.info(f"  모델: K={fitted_params['K']:.3f}, τ={fitted_params['tau']:.3f}s, L={fitted_params['delay']:.3f}s")
            logger.info(f"  적합도: R²={goodness['r_squared']:.4f}")
            
            K_normalized = fitted_params['K'] / representative_duty
            tau = fitted_params['tau']
            delay = max(fitted_params['delay'], 0.01)
            
            from ..tuning.pid_tuner_zn import ZieglerNichols
            from ..tuning.pid_tuner_cc import CohenCoon
            from ..tuning.pid_tuner_imc import IMCTuner
            
            zn = ZieglerNichols()
            zn_gains = zn.tune(K_normalized, tau, delay)
            
            cc = CohenCoon()
            cc_gains = cc.tune(K_normalized, tau, delay)
            
            imc = IMCTuner(lambda_factor=2.0)
            imc_gains = imc.tune(K_normalized, tau)
            
            return {
                'model_params': fitted_params,
                'goodness_of_fit': goodness,
                'pid_gains': {
                    'ziegler_nichols': zn_gains,
                    'cohen_coon': cc_gains,
                    'imc': imc_gains
                }
            }
            
        except Exception as e:
            logger.error(f"  모델 피팅 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'model_params': {}, 'pid_gains': {}}
    
    def analyze_group(self, parsed_group: Dict) -> Dict:
        """그룹 전체 분석"""
        group_name = parsed_group['group_name']
        logger.info(f"\n{'='*80}")
        logger.info(f"[{group_name}] 통합 분석 시작")
        logger.info(f"{'='*80}")
        
        velocity_result = self.extract_steady_state_velocities(parsed_group)
        ff_result = self.estimate_ff_gain(
            velocity_result['duty_values'],
            velocity_result['all_velocities']
        )
        pid_result = self.estimate_pid_gains_from_model(parsed_group)
        
        safety_factor = 0.8
        if pid_result.get('pid_gains') and pid_result['pid_gains'].get('imc'):
            imc_gains = pid_result['pid_gains']['imc']
            conservative_gains = {
                'Kp': imc_gains.get('Kp', 0) * safety_factor,
                'Ki': imc_gains.get('Ki', 0) * safety_factor,
                'Kd': imc_gains.get('Kd', 0) * safety_factor
            }
        else:
            conservative_gains = {'Kp': 0, 'Ki': 0, 'Kd': 0}
        
        result = {
            'group_name': group_name,
            'ff_gain': ff_result,
            'pid_gains': pid_result.get('pid_gains', {}),
            'conservative_gains': conservative_gains,
            'model_params': pid_result.get('model_params', {}),
            'goodness_of_fit': pid_result.get('goodness_of_fit', {}),
            'velocity_data': velocity_result,
            'parsed_samples': velocity_result['parsed_data_samples']
        }
        
        logger.info(f"\n{'='*80}")
        logger.info(f"[{group_name}] 최종 결과")
        logger.info(f"{'='*80}")
        logger.info(f"FF 게인 (Kv): {ff_result['kv']:.4f} %/(deg/s)")
        logger.info(f"PID 게인 (보수적):")
        logger.info(f"  Kp: {conservative_gains['Kp']:.4f}")
        logger.info(f"  Ki: {conservative_gains['Ki']:.4f}")
        logger.info(f"  Kd: {conservative_gains['Kd']:.4f}")
        
        return result
    
    def save_file_statistics(self):
        """파일 처리 통계를 CSV로 저장"""
        if not self.file_stats:
            return
        
        df = pd.DataFrame(self.file_stats)
        output_path = self.debug_dir / 'file_statistics.csv'
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"\n파일 통계 저장: {output_path}")
        
        # 요약 통계
        logger.info("\n=== 파일 처리 통계 요약 ===")
        logger.info(f"  정상 데이터: {self.normal_count}개")
        logger.info(f"  비정상 데이터: {self.abnormal_count}개")
        logger.info(f"  전체: {len(self.file_stats)}개")
        logger.info("\n상태별 분류:")
        status_counts = df['status'].value_counts()
        for status, count in status_counts.items():
            logger.info(f"  {status}: {count}개")
    
    def save_parsed_samples(self, all_samples: List[Dict]):
        """파싱된 샘플 데이터를 CSV로 저장 (10개당 1개)"""
        sample_interval = 10
        for idx, sample in enumerate(all_samples):
            if (idx + 1) % sample_interval == 0:
                try:
                    df = pd.DataFrame({
                        'time': sample['time'],
                        'angle': sample['angle'],
                        'velocity': sample['velocity']
                    })
                    
                    filename = f"{sample['group']}_D{sample['duty']}_{sample['mode']}_{sample['load']}.csv"
                    output_path = self.csv_dir / filename
                    df.to_csv(output_path, index=False, encoding='utf-8-sig')
                    
                    # 메타데이터 추가
                    with open(output_path.with_suffix('.meta'), 'w', encoding='utf-8') as f:
                        f.write(f"Group: {sample['group']}\n")
                        f.write(f"Duty: {sample['duty']}%\n")
                        f.write(f"Mode: {sample['mode']}\n")
                        f.write(f"Load: {sample['load']}\n")
                        f.write(f"Source File: {sample['file']}\n")
                        f.write(f"Steady State Velocity: {sample['steady_velocity']:.4f} deg/s\n")
                    
                    logger.debug(f"  샘플 CSV 저장: {filename}")
                except Exception as e:
                    logger.error(f"  샘플 CSV 저장 실패: {str(e)}")

