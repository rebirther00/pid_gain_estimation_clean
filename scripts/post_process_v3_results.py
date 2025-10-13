"""
V3 결과 후처리 스크립트

V3는 각 파일의 속도만 계산했으므로:
1. 각 파일을 다시 읽어서 모델 피팅 + PID 계산
2. 통계적으로 유의미한 PID만 선택
3. 중앙값으로 최종 PID 도출
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import logging

# 프로젝트 루트 추가
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.parser.csv_parser import CSVParser
from src.parser.filename_parser import FilenameParser
from src.parser.data_validator import DataValidator
from src.identification.model_fitting import ModelFitter
from src.tuning.pid_tuner_imc import IMCTuner
from src.utils.constants import ANGLE_LIMITS

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def calculate_pid_for_file(file_path: str, duty: int) -> dict:
    """단일 파일에 대해 모델 피팅 + PID 계산"""
    try:
        # 파싱
        csv_parser = CSVParser()
        validator = DataValidator(angle_limits=ANGLE_LIMITS)
        filename_parser = FilenameParser()
        
        data, _ = csv_parser.parse(file_path)
        metadata = filename_parser.parse(file_path)
        angle_col = filename_parser.get_axis_angle_column(metadata)
        
        # 검증
        data_valid = validator.filter_valid_data(data, angle_col, margin=3.0)
        data_valid = validator.filter_angle_limits(data_valid, angle_col)
        
        if len(data_valid) < 20:
            return None
        
        # 필터링
        if len(data_valid) >= 20:
            window = min(5, len(data_valid) // 4)
            angle_filtered = data_valid[angle_col].rolling(window=window, center=True).mean().fillna(data_valid[angle_col])
        else:
            angle_filtered = data_valid[angle_col]
        
        # 초기값 추정 (부호 고려!)
        final_angle = angle_filtered.iloc[-1]
        initial_angle = angle_filtered.iloc[0]
        angle_change = final_angle - initial_angle
        
        # Duty 부호 결정:
        # - 각도가 증가하면 (양수 변화): duty는 양수
        # - 각도가 감소하면 (음수 변화): duty는 음수로 처리
        if angle_change < 0:
            # In/Down 방향: 음수 duty
            effective_duty = -abs(duty)
        else:
            # Out/Up 방향: 양수 duty
            effective_duty = abs(duty)
        
        # K 계산: angle_change / effective_duty
        # 이제 K는 항상 양수가 나옴
        K_init = abs(angle_change / effective_duty) if effective_duty != 0 else 10.0
        K_init = max(K_init, 0.1)
        
        # 모델 피팅
        model_fitter = ModelFitter()
        model_fitter.fit_first_order(
            data_valid['time(s)'].values,
            angle_filtered.values,
            initial_guess={'K': K_init, 'tau': 0.5, 'delay': 0.05}
        )
        
        params = model_fitter.get_fitted_params()
        goodness = model_fitter.get_goodness_of_fit()
        
        # PID 계산 (K의 절대값 사용)
        K_abs = abs(params['K'])
        K_norm = K_abs / abs(effective_duty) if effective_duty != 0 else K_abs
        tau = params['tau']
        delay = max(params['delay'], 0.01)
        
        tuner = IMCTuner(lambda_factor=2.0)
        pid_gains = tuner.tune(K_norm, tau, delay)
        
        # 안전계수
        safety_factor = 0.8
        
        # 품질 플래그: K의 절대값이 너무 작으면 low_quality
        is_low_quality = K_abs < 0.01
        
        return {
            'K': params['K'],
            'K_abs': K_abs,
            'effective_duty': effective_duty,
            'angle_change': float(angle_change),
            'tau': params['tau'],
            'delay': params['delay'],
            'r_squared': goodness['r_squared'],
            'rmse': goodness['rmse'],
            'Kp': pid_gains['Kp'] * safety_factor,
            'Ki': pid_gains['Ki'] * safety_factor,
            'Kd': pid_gains['Kd'] * safety_factor,
            'low_quality': is_low_quality
        }
        
    except Exception as e:
        logger.debug(f"  {Path(file_path).name}: {str(e)}")
        return None


def main():
    logger.info("="*80)
    logger.info("V3 결과 후처리: 통계 기반 PID 게인 추정")
    logger.info("="*80)
    
    # V3 파일 통계 로드
    stats_path = project_root / 'output' / 'integrated_v3' / 'debug' / 'file_statistics.csv'
    
    if not stats_path.exists():
        logger.error(f"파일이 없습니다: {stats_path}")
        logger.info("먼저 V3를 실행하세요: python scripts/run_integrated_analysis_v3.py")
        return
    
    df = pd.read_csv(stats_path)
    logger.info(f"\nV3 파일 통계 로드: {len(df)} 파일")
    
    # 그룹별로 처리
    groups = df['group'].unique()
    
    all_results = {}
    
    for group_name in groups:
        logger.info(f"\n{'='*80}")
        logger.info(f"{group_name}")
        logger.info(f"{'='*80}")
        
        group_df = df[df['group'] == group_name]
        logger.info(f"파일 수: {len(group_df)}")
        
        # 데이터 디렉토리
        data_dir = project_root / 'data'
        
        # 각 파일에 대해 PID 계산
        pid_results = []
        
        for idx, row in group_df.iterrows():
            file_name = row['file']
            duty = row['duty']
            
            # 파일 경로 찾기
            file_path = None
            for folder in data_dir.iterdir():
                if folder.is_dir():
                    candidate = folder / file_name
                    if candidate.exists():
                        file_path = candidate
                        break
            
            if file_path is None:
                logger.debug(f"  {file_name}: 파일을 찾을 수 없음")
                continue
            
            result = calculate_pid_for_file(str(file_path), duty)
            
            if result is not None:
                result['file'] = file_name
                result['duty'] = duty
                result['mode'] = row['mode']
                result['load'] = row['load']
                pid_results.append(result)
                logger.debug(f"  {file_name}: OK (Kp={result['Kp']:.6f}, R²={result['r_squared']:.4f})")
        
        logger.info(f"\n성공적으로 계산된 PID: {len(pid_results)}개")
        
        if len(pid_results) == 0:
            logger.warning("유효한 PID 게인 없음!")
            all_results[group_name] = {'error': '유효한 PID 게인 없음'}
            continue
        
        # DataFrame으로 변환
        gains_df = pd.DataFrame(pid_results)
        
        # low_quality 샘플 분리
        n_low_quality = gains_df['low_quality'].sum() if 'low_quality' in gains_df.columns else 0
        high_quality_df = gains_df[~gains_df.get('low_quality', False)].copy() if 'low_quality' in gains_df.columns else gains_df.copy()
        
        logger.info(f"\n품질 분류:")
        logger.info(f"  전체 샘플: {len(gains_df)}개")
        logger.info(f"  고품질 (K >= 0.01): {len(high_quality_df)}개")
        logger.info(f"  저품질 (K < 0.01): {n_low_quality}개")
        
        # 고품질 샘플이 없으면 경고만 하고 전체 사용
        if len(high_quality_df) == 0:
            logger.warning("고품질 샘플 없음! 전체 샘플 사용 (참고용)")
            high_quality_df = gains_df.copy()
        
        # 이상치 제거 (IQR) - 고품질 샘플에서만
        logger.info(f"\n이상치 제거 (IQR, 고품질 샘플 대상):")
        logger.info(f"  제거 전: {len(high_quality_df)}개")
        
        mask = pd.Series([True] * len(high_quality_df))
        for col in ['Kp', 'Ki', 'Kd']:
            Q1 = high_quality_df[col].quantile(0.25)
            Q3 = high_quality_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            col_mask = (high_quality_df[col] >= lower) & (high_quality_df[col] <= upper)
            n_outliers = (~col_mask).sum()
            if n_outliers > 0:
                logger.info(f"    {col}: {n_outliers}개 이상치")
            mask = mask & col_mask
        
        gains_filtered = high_quality_df[mask].copy()
        logger.info(f"  제거 후: {len(gains_filtered)}개")
        
        if len(gains_filtered) == 0:
            logger.warning("이상치 제거 후 데이터 없음! 고품질 원본 사용")
            gains_filtered = high_quality_df
        
        # 최종 게인 (중앙값)
        final_kp = gains_filtered['Kp'].median()
        final_ki = gains_filtered['Ki'].median()
        final_kd = gains_filtered['Kd'].median()
        
        logger.info(f"\n최종 게인 (중앙값):")
        logger.info(f"  Kp: {final_kp:.6f}")
        logger.info(f"  Ki: {final_ki:.6f}")
        logger.info(f"  Kd: {final_kd:.6f}")
        
        # 통계
        logger.info(f"\n통계:")
        logger.info(f"  Kp: {gains_filtered['Kp'].mean():.6f} ± {gains_filtered['Kp'].std():.6f}")
        logger.info(f"      범위: [{gains_filtered['Kp'].min():.6f}, {gains_filtered['Kp'].max():.6f}]")
        logger.info(f"  Ki: {gains_filtered['Ki'].mean():.6f} ± {gains_filtered['Ki'].std():.6f}")
        logger.info(f"      범위: [{gains_filtered['Ki'].min():.6f}, {gains_filtered['Ki'].max():.6f}]")
        logger.info(f"  Kd: {gains_filtered['Kd'].mean():.6f} ± {gains_filtered['Kd'].std():.6f}")
        logger.info(f"      범위: [{gains_filtered['Kd'].min():.6f}, {gains_filtered['Kd'].max():.6f}]")
        logger.info(f"  R²: 평균 {gains_filtered['r_squared'].mean():.4f}, "
                   f"범위 [{gains_filtered['r_squared'].min():.4f}, {gains_filtered['r_squared'].max():.4f}]")
        
        # FF 게인 계산 (V3 속도 사용)
        valid_velocities = group_df[group_df['status'] == 'OK']
        if len(valid_velocities) >= 3:
            from scipy.optimize import curve_fit
            
            duties = valid_velocities['duty'].values
            velocities = valid_velocities['velocity'].values
            
            def linear_func(velocity, kv, offset):
                return kv * velocity + offset
            
            try:
                params, _ = curve_fit(linear_func, velocities, duties, p0=[1.0, 0.0])
                kv, k_offset = params
                
                duty_pred = linear_func(velocities, kv, k_offset)
                ss_res = np.sum((duties - duty_pred) ** 2)
                ss_tot = np.sum((duties - np.mean(duties)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                logger.info(f"\nFF 게인:")
                logger.info(f"  Kv: {kv:.4f} %/(deg/s)")
                logger.info(f"  Offset: {k_offset:.4f} %")
                logger.info(f"  R²: {r_squared:.4f}")
                
                ff_gain = {'Kv': kv, 'K_offset': k_offset, 'r_squared': r_squared}
            except:
                ff_gain = {'Kv': 0, 'K_offset': 0, 'r_squared': 0}
        else:
            ff_gain = {'Kv': 0, 'K_offset': 0, 'r_squared': 0}
        
        # 결과 저장
        all_results[group_name] = {
            'n_samples_total': len(pid_results),
            'n_samples_valid': len(gains_filtered),
            'n_outliers': len(pid_results) - len(gains_filtered),
            'final_gains': {
                'Kp': float(final_kp),
                'Ki': float(final_ki),
                'Kd': float(final_kd)
            },
            'ff_gain': ff_gain,
            'statistics': {
                'Kp': {
                    'mean': float(gains_filtered['Kp'].mean()),
                    'std': float(gains_filtered['Kp'].std()),
                    'min': float(gains_filtered['Kp'].min()),
                    'max': float(gains_filtered['Kp'].max())
                },
                'Ki': {
                    'mean': float(gains_filtered['Ki'].mean()),
                    'std': float(gains_filtered['Ki'].std()),
                    'min': float(gains_filtered['Ki'].min()),
                    'max': float(gains_filtered['Ki'].max())
                },
                'Kd': {
                    'mean': float(gains_filtered['Kd'].mean()),
                    'std': float(gains_filtered['Kd'].std()),
                    'min': float(gains_filtered['Kd'].min()),
                    'max': float(gains_filtered['Kd'].max())
                },
                'r_squared': {
                    'mean': float(gains_filtered['r_squared'].mean()),
                    'min': float(gains_filtered['r_squared'].min()),
                    'max': float(gains_filtered['r_squared'].max())
                }
            }
        }
        
        # 개별 게인 저장 (전체 샘플 저장, 저품질 포함)
        output_dir = project_root / 'output' / 'post_process_v3'
        output_dir.mkdir(exist_ok=True)
        gains_df.to_csv(output_dir / f'{group_name}_individual_gains.csv', index=False)
    
    # 최종 요약
    logger.info(f"\n{'='*80}")
    logger.info("최종 요약")
    logger.info(f"{'='*80}")
    
    for group_name, result in all_results.items():
        if 'error' in result:
            logger.info(f"\n{group_name}: {result['error']}")
        else:
            logger.info(f"\n{group_name}:")
            logger.info(f"  샘플: 전체 {result['n_samples_total']}, 유효 {result['n_samples_valid']}, 이상치 {result['n_outliers']}")
            logger.info(f"  최종 PID:")
            logger.info(f"    Kp: {result['final_gains']['Kp']:.6f}")
            logger.info(f"    Ki: {result['final_gains']['Ki']:.6f}")
            logger.info(f"    Kd: {result['final_gains']['Kd']:.6f}")
            logger.info(f"  FF:")
            logger.info(f"    Kv: {result['ff_gain']['Kv']:.4f}")
    
    # JSON으로 저장
    import json
    output_path = output_dir / 'final_gains.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n결과 저장: {output_path}")
    logger.info(f"개별 게인: {output_dir}/")
    logger.info(f"\n완료!")


if __name__ == '__main__':
    main()

