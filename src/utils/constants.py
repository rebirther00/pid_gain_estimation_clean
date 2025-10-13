"""
상수 정의 모듈
"""

# 각도 범위 제한 (degree)
ANGLE_LIMITS = {
    'Bkt_ang': {'min': -125.0, 'max': 40.0},
    'Arm_ang': {'min': -150.0, 'max': -40.0},
    'Boom_ang': {'min': 0.0, 'max': 55.0}
}

# 동작 매핑
ACTION_MAP = {
    'boom_up': 'Boom_up_duty',
    'boom_dn': 'Boom_dn_duty',
    'arm_in': 'Arm_in_duty',
    'arm_out': 'Arm_out_duty',
    'bkt_in': 'Bkt_in_duty',
    'bkt_out': 'Bkt_out_duty'
}

# 축 매핑
AXIS_MAP = {
    'A': ['Arm_ang', 'Bkt_ang'],  # Arm 폴더에는 Arm과 Bucket 각도 포함
    'B': ['Boom_ang']
}

# 방향 매핑
DIRECTION_MAP = {
    'in': 'In',
    'out': 'Out',
    'up': 'Up',
    'dn': 'Down'
}

# Duty 값 리스트
DUTY_VALUES = [40, 50, 60, 70, 80, 90, 100]

# 샘플링 레이트
SAMPLING_RATE = 0.01  # 10ms (100Hz)
SAMPLING_FREQUENCY = 100  # Hz

# 유효 데이터 마진
VALID_DATA_MARGIN = 3.0  # 최종 각도 - 3도

