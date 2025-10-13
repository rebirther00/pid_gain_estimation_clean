@echo off
chcp 65001 >nul
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
echo [3/5] 개별 게인 통합 중...
python scripts/combine_individual_gains.py
if %errorlevel% neq 0 (
    echo 오류: 통합 실패!
    pause
    exit /b 1
)

echo.
echo [4/5] 성능 분석 중...
python scripts/analyze_control_performance.py
if %errorlevel% neq 0 (
    echo 오류: 성능 분석 실패!
    pause
    exit /b 1
)

echo.
echo [5/7] 시각화 생성 중...
python scripts/visualize_performance_analysis.py
if %errorlevel% neq 0 (
    echo 오류: 시각화 실패!
    pause
    exit /b 1
)

echo.
echo [6/7] τ=2s 기준 PID 게인 재계산 중...
python scripts/recalculate_pid_with_tau2.py
if %errorlevel% neq 0 (
    echo 오류: τ=2s 재계산 실패!
    pause
    exit /b 1
)

echo.
echo [7/7] 최종 요약 문서 업데이트 중...
python scripts/update_final_summary.py
if %errorlevel% neq 0 (
    echo 오류: 최종 요약 업데이트 실패!
    pause
    exit /b 1
)

echo.
echo ================================================
echo 완료! 결과 확인:
echo.
echo [최종 게인]
echo   - output/post_process_v3/final_gains.json
echo   - output/post_process_v3/all_individual_gains.xlsx
echo   - FINAL_GAINS_SUMMARY.md
echo.
echo [성능 분석]
echo   - output/error_analysis/PERFORMANCE_ANALYSIS_REPORT.md
echo   - output/error_analysis/performance_analysis_overview.png
echo   - ERROR_ANALYSIS_SUMMARY.md
echo ================================================
pause

