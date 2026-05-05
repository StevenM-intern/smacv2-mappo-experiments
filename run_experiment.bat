@echo off
echo ===============================
echo SMACv2 Experiment Runner
echo ===============================

set MODE=%1

IF "%MODE%"=="" (
    echo Usage:
    echo run_experiment.bat baseline
    echo run_experiment.bat medivac
    echo run_experiment.bat zerg
    echo run_experiment.bat protoss
    exit /b
)

IF "%MODE%"=="baseline" (
    python light_train.py --env_name smacv2 --map_name terran_5_vs_5
)

IF "%MODE%"=="medivac" (
    python light_train.py --env_name smacv2 --map_name terran_5_vs_5 --use_medivac_heavy
)

IF "%MODE%"=="zerg" (
    python light_train.py --env_name smacv2 --map_name zerg_5_vs_5
)

IF "%MODE%"=="protoss" (
    python light_train.py --env_name smacv2 --map_name protoss_5_vs_5
)
