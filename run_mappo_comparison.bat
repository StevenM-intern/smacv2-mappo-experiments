@echo off
echo ===============================================
echo MAPPO SCALABILITY + TEAM COMPOSITION STUDY
echo ===============================================

REM Activate conda environment
call conda activate marl-ppo-suite

REM Set StarCraft II path
set SC2PATH=C:\Program Files (x86)\StarCraft II

echo.
echo -----------------------------------------------
echo Running MAPPO on 3m (3 agents - homogeneous)
echo -----------------------------------------------
python light_train.py --algo mappo --map_name 3m --seed 1 --max_steps 300000

echo.
echo -----------------------------------------------
echo Running MAPPO on 8m (8 agents - homogeneous)
echo -----------------------------------------------
python light_train.py --algo mappo --map_name 8m --seed 1 --max_steps 300000

echo.
echo -----------------------------------------------
echo Running MAPPO on 2s3z (5 agents - heterogeneous)
echo -----------------------------------------------
python light_train.py --algo mappo --map_name 2s3z --seed 1 --max_steps 300000

echo.
echo -----------------------------------------------
echo Running MAPPO on MMM (10 agents - heterogeneous)
echo -----------------------------------------------
python light_train.py --algo mappo --map_name MMM --seed 1 --max_steps 300000

echo.
echo ===============================================
echo ALL EXPERIMENTS COMPLETE
echo ===============================================

pause
