@echo off

echo ===============================
echo Running 10v11 Experiment
echo ===============================

echo Seed 1
python light_train.py --env_name smacv2 --map_name terran_10_vs_11 --cuda --max_steps 300000 --seed 1

echo Seed 2
python light_train.py --env_name smacv2 --map_name terran_10_vs_11 --cuda --max_steps 300000 --seed 2

echo Seed 3
python light_train.py --env_name smacv2 --map_name terran_10_vs_11 --cuda --max_steps 300000 --seed 3

echo ===============================
echo All experiments finished
echo ===============================

pause