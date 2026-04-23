@echo off

echo ===============================
echo MEDIVAC HEAVY 5v5 - 10M TIMESTEPS
echo ===============================

echo Seed 1
python light_train.py --env_name smacv2 --map_name terran_5_vs_5 --cuda --max_steps 10000000 --seed 1

echo Seed 2
python light_train.py --env_name smacv2 --map_name terran_5_vs_5 --cuda --max_steps 10000000 --seed 2

echo Seed 3
python light_train.py --env_name smacv2 --map_name terran_5_vs_5 --cuda --max_steps 10000000 --seed 3

pause