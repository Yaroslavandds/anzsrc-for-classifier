@echo off
rem Retraining on the cleaned labels, one step after another: about 5-6 hours on an RTX 3050 Ti Laptop.
rem Run it from Anaconda Prompt in the training folder:   run_clean_training.bat
rem If it stops (error, Ctrl+C, power), run it again: finished steps are skipped, interrupted training continues.
cd /d "%~dp0"
python 06_clean_labels.py || goto :error
if not exist models\baseline_clean\test_probs.npy (python 02_baseline.py --train train_clean || goto :error)
if not exist models\roberta_clean\test_probs.npy (python 03_train_transformer.py --model FacebookAI/roberta-base --train train_clean --smoke || goto :error)
if not exist models\specter2_clean\test_probs.npy (python 03_train_transformer.py --train train_clean || goto :error)
if not exist models\roberta_clean\test_probs.npy (python 03_train_transformer.py --model FacebookAI/roberta-base --train train_clean || goto :error)
python 04_ensemble.py --models baseline_clean specter2_clean roberta_clean --name ensemble_clean || goto :error
python 05_evaluate.py --ensemble ensemble_clean --compare baseline specter2 ensemble || goto :error
echo.
echo Done. Report: reports\05_test_report_clean.xlsx
goto :eof
:error
echo.
echo Stopped (code %errorlevel%). Run run_clean_training.bat again to continue.
exit /b 1
