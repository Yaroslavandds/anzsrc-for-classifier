@echo off
rem A large model, ModernBERT-large (395M parameters), trained on the cleaned labels with LoRA so that it fits
rem into 4 GB of GPU memory. About 7-8 hours on an RTX 3050 Ti Laptop: a 5-minute check first (it also downloads
rem the model, 1.6 GB), then the training, then the ensemble and the test report.
rem Run it from Anaconda Prompt in the training folder:   run_large_training.bat
rem If it stops (error, Ctrl+C, power), run it again: finished steps are skipped, interrupted training continues.
cd /d "%~dp0"
set ARGS=--model answerdotai/ModernBERT-large --train train_clean --lora 32 --batch 16
set M=models\modernbert_large_clean
if not exist %M%\last.pt if not exist %M%\test_probs.npy (python 03_train_transformer.py %ARGS% --smoke || goto :error)
if not exist %M%\test_probs.npy (python 03_train_transformer.py %ARGS% || goto :error)
python 04_ensemble.py --models baseline_clean specter2_clean roberta_clean modernbert_large_clean --name ensemble_large || goto :error
python 05_evaluate.py --ensemble ensemble_large --compare ensemble_clean ensemble || goto :error
echo.
echo Done. Report: reports\05_test_report_large.xlsx
goto :eof
:error
echo.
echo Stopped (code %errorlevel%). Run run_large_training.bat again to continue.
exit /b 1
