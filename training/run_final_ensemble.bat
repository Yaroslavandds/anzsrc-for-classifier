@echo off
rem The final combination: TF-IDF + SPECTER2 + ModernBERT-large, all on the cleaned labels. On val it is as good as
rem the five-model one and needs fewer models. A minute or two, then the test report with the earlier ones next to it.
rem Run it from Anaconda Prompt in the training folder:   run_final_ensemble.bat
cd /d "%~dp0"
python 04_ensemble.py --models baseline_clean specter2_clean modernbert_large_clean --name ensemble_final || goto :error
python 05_evaluate.py --ensemble ensemble_final --compare ensemble_large ensemble_clean ensemble || goto :error
echo.
echo Done. Report: reports\05_test_report_final.xlsx. predict.py now uses this combination by default.
goto :eof
:error
echo.
echo Stopped (code %errorlevel%).
exit /b 1
