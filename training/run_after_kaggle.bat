@echo off
rem After the Kaggle run: brings the downloaded models into models\, combines them with the others and scores test.
rem Download from the notebook's Output: deberta_large_clean.zip, modernbert_large_clean.zip (or roberta_large_clean.zip)
rem and reports_kaggle.zip; they are looked for in your Downloads folder (another folder: run_after_kaggle.bat "D:\folder").
rem Run it from Anaconda Prompt in the training folder:   run_after_kaggle.bat
cd /d "%~dp0"
python 08_import_kaggle.py %* || goto :error
python 04_ensemble.py --models baseline_clean specter2_clean roberta_clean deberta_large_clean modernbert_large_clean roberta_large_clean --name ensemble_large || goto :error
python 05_evaluate.py --ensemble ensemble_large --compare ensemble_clean ensemble || goto :error
echo.
echo Done. Report: reports\05_test_report_large.xlsx
goto :eof
:error
echo.
echo Stopped (code %errorlevel%).
exit /b 1
