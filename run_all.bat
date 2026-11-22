@echo off
setlocal
cd /d "%~dp0"

set PY=%~dp0.venv\Scripts\python.exe
if not exist "%PY%" goto noenv

echo ============================================================
echo   Flower Classification - Full Pipeline
echo   (download - split - train CNN - distill)
echo ============================================================
echo.

echo [1/4] Download images (existing files are skipped)
"%PY%" "1.download_data.py"
if errorlevel 1 goto failed

echo.
echo [2/4] Split train / test 8:2
"%PY%" "2.split_data.py"
if errorlevel 1 goto failed

echo.
echo [3/4] Train teacher CNN (about 10 minutes on CPU)
"%PY%" "3.train_cnn.py"
if errorlevel 1 goto failed

echo.
echo [4/4] Knowledge distillation
"%PY%" "6.distillation_cnn.py"
if errorlevel 1 goto failed

echo.
echo All done. Models are in models\ , charts are in the project root.
echo Next: double-click "run_ui.bat" or run 4.classifier_ui.py
pause
exit /b 0

:noenv
echo [ERROR] Virtual environment not found: %PY%
echo Please create it first:
echo     python -m venv .venv
echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
pause
exit /b 1

:failed
echo.
echo [FAILED] The last step reported an error. See the message above.
pause
exit /b 1
