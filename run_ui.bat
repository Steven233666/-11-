@echo off
setlocal
cd /d "%~dp0"

set PY=%~dp0.venv\Scripts\python.exe
if not exist "%PY%" goto noenv

:menu
echo ============================================================
echo   Flower Classification - Choose a window
echo ============================================================
echo   1 = Local image classifier        4.classifier_ui.py
echo   2 = Camera real-time classifier   5.classifier_camera_ui.py
echo   3 = Ensemble predictor            7.distillation_predictor.py
echo   0 = Exit
echo.
set "choice="
set /p choice=Enter 1 / 2 / 3 / 0 then press Enter:
if "%choice%"=="1" goto ui4
if "%choice%"=="2" goto ui5
if "%choice%"=="3" goto ui7
if "%choice%"=="0" goto end
echo Invalid input, please run again.
pause
exit /b 1

:ui4
echo Starting 4.classifier_ui.py ...
"%PY%" "4.classifier_ui.py"
goto end

:ui5
echo Starting 5.classifier_camera_ui.py ...
"%PY%" "5.classifier_camera_ui.py"
goto end

:ui7
echo Starting 7.distillation_predictor.py ...
"%PY%" "7.distillation_predictor.py"
goto end

:noenv
echo [ERROR] Virtual environment not found: %PY%
echo Please create it first:
echo     python -m venv .venv
echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
pause
exit /b 1

:end
exit /b 0
