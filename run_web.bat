@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" goto noenv

echo.
echo  FlowerVision Web UI starting ...
echo  User  : http://127.0.0.1:5000        (user  / 123456)
echo  Admin : http://127.0.0.1:5000/admin  (admin / 123456)
echo  Press Ctrl+C in this window to stop the server.
echo.

".venv\Scripts\python.exe" web_app.py
goto end

:noenv
echo [ERROR] .venv not found.
echo Run the setup steps in README.md first, then try again.
pause

:end
