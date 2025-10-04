@echo off
echo.
echo ===============================================
echo    🕵️ Multilingual Fake News Detector 🕵️
echo ===============================================
echo.
echo Choose your interface:
echo.
echo [1] Desktop GUI Application
echo [2] Web API Server  
echo [3] Test the System
echo [4] Exit
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" goto desktop
if "%choice%"=="2" goto api
if "%choice%"=="3" goto test
if "%choice%"=="4" goto end
goto invalid

:desktop
echo.
echo 🖥️ Starting Desktop GUI...
echo.
python desktop_gui.py
goto end

:api
echo.
echo 🌐 Starting Web API Server...
echo 📍 Access at: http://localhost:5000
echo.
python api.py
goto end

:test
echo.
echo 🧪 Running System Tests...
echo.
python test_project_comprehensive.py
pause
goto start

:invalid
echo.
echo ❌ Invalid choice. Please enter 1, 2, 3, or 4.
pause
goto start

:start
cls
goto :eof

:end
echo.
echo 👋 Thank you for using the Fake News Detector!
pause