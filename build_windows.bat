@echo off
setlocal

where py >nul 2>nul
if errorlevel 1 (
    echo Python's Windows launcher was not found.
    echo Install Python 3.11 or newer on the build computer, then try again.
    exit /b 1
)

py -m pip install --upgrade pip
if errorlevel 1 exit /b 1

py -m pip install -r requirements.txt -r requirements-build.txt
if errorlevel 1 exit /b 1

py -m PyInstaller --clean --noconfirm AutoAnki.spec
if errorlevel 1 exit /b 1

echo.
echo Build complete: dist\AutoAnki.exe
endlocal
