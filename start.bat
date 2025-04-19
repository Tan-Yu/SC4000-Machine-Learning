@echo off
:: Save current directory
set "CUR_DIR=%~dp0"

:: Check for admin rights
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting administrative privileges...
    powershell -Command "Start-Process '%~f0' -WorkingDirectory '%CUR_DIR%' -Verb runAs"
    exit /b
)

:: Return to original directory (in case it's not already there)
cd /d "%CUR_DIR%"

REM Media Campaign Cost Prediction Project Setup
echo Setting up Media Campaign Cost Prediction environment...

REM Create virtual environment
python -m venv venv

REM Activate virtual environment
IF EXIST "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) ELSE (
    echo Failed to activate virtual environment.
    echo Press any key to exit...
    pause >nul
    exit /b
)

REM Install dependencies
echo Installing required packages...
pip install -r requirements.txt

REM Install Jupyter notebook
echo Installing Jupyter notebook...
pip install jupyter

REM Create figures directory
echo Creating figures directory...
if not exist src\figures (
    mkdir src\figures
)

REM Check if datasets exist
echo Checking for datasets...
set missing=0

if not exist data/train.csv (
    set missing=1
)
if not exist data/test.csv (
    set missing=1
)

if %missing%==1 (
    echo WARNING: One or more datasets are missing. Please ensure you have the following files in the project directory:
    echo - data/train.csv (360,336 samples)
    echo - data/test.csv (240,224 samples)
)

echo Setup complete! To start working with the project:
echo 1. The environment is now activated
echo 2. To run the Python script: python src\assignment.py
echo 3. To run the Jupyter notebook: jupyter notebook
echo 4. To deactivate the virtual environment: deactivate

echo.
echo Press any key to exit...
pause >nul
