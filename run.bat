@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo ============================================
echo  FX予想ダッシュボード 起動
echo ============================================

REM --- Python検出 ---
set "PY="
where py >nul 2>&1 && set "PY=py -3"
if "%PY%"=="" (
    where python >nul 2>&1 && python --version >nul 2>&1 && set "PY=python"
)
if "%PY%"=="" (
    echo.
    echo [ERROR] Python がインストールされていません。
    echo   https://www.python.org/downloads/windows/ から Python 3.11 以上をインストールしてください。
    echo   インストール時に「Add python.exe to PATH」にチェックを入れてください。
    echo.
    start https://www.python.org/downloads/windows/
    pause
    exit /b 1
)

REM --- 仮想環境 ---
if not exist ".venv" (
    echo [INFO] 仮想環境を作成中...
    %PY% -m venv .venv
)
call .venv\Scripts\activate.bat

echo [INFO] 依存パッケージをインストール中... (初回は数分かかります)
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt

echo.
echo [INFO] ブラウザで http://localhost:8501 を開きます。
echo        終了するにはこのウィンドウで Ctrl+C を押してください。
echo.
streamlit run app.py
pause
