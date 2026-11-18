@echo off
rem Starts the local RAG chat and opens it in your browser.
rem Keep this window open while you use the chat; close it to stop the chat.
title Local RAG Chat
cd /d "%~dp0"

rem If the chat is already running, just open it.
curl -s -o nul http://127.0.0.1:8501/_stcore/health
if not errorlevel 1 (
    start "" http://localhost:8501
    exit /b
)

rem Start the Ollama background server (not the Ollama desktop window) only if it is not already running.
curl -s -o nul http://127.0.0.1:11434/api/version
if errorlevel 1 (
    echo Starting Ollama server...
    start "Ollama server" /min "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve
    timeout /t 3 /nobreak >nul
)

rem Open the browser once the chat server answers (headless mode skips Streamlit's first-run email prompt).
start "" /min powershell -NoProfile -WindowStyle Hidden -Command "for ($i = 0; $i -lt 60; $i++) { try { Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8501/_stcore/health | Out-Null; Start-Process 'http://localhost:8501'; break } catch { Start-Sleep 1 } }"

echo Local RAG Chat is running at http://localhost:8501
echo Close this window to stop it.
python -m streamlit run app.py --server.headless true --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
pause
