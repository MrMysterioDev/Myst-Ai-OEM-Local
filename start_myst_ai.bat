@echo off
setlocal
set "ROOT=E:\Myst_Studio"
set "MYST_AI=%ROOT%\11_Myst_AI"
echo Starting Mr.AI on http://127.0.0.1:5005
start "Myst_AI_5005" /min python "%MYST_AI%\myst_ai_server.py"
timeout /t 3 /nobreak >nul
echo Opening browser...
start "" "http://127.0.0.1:5005"
