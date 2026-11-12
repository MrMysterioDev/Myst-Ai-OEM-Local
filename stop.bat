@echo off
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5005" ^| findstr "LISTENING"') do taskkill /F /PID %%a 2>nul
taskkill /F /IM koboldcpp.exe 2>nul
echo Stopped
pause
