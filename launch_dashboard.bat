@echo off
cd /d "d:\sql\project1\spotify_analytics"
echo Starting Spotify Analytics Dashboard...
echo Dashboard will be available at: http://localhost:8504
echo.
C:\Users\ASUS\AppData\Local\Programs\Python\Python311\python.exe -m streamlit run dashboard.py --server.port 8504 --browser.gatherUsageStats false
pause
