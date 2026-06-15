@echo off
call .venv\Scripts\activate.bat
set PYTHONPATH=%CD%
echo ======================================
echo  Enterprise RAG CS - Running Tests
echo ======================================
python -m pytest tests/ -v
echo.
echo ======================================
echo  Tests Complete
echo ======================================
pause
