@echo off
:menu
cls
echo ==========================================
echo       Quick GitHub Push Menu
echo ==========================================
echo.
echo Current repository status:
git status -s
echo.
echo ------------------------------------------
echo 1. Add all, Commit, and Push
echo 2. Exit
echo ------------------------------------------
set /p choice="Select an option (1-2): "

if "%choice%"=="1" goto dopush
if "%choice%"=="2" goto end
goto menu

:dopush
echo.
set /p commit_msg="Enter commit message (or press Enter for 'Update'): "
if "%commit_msg%"=="" set commit_msg=Update

echo.
echo [1/3] Adding files (git add .)...
git add .

echo [2/3] Creating commit (git commit -m "%commit_msg%")...
git commit -m "%commit_msg%"

echo [3/3] Pushing to server (git push)...
git push

echo.
echo Done!
pause
goto menu

:end
exit
