# Начните здесь

Установите Python 3.11 или 3.12. Откройте PowerShell в папке ab-experiment:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m abtest demo --out local/first-demo --seed 42
```

Результат: local/first-demo/analysis/REPORT.md.
Готовый пример уже есть в demo/analysis/REPORT.md.
Все данные — учебная симуляция. Параметры, собственный CSV и ограничения описаны в README.md.
