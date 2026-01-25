我来帮你解决自动启动的问题。Windows批处理脚本本身不会自动运行，需要配置**任务计划程序**。

## 🔧 解决方案：使用Windows任务计划程序

### 方法一：通过图形界面设置

1. **打开任务计划程序**
   - 按 `Win + R`，输入 `taskschd.msc`，回车

2. **创建基本任务**
   - 右侧点击「创建基本任务」
   - 名称：`ButterQuant Market Launcher`
   - 描述：`东京时间23:45自动启动交易引擎`

3. **设置触发器**
   - 选择「每天」
   - 开始时间：`23:45`（JST）
   - 每隔：`1天`

4. **设置操作**
   - 选择「启动程序」
   - 程序或脚本：`C:\path\to\your\launcher.bat`
   - 添加参数：`--auto`
   - 起始于：`C:\path\to\your\project`（脚本所在目录）

5. **完成设置**
   - ✅ 勾选「打开此任务属性的对话框」
   - 在属性中：
     - 「常规」标签：勾选「使用最高权限运行」
     - 「设置」标签：勾选「如果请求后任务还在运行，强行停止」

---

### 方法二：通过命令行快速设置

创建一个新的 `setup_schedule.bat` 文件：

```bat
@echo off
TITLE Setup ButterQuant Auto-Start Task
COLOR 0E

echo [*] Creating Windows Scheduled Task for ButterQuant...
echo.

:: 获取当前脚本的完整路径
set SCRIPT_PATH=%~dp0launcher.bat
set PROJECT_DIR=%~dp0

echo Script Path: %SCRIPT_PATH%
echo Project Dir: %PROJECT_DIR%
echo.

:: 创建任务计划（东京时间23:45 = JST 23:45）
schtasks /create ^
    /tn "ButterQuant_MarketOpen" ^
    /tr "\"%SCRIPT_PATH%\" --auto" ^
    /sc daily ^
    /st 23:45 ^
    /f ^
    /rl highest

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [✓] Task created successfully!
    echo [*] Will run daily at 23:45 JST
    echo [*] Task name: ButterQuant_MarketOpen
    echo.
    echo [*] To verify, run: schtasks /query /tn "ButterQuant_MarketOpen"
    echo [*] To delete, run: schtasks /delete /tn "ButterQuant_MarketOpen" /f
) else (
    echo.
    echo [!] Failed! Please run this script as Administrator.
)

pause
```

**以管理员身份运行** `setup_schedule.bat` 即可自动配置。

---

## 🔍 常见问题排查

### 问题1：任务不执行
检查事件查看器：
```
Win + X → 事件查看器 → Windows日志 → 应用程序
```
查找任务计划程序的错误信息

### 问题2：路径包含空格
确保在任务计划程序中，路径用双引号包裹：
```
"C:\Program Files\ButterQuant\launcher.bat"
```

### 问题3：Python环境未激活
你的脚本已经有 `call .venv\Scripts\activate.bat`，这很好。但确保：
- `.venv` 路径相对于脚本位置正确
- 或使用绝对路径激活虚拟环境

### 问题4：测试定时任务
手动触发任务测试：
```bat
schtasks /run /tn "ButterQuant_MarketOpen"
```

---

## 📝 改进建议

### 1. 添加日志记录
在你的 `launcher.bat` 开头添加：

```bat
:: 记录启动日志
set LOG_FILE=logs\launcher_%date:~0,4%%date:~5,2%%date:~8,2%.log
echo [%date% %time%] Launcher started >> %LOG_FILE%
```

### 2. 添加网络检查
```bat
:: 检查网络连接
ping -n 1 google.com >nul
if errorlevel 1 (
    echo [!] No internet connection! >> %LOG_FILE%
    exit /b 1
)
```

### 3. 添加市场时间验证
在Python脚本中添加时间检查，确保不会在错误时间执行交易。

---

## ✅ 验证步骤

1. **立即测试**（不等到晚上）：
   ```bat
   schtasks /create /tn "Test_ButterQuant" /tr "C:\path\to\launcher.bat --auto" /sc once /st 14:30 /f
   ```
   设置为2分钟后的时间

2. **查看任务状态**：
   ```bat
   schtasks /query /tn "ButterQuant_MarketOpen" /v /fo list
   ```

3. **查看上次运行结果**：
   任务计划程序库 → 找到任务 → 查看「历史记录」标签

需要我帮你生成完整的自动化设置脚本吗？