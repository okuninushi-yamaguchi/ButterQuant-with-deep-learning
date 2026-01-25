# ButterQuant v1.2: Long-term Operation & Maintenance Guide

This guide ensures the stable, automated operation of your AI trading system for the next 6+ months.

## 1. Daily Operational Workflow

To maintain consistency, follow this sequence every trading day:

1.  **Market Pre-Scan (15:00 - 15:30 EST)**:
    - Run `python backend/daily_scanner.py`.
    - Updates rankings and AI success probabilities based on the latest daily close.
2.  **Order Execution (15:30 - 16:00 EST or Market Open next day)**:
    - Run `python backend/execution_engine.py`.
    - Automatically manages the 20-slot portfolio and places orders in IBKR.

## 2. Automated Scheduling (Windows Task Scheduler)

To run without manual intervention, set up two tasks:

- **Task A (Scanner)**:
    - Trigger: Daily at 15:15.
    - Action: `powershell.exe -ExecutionPolicy Bypass -File "C:\path\to\run_scanner.ps1"`
- **Task B (Engine)**:
    - Trigger: Daily at 15:45.
    - Action: `powershell.exe -ExecutionPolicy Bypass -File "C:\path\to\run_engine.ps1"`

> [!TIP]
> Ensure IBKR TWS is open and logged in before these tasks trigger.

## 3. Maintenance Checklist

### Weekly (Every Weekend)
- **Check Logs**: Review `backend/logs/scanner.log` and terminal outputs for errors.
- **Database Backup**: Copy `backend/data/*.db` to a secure location/cloud.
- **Trade Journal Review**: Open the frontend and check the "Profit/Loss" of previous butterfly strategies.

### Monthly
- **Data Cleanup**: If `history.db` grows too large (>5GB), consider archiving old analysis data.
- **Model Drift Analysis**: Compare the "AI Probability" vs "Actual Hit Rate". If the hit rate significantly drops (e.g., < 50%), consider retraining.

### Semi-Annually (6 Months)
- **ML Retraining**:
    1. Export fresh data using `backend/database.py`.
    2. Run the training script with the latest market regimes.
    3. Update the ONNX model in `backend/models/`.

## 4. Performance Analysis Strategy

Use the **Trade Journal** to track:
- **Win Rate**: % of butterfly strategies that expired in the profit zone.
- **Average Yield**: Net profit per $1,000 allocation.
- **Max Drawdown**: Total account value dip during volatile periods.

---

**System Health Status**: ✅ READY FOR LIVE SIMULATION
**Last Maintenance Check**: 2026-01-13
心力物语 (Deep Learning Integration).
