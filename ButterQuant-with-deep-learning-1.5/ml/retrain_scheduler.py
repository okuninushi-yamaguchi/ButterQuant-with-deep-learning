# -*- coding: utf-8 -*-
"""
ML模型重训调度器 / ML Model Retrain Scheduler

实现反馈闭环机制 / Implements feedback loop mechanism:
1. 定期检查模型表现
2. 收集实际交易结果
3. 自动触发重训

用法 / Usage:
    # 检查是否需要重训 / Check if retrain needed
    python ml/retrain_scheduler.py --check
    
    # 执行完整重训流程 / Execute full retrain
    python ml/retrain_scheduler.py --retrain
    
    # 生成重训报告 / Generate retrain report
    python ml/retrain_scheduler.py --report
"""

import sys
import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import shutil

# 添加项目路径 / Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'backend'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RetrainScheduler:
    """
    模型重训调度器 / Model Retrain Scheduler
    
    决策逻辑 / Decision Logic:
    1. 距离上次训练超过30天 → 建议重训
    2. 预测准确率下降超过10% → 建议重训
    3. 新增训练样本超过1000条 → 建议重训
    """
    
    def __init__(self):
        self.ml_dir = PROJECT_ROOT / 'ml'
        self.models_dir = self.ml_dir / 'models'
        self.backup_dir = self.ml_dir / 'models_backup'
        self.data_dir = PROJECT_ROOT / 'backend' / 'data'
        
        # 配置 / Configuration
        self.config = {
            'retrain_interval_days': 30,  # 最少间隔天数 / Min days between retrains
            'accuracy_drop_threshold': 0.10,  # 准确率下降阈值 / Accuracy drop threshold
            'min_new_samples': 10000,  # 最少新增样本数 / Min new samples for retrain (每天扫描500+股票)
        }
        
        # 状态文件 / State file
        self.state_file = self.ml_dir / 'retrain_state.json'
    
    def load_state(self):
        """加载重训状态 / Load retrain state"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            'last_retrain_date': None,
            'last_accuracy': None,
            'training_samples_at_last_retrain': 0
        }
    
    def save_state(self, state):
        """保存重训状态 / Save retrain state"""
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def check_retrain_needed(self) -> dict:
        """
        检查是否需要重训 / Check if retrain is needed
        
        返回 / Returns:
            {
                'should_retrain': bool,
                'reasons': list[str],
                'details': dict
            }
        """
        logger.info("🔍 检查是否需要重训 / Checking if retrain needed...")
        
        state = self.load_state()
        reasons = []
        details = {}
        
        # 检查1: 时间间隔 / Check 1: Time interval
        if state['last_retrain_date']:
            last_date = datetime.fromisoformat(state['last_retrain_date'])
            days_since = (datetime.now() - last_date).days
            details['days_since_last_retrain'] = days_since
            
            if days_since >= self.config['retrain_interval_days']:
                reasons.append(f"距离上次训练已 {days_since} 天 (阈值: {self.config['retrain_interval_days']})")
        else:
            details['days_since_last_retrain'] = 'N/A (首次)'
            reasons.append("无历史训练记录，建议初始训练")
        
        # 检查2: 训练数据量 / Check 2: Training data volume
        training_data_path = self.ml_dir / 'training_data_deep.parquet'
        if training_data_path.exists():
            import pandas as pd
            df = pd.read_parquet(training_data_path)
            current_samples = len(df)
            details['current_samples'] = current_samples
            
            prev_samples = state.get('training_samples_at_last_retrain', 0)
            new_samples = current_samples - prev_samples
            details['new_samples_since_retrain'] = new_samples
            
            if new_samples >= self.config['min_new_samples']:
                reasons.append(f"新增 {new_samples} 条训练样本 (阈值: {self.config['min_new_samples']})")
        else:
            details['current_samples'] = 0
            details['new_samples_since_retrain'] = 0
        
        # 检查3: 模型文件存在性 / Check 3: Model file existence
        model_path = self.models_dir / 'success_model_v2.onnx'
        scaler_path = self.models_dir / 'scaler_v2.joblib'
        
        details['model_exists'] = model_path.exists()
        details['scaler_exists'] = scaler_path.exists()
        
        if not model_path.exists() or not scaler_path.exists():
            reasons.append("模型文件缺失，需要训练")
        
        should_retrain = len(reasons) > 0
        
        return {
            'should_retrain': should_retrain,
            'reasons': reasons,
            'details': details
        }
    
    def backup_current_model(self):
        """备份当前模型 / Backup current model"""
        if not self.models_dir.exists():
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = self.backup_dir / f'backup_{timestamp}'
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # 复制模型文件 / Copy model files
        files_to_backup = [
            'success_model_v2.onnx',
            'success_model_v2.pth',
            'scaler_v2.joblib',
            'confusion_matrix.png'
        ]
        
        for filename in files_to_backup:
            src = self.models_dir / filename
            if src.exists():
                shutil.copy2(src, backup_path / filename)
        
        logger.info(f"✅ 模型已备份到 / Model backed up to: {backup_path}")
        return backup_path
    
    def run_retrain(self, force: bool = False):
        """
        执行重训流程 / Execute retrain process
        
        参数 / Parameters:
            force: 强制重训 / Force retrain regardless of checks
        """
        logger.info("=" * 60)
        logger.info("🔄 ML 模型重训流程 / Model Retrain Process")
        logger.info("=" * 60)
        
        # Step 1: 检查是否需要重训 / Check if retrain needed
        if not force:
            check_result = self.check_retrain_needed()
            
            if not check_result['should_retrain']:
                logger.info("✅ 当前不需要重训 / Retrain not needed at this time")
                return False
            
            logger.info(f"⚠️ 需要重训,原因 / Retrain needed, reasons:")
            for reason in check_result['reasons']:
                logger.info(f"   - {reason}")
        else:
            logger.info("⚠️ 强制重训模式 / Force retrain mode")
        
        # Step 2: 备份当前模型 / Backup current model
        logger.info("\n📦 Step 1: 备份当前模型...")
        self.backup_current_model()
        
        # Step 3: 执行训练 / Execute training
        logger.info("\n🚀 Step 2: 执行模型训练...")
        
        try:
            # 导入训练模块 / Import training module
            from ml.train_model import ModelTrainer
            
            data_path = self.ml_dir / 'training_data_deep.parquet'
            if not data_path.exists():
                logger.error(f"❌ 训练数据不存在: {data_path}")
                return False
            
            trainer = ModelTrainer(str(data_path))
            X, y = trainer.load_data()
            trainer.train(X, y)
            
            logger.info("✅ 训练完成 / Training completed")
            
        except Exception as e:
            logger.error(f"❌ 训练失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Step 4: 更新状态 / Update state
        logger.info("\n📝 Step 3: 更新重训状态...")
        
        import pandas as pd
        df = pd.read_parquet(self.ml_dir / 'training_data_deep.parquet')
        
        state = {
            'last_retrain_date': datetime.now().isoformat(),
            'last_accuracy': None,  # TODO: 从训练结果中提取
            'training_samples_at_last_retrain': len(df)
        }
        self.save_state(state)
        
        logger.info("=" * 60)
        logger.info("✅ 重训流程完成 / Retrain process completed")
        logger.info("=" * 60)
        
        return True
    
    def generate_report(self):
        """生成重训报告 / Generate retrain report"""
        logger.info("=" * 60)
        logger.info("📊 ML 重训状态报告 / Retrain Status Report")
        logger.info("=" * 60)
        
        state = self.load_state()
        check_result = self.check_retrain_needed()
        
        print(f"\n上次重训时间 / Last retrain: {state.get('last_retrain_date', 'Never')}")
        print(f"上次训练样本数 / Samples at last retrain: {state.get('training_samples_at_last_retrain', 0)}")
        
        print(f"\n当前状态 / Current status:")
        for key, value in check_result['details'].items():
            print(f"   {key}: {value}")
        
        print(f"\n是否需要重训 / Retrain needed: {'✅ 是' if check_result['should_retrain'] else '❌ 否'}")
        
        if check_result['reasons']:
            print("\n原因 / Reasons:")
            for reason in check_result['reasons']:
                print(f"   - {reason}")
        
        # 列出备份 / List backups
        if self.backup_dir.exists():
            backups = list(self.backup_dir.iterdir())
            if backups:
                print(f"\n备份数量 / Number of backups: {len(backups)}")
                print("最近备份 / Recent backups:")
                for backup in sorted(backups, reverse=True)[:3]:
                    print(f"   - {backup.name}")


def main():
    parser = argparse.ArgumentParser(description='ML模型重训调度器 / ML Retrain Scheduler')
    parser.add_argument('--check', action='store_true', help='检查是否需要重训 / Check if retrain needed')
    parser.add_argument('--retrain', action='store_true', help='执行重训 / Execute retrain')
    parser.add_argument('--force', action='store_true', help='强制重训 / Force retrain')
    parser.add_argument('--report', action='store_true', help='生成报告 / Generate report')
    args = parser.parse_args()
    
    scheduler = RetrainScheduler()
    
    if args.check:
        result = scheduler.check_retrain_needed()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    elif args.retrain:
        scheduler.run_retrain(force=args.force)
        
    elif args.report:
        scheduler.generate_report()
        
    else:
        # 默认: 显示帮助 / Default: show help
        print("用法 / Usage:")
        print("  python ml/retrain_scheduler.py --check   # 检查是否需要重训")
        print("  python ml/retrain_scheduler.py --retrain # 执行重训")
        print("  python ml/retrain_scheduler.py --report  # 生成报告")


if __name__ == "__main__":
    main()
