# -*- coding: utf-8 -*-
"""
ButterQuant ML Model Training - Robust V2.0 / ButterQuant ML 模型训练 - 增强版 V2.0
专门针对严重类别不平衡的情况 / Specialized for severe class imbalance

关键修复 / Key Fixes:
1. 特征维度检查和自动填充 (Feature dimension check & auto-pad)
2. 类别权重计算修正 (反向权重) (Inverse frequency class weights)
3. 过采样/欠采样处理不平衡 (Oversampling/Undersampling)
4. ONNX导出错误处理 (Robust ONNX export)
5. 早停和正则化增强 (Early stopping & regularization)
"""

import sys
from pathlib import Path

# Add project root to path to resolve 'ml' module
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import joblib
import os
from ml.features import FeatureExtractor

# 设置日志 / Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiClassSuccessClassifier(nn.Module):
    """
    4分类模型 - 增强正则化 / 4-class model with enhanced regularization
    """
    
    def __init__(self, input_dim=22, hidden_dims=[64, 32, 16], num_classes=4, dropout=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class ModelTrainer:
    """模型训练器 - 修复版 / Robust Model Trainer"""
    
    def __init__(self, data_path: str, output_dir: str = "ml/models"):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备 / Using device: {self.device}")
        if self.device.type == 'cuda':
             logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

        self.config = {
            'input_dim': 22,
            'hidden_dims': [64, 32, 16],
            'num_classes': 4,
            'dropout': 0.3,  # 增加dropout / Increased dropout
            'learning_rate': 0.0005,  # 降低学习率 / Lower LR
            'batch_size': 64,
            'num_epochs': 50,
            'early_stopping_patience': 15,
            'test_size': 0.2,
            'val_size': 0.1,
            'random_state': 42,
            'use_class_weights': True,
            'use_sampling': True  # 是否使用采样平衡 / Use sampling balance
        }
        
        self.model = None
        self.scaler = None
        self.class_weights = None
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    def load_data(self):
        """加载并验证数据 / Load and validate data"""
        logger.info(f"📥 加载数据: {self.data_path}")
        
        df = pd.read_parquet(self.data_path)
        logger.info(f"已加载 {len(df)} 个样本 / Loaded {len(df)} samples.")
        
        # 使用统一的22维特征列表 / Use unified 22-dim feature list
        feature_cols = FeatureExtractor.FEATURE_NAMES
        
        # 验证特征是否完整 / Verify features exist
        missing_cols = [c for c in feature_cols if c not in df.columns]
        if missing_cols:
            logger.error(f"❌ 数据缺失关键特征 / Missing features: {missing_cols}")
            # 进行简单填充以防崩溃 / Pad to prevent crash
            for c in missing_cols:
                df[c] = 0.0
        
        logger.info(f"特征选择 / Feature selection: {len(feature_cols)} features")
        
        # 确保顺序一致，这里简单按列表顺序，实际应尽量与features.py对齐
        # 但既然是修复训练，只要保证输入维度对即可
        
        X = df[feature_cols].values
        y = df['label'].values
        
        # 数据质量检查 / Data quality check
        if np.isnan(X).any():
            logger.warning("⚠️ 特征中包含NaN, 替换为0 / Features contain NaN, replaced with 0")
            X = np.nan_to_num(X, nan=0.0)
        
        if np.isinf(X).any():
            logger.warning("⚠️ 特征中包含Inf, 替换为0 / Features contain Inf, replaced with 0")
            X = np.nan_to_num(X, posinf=0.0, neginf=0.0)
        
        # 标签分布 / Label distribution
        label_counts = pd.Series(y).value_counts().sort_index().to_dict()
        logger.info(f"标签分布 / Label distribution: {label_counts}")
        
        return X, y
    
    def balance_dataset(self, X, y):
        """平衡数据集 - 混合采样策略 / Balance dataset - Mixed sampling strategy"""
        from collections import Counter
        
        logger.info("🔄 应用数据平衡策略 / Applying balancing strategy...")
        
        class_counts = Counter(y)
        logger.info(f"原始分布 / Original dist: {dict(class_counts)}")
        
        # 策略: 欠采样多数类 + 过采样少数类
        max_count = max(class_counts.values())
        
        # 目标分布 (缓和版) / Target distribution (Soft)
        target_counts = {
            0: min(class_counts[0], int(max_count * 0.5)),  # 亏损类欠采样到50% / Loss: undersample to 50%
            1: min(class_counts[1] * 3, int(max_count * 0.3)),  # 微利类过采样3倍 / Minor: oversample 3x
            2: min(class_counts[2] * 2, int(max_count * 0.3)),  # 良好类过采样2倍 / Good: oversample 2x
            3: class_counts[3]  # 优秀类保持不变 / Excellent: keep
        }
        
        balanced_indices = []
        
        for cls in range(4):
            cls_indices = np.where(y == cls)[0]
            target = target_counts.get(cls, len(cls_indices))
            
            if len(cls_indices) == 0:
                continue

            if len(cls_indices) > target:
                # 欠采样 / Undersample
                sampled = np.random.choice(cls_indices, target, replace=False)
            else:
                # 过采样 / Oversample
                sampled = np.random.choice(cls_indices, target, replace=True)
            
            balanced_indices.extend(sampled)
        
        balanced_indices = np.array(balanced_indices)
        np.random.shuffle(balanced_indices)
        
        X_balanced = X[balanced_indices]
        y_balanced = y[balanced_indices]
        
        new_counts = Counter(y_balanced)
        logger.info(f"平衡后分布 / Balanced dist: {dict(new_counts)}")
        logger.info(f"样本数量 / Sample count: {len(y)} → {len(y_balanced)}")
        
        return X_balanced, y_balanced
    
    def compute_class_weights(self, y):
        """
        计算类别权重 - 反向频率权重 / Compute class weights - Inverse Frequency
        weight = 1 / frequency
        """
        from collections import Counter
        
        class_counts = Counter(y)
        total = len(y)
        
        # 计算频率 / Frequencies
        frequencies = {cls: count / total for cls, count in class_counts.items()}
        
        # 反向权重 / Inverse weights
        weights = []
        # 假设4类 / Assume 4 classes
        for cls in range(4):
            freq = frequencies.get(cls, 1e-6) # 避免除零
            weight = 1.0 / freq
            weights.append(weight)
        
        # 归一化 (使得平均权重为1) / Normalize (mean=1)
        weights = np.array(weights)
        weights = weights / weights.mean()
        
        logger.info(f"类别权重 / Class weights: {weights}")
        return torch.tensor(weights, dtype=torch.float32)
    
    def prepare_dataloaders(self, X, y):
        """准备数据加载器 / Prepare DataLoaders"""
        
        # 数据平衡 (仅对训练集?) -> 这里对全部数据进行了平衡，然后split。
        # 更好的做法通常是只对训练集做过采样，验证集保持原样。
        # 但为了简化实现，我们可以先split再balance训练集。
        
        # 1. 划分 Test (保持真实分布)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.config['test_size'], stratify=y, random_state=self.config['random_state']
        )
        
        # 2. 划分 Val (保持真实分布)
        X_train_raw, X_val, y_train_raw, y_val = train_test_split(
            X_temp, y_temp, test_size=0.15, stratify=y_temp, random_state=self.config['random_state']
        )
        
        # 3. 对 Train 进行平衡 (Sampling)
        if self.config['use_sampling']:
            X_train, y_train = self.balance_dataset(X_train_raw, y_train_raw)
        else:
            X_train, y_train = X_train_raw, y_train_raw

        logger.info(f"🔀 数据分割 / Data Split:")
        logger.info(f"   训练集 / Train: {len(X_train)}")
        logger.info(f"   验证集 / Val:   {len(X_val)}")
        logger.info(f"   测试集 / Test:  {len(X_test)}")
        
        # 标准化 / Scaling
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        # DataLoader
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader, model, criterion, optimizer):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            loss.backward()
            
            # 梯度裁剪 / Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
        
        return total_loss / len(train_loader), correct / total
    
    def validate_epoch(self, val_loader, model, criterion):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        return total_loss / len(val_loader), correct / total
    
    def train(self, X, y):
        """完整训练流程 / Complete training flow"""
        
        logger.info("🚀 开始训练 / Starting training...")
        
        # 准备数据 / Prepare Data
        train_loader, val_loader, test_loader = self.prepare_dataloaders(X, y)
        
        # 计算权重 (基于训练集或全局? 这里用全局简单些，或者基于均衡后的其实不需要权重了)
        # 如果已经做了Sampling平衡，class_weights可以设为None，或者弱化。
        # 这里为了保险，还是算一下，但基于y (原始) 还是 y_train (平衡后)?
        # 既然已经平衡了，CrossEntropy的weight应该靠近1。
        # 我们用平衡后的y_train来算权重，应该很接近1。
        # 或者，为了处理Sampling没完全平衡的部分，再加一层保险。
        
        # 获取训练集所有的标签用于计算权重
        all_train_labels = []
        for _, y_batch in train_loader:
            all_train_labels.extend(y_batch.numpy())
        
        if self.config['use_class_weights']:
            self.class_weights = self.compute_class_weights(all_train_labels)
        
        # 初始化模型 / Init Model
        self.model = MultiClassSuccessClassifier(
            input_dim=self.config['input_dim'],
            hidden_dims=self.config['hidden_dims'],
            num_classes=self.config['num_classes'],
            dropout=self.config['dropout']
        ).to(self.device)
        
        # 损失函数 / Criterion
        if self.class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # 训练循环 / Training Loop
        best_val_loss = float('inf')
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(self.config['num_epochs']):
            train_loss, train_acc = self.train_epoch(train_loader, self.model, criterion, optimizer)
            val_loss, val_acc = self.validate_epoch(val_loader, self.model, criterion)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{self.config['num_epochs']}], Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            scheduler.step(val_loss)
            
            # 保存最佳 (优先看Val Loss，其次看Acc)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc # Update this too
                patience_counter = 0
                self.save_checkpoint('best_model.pt')
            else:
                patience_counter += 1
            
            if patience_counter >= self.config['early_stopping_patience']:
                logger.info(f"⏸️ Early stopping at epoch {epoch+1}")
                break
        
        # 加载最佳模型 / Load best
        try:
            self.load_checkpoint('best_model.pt')
        except:
            logger.warning("无法加载最佳模型，使用最终模型 / Could not load best model, using final")
        
        # 评估 / Evaluate
        logger.info("\n评估报告 / Evaluation Report:")
        self.evaluate(test_loader)
        
        # 保存 / Save
        self.save_model()
        
        logger.info(f"✅ 模型已保存 / Model saved to {self.output_dir}/")
        logger.info(f"最佳验证准确率 / Best Val accuracy: {best_val_acc:.4f}")
    
    def evaluate(self, test_loader):
        """评估模型 / Evaluate"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # 报告
        target_names = ['Loss/亏损', 'Minor/微利', 'Good/良好', 'Excellent/优秀']
        # 处理可能出现的未预测到的列 (例如只有某些类被预测到)
        present_labels = sorted(list(set(all_labels) | set(all_preds)))
        target_names_subset = [target_names[i] for i in present_labels]
        
        logger.info("\n" + classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))
        
        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        logger.info(f"混淆矩阵 / Confusion Matrix:\n{cm}")
        
        try:
            self.plot_confusion_matrix(cm, target_names)
        except Exception as e:
            logger.warning(f"无法绘制混淆矩阵: {e}")
    
    def plot_confusion_matrix(self, cm, target_names):
        """绘制混淆矩阵 / Plot CM"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=150)
        plt.close()
    
    def save_checkpoint(self, filename):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, self.output_dir / filename)
    
    def load_checkpoint(self, filename):
        checkpoint = torch.load(self.output_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def save_model(self):
        """最后保存 / Final save"""
        # PyTorch
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'class_weights': self.class_weights
        }, self.output_dir / 'success_model_v2.pth') # Keep .pth standard
        
        # Scaler
        joblib.dump(self.scaler, self.output_dir / 'scaler_v2.joblib')
        
        # ONNX
        try:
            self.export_to_onnx()
        except Exception as e:
            logger.warning(f"ONNX导出失败 / ONNX export failed: {e}")
            logger.info("   可以稍后补救 / Can try later")

    def export_to_onnx(self):
        self.model.eval()
        dummy_input = torch.randn(1, self.config['input_dim'], device=self.device)
        
        torch.onnx.export(
            self.model,
            dummy_input,
            self.output_dir / 'success_model_v2.onnx',
            export_params=True,
            opset_version=14, # Newer opset
            input_names=['features'],
            output_names=['logits'],
            dynamic_axes={'features': {0: 'batch_size'}, 'logits': {0: 'batch_size'}}
        )
        logger.info("✅ ONNX模型已导出 / ONNX exported")


def main():
    data_path = "ml/training_data_deep.parquet"
    if not Path(data_path).exists():
        logger.error(f"❌ 数据文件不存在: {data_path}")
        return
    
    trainer = ModelTrainer(data_path)
    X, y = trainer.load_data()
    trainer.train(X, y)


if __name__ == "__main__":
    main()
