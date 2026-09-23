#!/usr/bin/env python3
"""训练三个专家模型（Hunter, Sniper, Defender）"""

import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import pickle
import json
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger

# M3 Metal加速配置
try:
    # 尝试设置LightGBM使用Metal（M1/M2/M3 GPU）
    lgb.config.set(device='gpu', gpu_platform_id=1, gpu_device_id=0)
    logger.info("LightGBM configured for Metal GPU acceleration")
except:
    logger.info("LightGBM using CPU (Metal not available)")

class ModelTrainer:
    """模型训练器 - 训练三个专家模型"""
    
    def __init__(self, data_dir: str = "./data/models"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 模型配置
        self.model_configs = {
            'hunter': {
                'description': 'Trending状态专家，追求高赔率',
                'target_threshold': 0.02,  # 2%收益
                'time_horizon': 120,  # 2小时
                'params': {
                    'objective': 'binary',
                    'metric': 'auc',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1,
                    'num_threads': 4,  # M3优化
                    'device': 'cpu',  # 或'gpu'如果支持Metal
                }
            },
            'sniper': {
                'description': 'Ranging状态专家，追求高胜率',
                'target_threshold': 0.005,  # 0.5%收益
                'time_horizon': 30,  # 30分钟
                'params': {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': 15,
                    'learning_rate': 0.01,
                    'feature_fraction': 0.7,
                    'bagging_fraction': 0.7,
                    'bagging_freq': 5,
                    'verbose': -1,
                    'num_threads': 4,
                    'device': 'cpu',
                }
            },
            'defender': {
                'description': 'High_Vol/Crash状态专家，预测回撤',
                'target_threshold': -0.01,  # -1%回撤
                'time_horizon': 10,  # 10分钟
                'params': {
                    'objective': 'regression',
                    'metric': 'mae',
                    'boosting_type': 'gbdt',
                    'num_leaves': 20,
                    'learning_rate': 0.01,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1,
                    'num_threads': 4,
                    'device': 'cpu',
                }
            }
        }
        
        # 特征列（基于Day 3的因子）
        self.feature_columns = [
            'rsi_14', 'macd_hist', 'bollinger_pctb', 'rolling_vol_ratio',
            'basis_z_score', 'oi_change_rate', 'net_taker_flow',
            'downside_beta', 'funding_cost_pressure',
            'returns', 'range_pct', 'body_pct', 'volatility_5m', 'atr_pct_14',
            'hour', 'weekday', 'market_hours'
        ]
        
        logger.info("ModelTrainer initialized for M3 optimization")
    
    def generate_synthetic_data(self, n_samples: int = 10000) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """生成合成训练数据（用于演示，实际应使用真实数据）"""
        logger.info(f"Generating {n_samples} synthetic samples...")
        
        np.random.seed(42)
        
        # 生成特征
        features = {}
        for col in self.feature_columns:
            if col in ['rsi_14', 'bollinger_pctb']:
                features[col] = np.random.uniform(0, 100, n_samples)
            elif col in ['macd_hist', 'returns', 'range_pct', 'body_pct']:
                features[col] = np.random.normal(0, 0.01, n_samples)
            elif col in ['basis_z_score', 'net_taker_flow']:
                features[col] = np.random.normal(0, 1, n_samples)
            elif col in ['downside_beta', 'funding_cost_pressure']:
                features[col] = np.random.uniform(-0.1, 0.1, n_samples)
            elif col in ['hour']:
                features[col] = np.random.randint(0, 24, n_samples)
            elif col in ['weekday']:
                features[col] = np.random.randint(1, 8, n_samples)
            elif col in ['market_hours']:
                features[col] = np.random.randint(0, 2, n_samples)
            else:
                features[col] = np.random.normal(0, 1, n_samples)
        
        features_df = pd.DataFrame(features)
        
        # 生成标签
        labels = {}
        
        # Hunter标签：2小时内收益 > 2%
        future_returns_hunter = np.random.normal(0, 0.03, n_samples)
        labels['hunter'] = (future_returns_hunter > 0.02).astype(int)
        
        # Sniper标签：30分钟内收益 > 0.5%
        future_returns_sniper = np.random.normal(0, 0.01, n_samples)
        labels['sniper'] = (future_returns_sniper > 0.005).astype(int)
        
        # Defender标签：10分钟内最大回撤
        max_drawdown = -np.abs(np.random.normal(0, 0.02, n_samples))
        labels['defender'] = max_drawdown
        
        # 添加一些模式使学习成为可能
        self._add_patterns(features_df, labels)
        
        logger.info(f"Synthetic data generated: {features_df.shape}")
        logger.info(f"Label distributions:")
        for model_name, label in labels.items():
            if model_name != 'defender':
                logger.info(f"  {model_name}: {np.mean(label):.1%} positive")
            else:
                logger.info(f"  {model_name}: mean drawdown = {np.mean(label):.3%}")
        
        return features_df, labels
    
    def _add_patterns(self, features: pd.DataFrame, labels: Dict[str, np.ndarray]):
        """向合成数据添加一些可学习的模式"""
        n_samples = len(features)
        
        # 模式1：低RSI + 高成交量 -> 上涨概率增加
        mask1 = (features['rsi_14'] < 30) & (features['rolling_vol_ratio'] > 1.5)
        labels['hunter'][mask1] = np.random.binomial(1, 0.7, mask1.sum())
        labels['sniper'][mask1] = np.random.binomial(1, 0.8, mask1.sum())
        
        # 模式2：高RSI + 负资金费率 -> 下跌概率增加
        mask2 = (features['rsi_14'] > 70) & (features['funding_cost_pressure'] < 0)
        labels['hunter'][mask2] = 0
        labels['sniper'][mask2] = 0
        labels['defender'][mask2] = -0.03  # 更大回撤
        
        # 模式3：高波动率 + 负beta -> 高回撤风险
        mask3 = (features['volatility_5m'] > 0.02) & (features['downside_beta'] < -0.5)
        labels['defender'][mask3] = -0.05
        
        logger.info(f"Added learning patterns: {mask1.sum()} + {mask2.sum()} + {mask3.sum()} samples")
    
    def prepare_data(self, features_df: pd.DataFrame, labels: Dict[str, np.ndarray], 
                    model_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """为特定模型准备数据"""
        X = features_df.values
        y = labels[model_name]
        
        # 对于分类问题，检查类别平衡
        if model_name in ['hunter', 'sniper']:
            pos_ratio = np.mean(y)
            logger.info(f"{model_name}: positive ratio = {pos_ratio:.1%}")
            
            # 如果类别不平衡，调整样本权重
            if pos_ratio < 0.3 or pos_ratio > 0.7:
                sample_weight = np.where(y == 1, 1/pos_ratio, 1/(1-pos_ratio))
            else:
                sample_weight = None
        else:
            sample_weight = None
        
        # 时间序列分割（避免未来信息泄漏）
        tscv = TimeSeriesSplit(n_splits=5)
        
        # 使用最后一个fold作为测试集
        train_idx, test_idx = None, None
        for train_index, test_index in tscv.split(X):
            train_idx, test_idx = train_index, test_index
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    
    def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray = None, y_val: np.ndarray = None) -> lgb.Booster:
        """训练单个模型"""
        config = self.model_configs[model_name]
        logger.info(f"Training {model_name} model: {config['description']}")
        
        # 准备数据集
        train_data = lgb.Dataset(X_train, label=y_train)
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets = [train_data, val_data]
            valid_names = ['train', 'valid']
        else:
            valid_sets = [train_data]
            valid_names = ['train']
        
        # 训练模型
        model = lgb.train(
            config['params'],
            train_data,
            num_boost_round=1000,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=100, show_stdv=False)
            ]
        )
        
        logger.info(f"{model_name} model trained with {model.num_trees()} trees")
        return model
    
    def evaluate_model(self, model: lgb.Booster, model_name: str, 
                      X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """评估模型性能"""
        y_pred = model.predict(X_test)
        
        metrics = {}
        
        if model_name in ['hunter', 'sniper']:  # 分类模型
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            metrics['accuracy'] = accuracy_score(y_test, y_pred_binary)
            metrics['precision'] = precision_score(y_test, y_pred_binary, zero_division=0)
            metrics['recall'] = recall_score(y_test, y_pred_binary, zero_division=0)
            metrics['f1'] = f1_score(y_test, y_pred_binary, zero_division=0)
            metrics['auc'] = roc_auc_score(y_test, y_pred) if len(np.unique(y_test)) > 1 else 0.5
            
            logger.info(f"{model_name} test metrics:")
            for metric_name, value in metrics.items():
                logger.info(f"  {metric_name}: {value:.3f}")
                
        else:  # 回归模型（defender）
            metrics['mae'] = np.mean(np.abs(y_pred - y_test))
            metrics['mse'] = np.mean((y_pred - y_test) ** 2)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2'] = 1 - metrics['mse'] / np.var(y_test)
            
            logger.info(f"{model_name} test metrics:")
            for metric_name, value in metrics.items():
                logger.info(f"  {metric_name}: {value:.4f}")
        
        return metrics
    
    def save_model(self, model: lgb.Booster, model_name: str, scaler: StandardScaler,
                  metrics: Dict[str, float], feature_importance: np.ndarray):
        """保存模型和相关元数据"""
        model_dir = self.data_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        # 保存模型
        model_path = model_dir / f"{model_name}_model.txt"
        model.save_model(str(model_path))
        
        # 保存scaler
        scaler_path = model_dir / f"{model_name}_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # 保存特征重要性
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        importance_path = model_dir / f"{model_name}_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        
        # 保存元数据
        metadata = {
            'model_name': model_name,
            'description': self.model_configs[model_name]['description'],
            'training_date': datetime.now().isoformat(),
            'features': self.feature_columns,
            'metrics': metrics,
            'num_trees': model.num_trees(),
            'feature_importance_top5': importance_df.head(5).to_dict('records')
        }
        
        metadata_path = model_dir / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Top 5 important features for {model_name}:")
        for i, row in enumerate(importance_df.head(5).itertuples()):
            logger.info(f"  {i+1}. {row.feature}: {row.importance:.4f}")
        
        return str(model_path)
    
    def train_all_models(self, use_synthetic: bool = True, 
                        real_features: pd.DataFrame = None,
                        real_labels: Dict[str, np.ndarray] = None):
        """训练所有三个模型"""
        logger.info("Starting training of all 3 expert models...")
        
        # 准备数据
        if use_synthetic or real_features is None:
            features_df, labels = self.generate_synthetic_data(10000)
        else:
            features_df, labels = real_features, real_labels
        
        results = {}
        
        for model_name in ['hunter', 'sniper', 'defender']:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training {model_name.upper()} model")
            logger.info(f"{'='*60}")
            
            try:
                # 准备数据
                X_train, X_test, y_train, y_test, scaler = self.prepare_data(
                    features_df, labels, model_name
                )
                
                # 训练模型
                model = self.train_model(model_name, X_train, y_train)
                
                # 评估模型
                metrics = self.evaluate_model(model, model_name, X_test, y_test)
                
                # 获取特征重要性
                importance = model.feature_importance(importance_type='gain')
                
                # 保存模型
                model_path = self.save_model(model, model_name, scaler, metrics, importance)
                
                results[model_name] = {
                    'model_path': model_path,
                    'metrics': metrics,
                    'status': 'success'
                }
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                import traceback
                traceback.print_exc()
                results[model_name] = {'status': 'failed', 'error': str(e)}
        
        # 保存训练摘要
        summary_path = self.data_dir / "training_summary.json"
        summary = {
            'training_date': datetime.now().isoformat(),
            'data_source': 'synthetic' if use_synthetic else 'real',
            'n_samples': len(features_df),
            'results': results
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\nTraining summary saved to {summary_path}")
        logger.info("\nTraining complete!")
        
        return results

def main():
    """主训练函数"""
    logger.info("Model Training for DOGE-USDT Trading System")
    logger.info("=" * 60)
    
    # 创建训练器
    trainer = ModelTrainer()
    
    # 训练所有模型
    results = trainer.train_all_models(use_synthetic=True)
    
    # 打印结果
    logger.info("\n" + "=" * 60)
    logger.info("Training Results Summary")
    logger.info("=" * 60)
    
    for model_name, result in results.items():
        if result['status'] == 'success':
            logger.info(f"\n{model_name.upper()}: ✅ Success")
            for metric_name, value in result['metrics'].items():
                logger.info(f"  {metric_name}: {value:.4f}")
        else:
            logger.info(f"\n{model_name.upper()}: ❌ Failed - {result.get('error', 'Unknown error')}")
    
    logger.info("\n🎉 Model training complete! Models saved to ./data/models/")

if __name__ == "__main__":
    # 配置日志
    import sys
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>ModelTrainer</cyan> | <level>{message}</level>",
        level="INFO"
    )
    
    main()