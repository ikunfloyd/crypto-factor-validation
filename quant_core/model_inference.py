#!/usr/bin/env python3
"""模型推理引擎 - 加载三个专家模型并进行推理"""

import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
import pickle
import json
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path
from loguru import logger

# M3 Metal加速
try:
    lgb.config.set(device='gpu', gpu_platform_id=1, gpu_device_id=0)
    logger.info("LightGBM inference configured for Metal GPU")
except:
    logger.info("LightGBM inference using CPU")

class ModelInference:
    """模型推理引擎（Mixture of Experts）"""
    
    def __init__(self, model_dir: str = "./data/models", fallback_mode: bool = True):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.scalers = {}
        self.metadata = {}
        self.fallback_mode = fallback_mode  # 是否启用降级模式
        self.model_status = {}  # 跟踪每个模型的状态
        
        # 动态门控权重
        self.gating_weights = {
            'trending': {'hunter': 0.7, 'sniper': 0.3, 'defender': 0.0},
            'ranging': {'hunter': 0.2, 'sniper': 0.8, 'defender': 0.0},
            'high_vol': {'hunter': 0.0, 'sniper': 0.0, 'defender': 1.0},
            'crash': {'hunter': 0.0, 'sniper': 0.0, 'defender': 1.0}
        }
        
        # 特征顺序（必须与训练时一致）
        self.feature_columns = [
            'rsi_14', 'macd_hist', 'bollinger_pctb', 'rolling_vol_ratio',
            'basis_z_score', 'oi_change_rate', 'net_taker_flow',
            'downside_beta', 'funding_cost_pressure',
            'returns', 'range_pct', 'body_pct', 'volatility_5m', 'atr_pct_14',
            'hour', 'weekday', 'market_hours'
        ]
        
        # 加载模型
        self._load_models()
        
        # 检查是否需要降级模式
        if len(self.models) == 0 and self.fallback_mode:
            logger.warning("⚠️  No models loaded, using fallback rule-based strategy")
        elif len(self.models) < 3:
            logger.warning(f"⚠️  Only {len(self.models)}/3 models loaded, using hybrid strategy")
        
        logger.info("ModelInference initialized with MoE architecture")
    
    def _load_models(self):
        """加载所有专家模型，带有详细的错误处理和状态跟踪"""
        model_names = ['hunter', 'sniper', 'defender']
        
        for model_name in model_names:
            self.model_status[model_name] = {'loaded': False, 'error': None}
            
            # 模型由 quant_core/training/train_models.py 生成，保存为 LightGBM 原生
            # .txt 格式（首选），同目录另有 .pkl 兼容格式。若两者都不存在，则在下方
            # 走 fallback 规则策略（并提示运行训练脚本）。
            model_path_txt = self.model_dir / model_name / f"{model_name}_model.txt"
            model_path_pkl = self.model_dir / model_name / f"{model_name}_model.pkl"

            if model_path_txt.exists():
                model_path = model_path_txt
            elif model_path_pkl.exists():
                model_path = model_path_pkl
            else:
                model_path = model_path_txt  # 使用首选路径以便后续错误处理
            scaler_path = self.model_dir / model_name / f"{model_name}_scaler.pkl"
            metadata_path = self.model_dir / model_name / f"{model_name}_metadata.json"
            
            try:
                # 检查模型文件是否存在（如果已经在上面加载，则跳过）
                if model_name in self.models:
                    continue
                    
                if not model_path.exists():
                    error_msg = f"Model file not found: {model_path}. Please train models or create symlinks."
                    logger.warning(f"❌ {error_msg}")
                    self.model_status[model_name]['error'] = error_msg
                    continue
                
                # 加载模型
                try:
                    self.models[model_name] = lgb.Booster(model_file=str(model_path))
                    logger.info(f"✅ Loaded {model_name} model from {model_path}")
                except Exception as e:
                    error_msg = f"Failed to load model file: {e}"
                    logger.error(f"❌ {model_name}: {error_msg}")
                    self.model_status[model_name]['error'] = error_msg
                    continue
                
                # 加载scaler（可选）
                if scaler_path.exists():
                    try:
                        with open(scaler_path, 'rb') as f:
                            self.scalers[model_name] = pickle.load(f)
                        logger.info(f"✅ Loaded {model_name} scaler")
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to load scaler for {model_name}: {e}")
                        # Scaler不是必需的，可以继续
                else:
                    logger.warning(f"⚠️  Scaler file not found for {model_name}: {scaler_path}")
                
                # 加载元数据（可选）
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            self.metadata[model_name] = json.load(f)
                        logger.debug(f"Loaded metadata for {model_name}")
                    except Exception as e:
                        logger.warning(f"Failed to load metadata for {model_name}: {e}")
                
                # 标记为成功加载
                self.model_status[model_name]['loaded'] = True
                
            except Exception as e:
                error_msg = f"Unexpected error loading {model_name}: {e}"
                logger.error(f"❌ {error_msg}")
                self.model_status[model_name]['error'] = error_msg
        
        # 汇总加载结果
        loaded_count = sum(1 for status in self.model_status.values() if status['loaded'])
        if loaded_count == 3:
            logger.info("✅ All 3 expert models loaded successfully")
        elif loaded_count > 0:
            logger.warning(f"⚠️  Only {loaded_count}/3 models loaded: {[k for k, v in self.model_status.items() if v['loaded']]}")
        else:
            logger.error("❌ No models loaded! Using fallback strategy")
            
        # 创建模型目录结构（如果不存在）
        if loaded_count == 0 and self.fallback_mode:
            self._create_model_directories()
    
    def _create_model_directories(self):
        """创建模型目录结构"""
        model_names = ['hunter', 'sniper', 'defender']
        for model_name in model_names:
            model_dir = self.model_dir / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created model directory structure at {self.model_dir}")
    
    def prepare_features(self, factors_df: pl.DataFrame,
                        candles_df: "pl.DataFrame" = None) -> Optional[np.ndarray]:
        """从因子数据准备特征向量。

        17 个特征里有 8 个是 DataProcessor 计算的 K 线派生特征
        （returns/range_pct/body_pct/volatility_5m/atr_pct_14/hour/weekday/market_hours），
        它们不在 factors_df 里。传入已处理的 candles_df 才能取到真实值，
        否则这些特征只能退化为默认值（会造成训练/推理特征错位）。
        """
        try:
            if factors_df is None or len(factors_df) == 0:
                logger.warning("No factor data for inference")
                return None

            # 获取最新的因子值 / K线派生特征
            latest_factors = factors_df.tail(1)
            latest_candles = (candles_df.tail(1)
                              if candles_df is not None and len(candles_df) > 0 else None)

            def _from(frame, name):
                if frame is not None and name in frame.columns:
                    val = frame[name][0]
                    return None if val is None else float(val)
                return None

            # 构建特征字典：因子优先，其次 K 线派生特征，最后默认值
            features_dict = {}
            for feature in self.feature_columns:
                val = _from(latest_factors, feature)
                if val is None:
                    val = _from(latest_candles, feature)
                if val is not None:
                    features_dict[feature] = val
                elif feature in ['rsi_14', 'bollinger_pctb']:
                    features_dict[feature] = 50.0
                elif feature == 'hour':
                    features_dict[feature] = datetime.now().hour
                elif feature == 'weekday':
                    features_dict[feature] = datetime.now().isoweekday()
                elif feature == 'market_hours':
                    features_dict[feature] = 1 if 9 <= datetime.now().hour < 17 else 0
                else:
                    features_dict[feature] = 0.0
            
            # 转换为DataFrame确保顺序
            features_df = pd.DataFrame([features_dict])
            
            # 确保特征顺序一致
            features_df = features_df[self.feature_columns]
            
            logger.debug(f"Prepared features: {features_df.iloc[0].to_dict()}")
            return features_df.values
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None
    
    def predict_single_model(self, model_name: str, features: np.ndarray) -> float:
        """使用单个模型进行预测"""
        try:
            if model_name not in self.models:
                logger.error(f"Model {model_name} not loaded")
                return 0.0
            
            # 标准化特征
            if model_name in self.scalers:
                features_scaled = self.scalers[model_name].transform(features)
            else:
                features_scaled = features
            
            # 进行预测
            prediction = self.models[model_name].predict(features_scaled)[0]
            
            # 对于defender模型，预测的是回撤值（负数），我们转换为风险概率
            if model_name == 'defender':
                # 回撤越负，风险越高
                risk_prob = min(1.0, max(0.0, abs(prediction) * 10))  # 简单转换
                return risk_prob
            else:
                # 对于分类模型，返回的是概率
                return float(prediction)
                
        except Exception as e:
            logger.error(f"Error predicting with {model_name}: {e}")
            return 0.0
    
    def _fallback_prediction(self, features: np.ndarray, regime: str) -> Dict[str, Any]:
        """
        基于规则的降级预测策略（当模型未加载时）
        使用简单的技术指标规则
        """
        try:
            # 提取关键技术指标
            feature_dict = {}
            for i, col in enumerate(self.feature_columns):
                feature_dict[col] = features[0][i] if len(features.shape) > 1 else features[i]
            
            # 基于RSI和MACD的简单规则
            rsi = feature_dict.get('rsi_14', 50)
            macd = feature_dict.get('macd_hist', 0)
            vol_ratio = feature_dict.get('rolling_vol_ratio', 1.0)
            
            # 简单的规则策略
            score = 0.5  # 中性起点
            
            # RSI信号
            if rsi < 30:
                score += 0.3  # 超卖，看涨
            elif rsi > 70:
                score -= 0.3  # 超买，看跌
            
            # MACD信号
            if macd > 0:
                score += 0.2
            elif macd < 0:
                score -= 0.2
            
            # 波动率调整
            if vol_ratio > 2.0:
                score *= 0.5  # 高波动时减小仓位
            
            predictions = {
                'hunter': score,
                'sniper': score,
                'defender': 1.0 if vol_ratio > 3.0 else 0.0
            }
            
            weights = self.gating_weights.get(regime, self.gating_weights['trending'])
            signal = self._generate_signal(score, predictions, regime)
            
            return {
                'timestamp': datetime.now().timestamp(),
                'regime': regime,
                'final_score': score,
                'predictions': predictions,
                'weights': weights,
                'signal': signal,
                'fallback_mode': True,
                'warning': 'Using rule-based fallback strategy'
            }
            
        except Exception as e:
            logger.error(f"Error in fallback prediction: {e}")
            return {'error': str(e)}
    
    def predict_with_moe(self, features: np.ndarray,
                        regime: str = 'trending') -> Dict[str, Any]:
        """
        使用混合专家模型进行预测
        
        Args:
            features: 特征向量
            regime: 市场状态（trending, ranging, high_vol, crash）
        """
        try:
            if features is None or len(features) == 0:
                return {'error': 'No features provided'}
            
            # 获取各个专家的预测
            predictions = {}
            for model_name in ['hunter', 'sniper', 'defender']:
                if model_name in self.models:
                    predictions[model_name] = self.predict_single_model(model_name, features)
                else:
                    predictions[model_name] = 0.0
            
            # 获取门控权重
            if regime in self.gating_weights:
                weights = self.gating_weights[regime]
            else:
                weights = self.gating_weights['trending']  # 默认
            
            # 计算最终得分
            final_score = 0.0
            for model_name, weight in weights.items():
                if model_name in predictions:
                    final_score += weight * predictions[model_name]
            
            # 熔断机制：如果defender预测风险概率 > 80%，强制清仓
            if predictions.get('defender', 0) > 0.8:
                final_score = -1.0
                logger.warning(f"🚨 Defender熔断触发: risk={predictions['defender']:.1%}")
            
            # 生成信号
            signal = self._generate_signal(final_score, predictions, regime)
            
            result = {
                'timestamp': datetime.now().timestamp(),
                'regime': regime,
                'final_score': final_score,
                'predictions': predictions,
                'weights': weights,
                'signal': signal,
                'features_used': len(self.feature_columns)
            }
            
            logger.info(f"MoE prediction: regime={regime}, score={final_score:.3f}, signal={signal['direction']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in MoE prediction: {e}")
            return {'error': str(e)}
    
    def _generate_signal(self, final_score: float, 
                        predictions: Dict[str, float],
                        regime: str) -> Dict[str, Any]:
        """根据预测结果生成交易信号"""
        
        # 信号阈值
        thresholds = {
            'trending': {'long': 0.6, 'short': 0.4},
            'ranging': {'long': 0.7, 'short': 0.3},
            'high_vol': {'long': 0.8, 'short': 0.2},
            'crash': {'long': 0.9, 'short': 0.1}
        }
        
        current_threshold = thresholds.get(regime, thresholds['trending'])
        
        # 确定方向
        if final_score >= current_threshold['long']:
            direction = 'LONG'
            confidence = min(1.0, (final_score - current_threshold['long']) / (1 - current_threshold['long']))
        elif final_score <= current_threshold['short']:
            direction = 'SHORT'
            confidence = min(1.0, (current_threshold['short'] - final_score) / current_threshold['short'])
        else:
            direction = 'NEUTRAL'
            confidence = 0.0
        
        # 计算仓位大小（基于置信度）
        if direction != 'NEUTRAL':
            # 基础仓位
            base_position = 0.5
            
            # 根据regime调整
            if regime == 'trending':
                position = base_position * confidence
            elif regime == 'ranging':
                position = base_position * 0.7 * confidence  # 震荡市减仓
            elif regime in ['high_vol', 'crash']:
                position = base_position * 0.3 * confidence  # 高风险市大幅减仓
            else:
                position = base_position * confidence
        else:
            position = 0.0
        
        # 风险控制
        defender_risk = predictions.get('defender', 0)
        if defender_risk > 0.5:
            position *= (1 - defender_risk)  # 高风险时减仓
        
        signal = {
            'direction': direction,
            'confidence': confidence,
            'position_size': min(1.0, max(0.0, position)),  # 限制在0-1之间
            'defender_risk': defender_risk,
            'regime': regime,
            'score': final_score,
            'timestamp': datetime.now().isoformat()
        }
        
        return signal
    
    def update_gating_weights(self, regime: str, 
                             performance_feedback: Dict[str, float] = None):
        """根据性能反馈更新门控权重（在线学习）"""
        try:
            if performance_feedback:
                # 简单示例：根据各个专家的历史表现调整权重
                total_perf = sum(performance_feedback.values())
                if total_perf > 0:
                    for model_name in self.gating_weights[regime]:
                        if model_name in performance_feedback:
                            # 根据表现比例调整权重
                            self.gating_weights[regime][model_name] = \
                                performance_feedback[model_name] / total_perf
            
            logger.info(f"Updated gating weights for {regime}: {self.gating_weights[regime]}")
            
        except Exception as e:
            logger.error(f"Error updating gating weights: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            'loaded_models': list(self.models.keys()),
            'gating_weights': self.gating_weights,
            'feature_count': len(self.feature_columns),
            'metadata': {}
        }
        
        for model_name, metadata in self.metadata.items():
            info['metadata'][model_name] = {
                'description': metadata.get('description', ''),
                'training_date': metadata.get('training_date', ''),
                'metrics': metadata.get('metrics', {})
            }
        
        return info

def test_inference():
    """测试推理功能"""
    logger.info("Testing Model Inference...")
    
    # 创建推理引擎
    inference = ModelInference()
    
    # 生成测试特征
    np.random.seed(42)
    test_features = np.random.randn(1, len(inference.feature_columns))
    
    # 测试不同市场状态下的预测
    regimes = ['trending', 'ranging', 'high_vol', 'crash']
    
    for regime in regimes:
        logger.info(f"\nTesting regime: {regime}")
        result = inference.predict_with_moe(test_features, regime)
        
        if 'error' not in result:
            logger.info(f"  Final score: {result['final_score']:.3f}")
            logger.info(f"  Signal: {result['signal']['direction']} (confidence: {result['signal']['confidence']:.2%})")
            logger.info(f"  Defender risk: {result['predictions']['defender']:.2%}")
        else:
            logger.error(f"  Error: {result['error']}")
    
    # 显示模型信息
    info = inference.get_model_info()
    logger.info(f"\nModel info: {len(info['loaded_models'])} models loaded")
    
    return True

if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{message}")
    
    test_inference()