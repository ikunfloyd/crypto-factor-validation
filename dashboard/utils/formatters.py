"""
格式化工具模块
提供各种数据格式化函数
"""

from datetime import datetime, timedelta
from typing import Union, Optional


def format_number(value: Union[int, float], decimals: int = 2) -> str:
    """格式化数字，添加千位分隔符"""
    if value is None:
        return "N/A"
    try:
        if isinstance(value, int) or value.is_integer():
            return f"{int(value):,}"
        else:
            return f"{value:,.{decimals}f}"
    except (AttributeError, ValueError):
        return str(value)


def format_percentage(value: Union[int, float], decimals: int = 2, show_sign: bool = True) -> str:
    """格式化百分比"""
    if value is None:
        return "N/A"
    try:
        sign = "+" if value > 0 and show_sign else ""
        return f"{sign}{value:.{decimals}f}%"
    except (AttributeError, ValueError, TypeError):
        return str(value)


def format_price(value: Union[int, float], decimals: int = 5) -> str:
    """格式化价格"""
    if value is None:
        return "N/A"
    try:
        return f"${value:.{decimals}f}"
    except (AttributeError, ValueError, TypeError):
        return str(value)


def format_crypto_amount(value: Union[int, float], symbol: str = "DOGE") -> str:
    """格式化加密货币数量"""
    if value is None:
        return "N/A"
    try:
        formatted = format_number(value, decimals=2)
        return f"{formatted} {symbol}"
    except Exception:
        return f"{value} {symbol}"


def format_bytes(bytes_value: Union[int, float]) -> str:
    """格式化字节大小"""
    if bytes_value is None:
        return "N/A"
    
    try:
        bytes_value = float(bytes_value)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if abs(bytes_value) < 1024.0:
                return f"{bytes_value:.2f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.2f} PB"
    except (ValueError, TypeError):
        return str(bytes_value)


def format_uptime(seconds: Union[int, float]) -> str:
    """格式化运行时长"""
    if seconds is None:
        return "N/A"
    
    try:
        seconds = int(seconds)
        days = seconds // 86400
        hours = (seconds % 86400) // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        
        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0 or not parts:
            parts.append(f"{secs}s")
        
        return " ".join(parts)
    except (ValueError, TypeError):
        return str(seconds)


def format_timestamp(ts: Union[datetime, str, int, float], fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """格式化时间戳"""
    if ts is None:
        return "N/A"
    
    try:
        if isinstance(ts, datetime):
            return ts.strftime(fmt)
        elif isinstance(ts, str):
            # 尝试解析字符串
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            return dt.strftime(fmt)
        elif isinstance(ts, (int, float)):
            # Unix时间戳
            dt = datetime.fromtimestamp(ts)
            return dt.strftime(fmt)
        else:
            return str(ts)
    except Exception:
        return str(ts)


def format_relative_time(dt: Union[datetime, str, int, float]) -> str:
    """格式化相对时间（例如：5分钟前）"""
    if dt is None:
        return "N/A"
    
    try:
        if isinstance(dt, str):
            dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
        elif isinstance(dt, (int, float)):
            dt = datetime.fromtimestamp(dt)
        
        now = datetime.now()
        diff = now - dt
        
        seconds = diff.total_seconds()
        
        if seconds < 60:
            return f"{int(seconds)}秒前"
        elif seconds < 3600:
            return f"{int(seconds / 60)}分钟前"
        elif seconds < 86400:
            return f"{int(seconds / 3600)}小时前"
        else:
            return f"{int(seconds / 86400)}天前"
    except Exception:
        return str(dt)


def format_pnl(value: Union[int, float], show_currency: bool = True) -> str:
    """格式化盈亏"""
    if value is None:
        return "N/A"
    
    try:
        sign = "+" if value > 0 else ""
        if show_currency:
            return f"{sign}${value:,.2f}"
        else:
            return f"{sign}{value:,.2f}"
    except (ValueError, TypeError):
        return str(value)


def format_rate(value: Union[int, float], per_second: bool = False) -> str:
    """格式化速率"""
    if value is None:
        return "N/A"
    
    try:
        formatted = format_number(value, decimals=0)
        unit = "条/秒" if per_second else "条/分"
        return f"{formatted} {unit}"
    except Exception:
        return str(value)


def format_latency(ms: Union[int, float]) -> str:
    """格式化延迟"""
    if ms is None:
        return "N/A"
    
    try:
        if ms < 1000:
            return f"{ms:.0f} ms"
        else:
            return f"{ms / 1000:.2f} s"
    except (ValueError, TypeError):
        return str(ms)


def format_status_emoji(status: str) -> str:
    """根据状态返回emoji"""
    status_map = {
        'running': '🟢',
        'active': '🟢',
        'healthy': '🟢',
        'stopped': '🔴',
        'error': '🔴',
        'failed': '🔴',
        'warning': '🟡',
        'degraded': '🟡',
        'unknown': '⚪',
        'pending': '🔵',
        'starting': '🔵'
    }
    return status_map.get(status.lower() if isinstance(status, str) else '', '⚪')


def format_level_color(level: str) -> str:
    """根据日志级别返回颜色"""
    level_map = {
        'DEBUG': 'gray',
        'INFO': 'blue',
        'WARNING': 'orange',
        'ERROR': 'red',
        'CRITICAL': 'red'
    }
    return level_map.get(level.upper() if isinstance(level, str) else '', 'gray')


def format_signal(signal: str) -> str:
    """格式化交易信号"""
    signal_map = {
        'LONG': '📈 LONG',
        'SHORT': '📉 SHORT',
        'FLAT': '➖ FLAT',
        'HOLD': '⏸️ HOLD'
    }
    return signal_map.get(signal.upper() if isinstance(signal, str) else '', signal)


def format_regime(regime: str) -> str:
    """格式化市场regime"""
    regime_map = {
        'trending_up': '📈 上涨趋势',
        'trending_down': '📉 下跌趋势',
        'ranging': '↔️ 震荡',
        'volatile': '⚡ 高波动',
        'quiet': '😴 低波动'
    }
    return regime_map.get(regime.lower() if isinstance(regime, str) else '', regime)


def truncate_string(s: str, max_length: int = 50, suffix: str = "...") -> str:
    """截断字符串"""
    if s is None:
        return ""
    
    s = str(s)
    if len(s) <= max_length:
        return s
    else:
        return s[:max_length - len(suffix)] + suffix


def format_dict_to_table(data: dict, key_name: str = "指标", value_name: str = "值") -> str:
    """将字典格式化为markdown表格"""
    if not data:
        return f"| {key_name} | {value_name} |\n|---|---|\n| 无数据 | - |"
    
    lines = [f"| {key_name} | {value_name} |", "|---|---|"]
    for key, value in data.items():
        lines.append(f"| {key} | {value} |")
    
    return "\n".join(lines)


def safe_divide(numerator: Union[int, float], denominator: Union[int, float], 
                default: float = 0.0) -> float:
    """安全除法，避免除零错误"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default


def color_by_value(value: Union[int, float], 
                   positive_color: str = "green",
                   negative_color: str = "red",
                   neutral_color: str = "gray") -> str:
    """根据数值正负返回颜色"""
    try:
        if value > 0:
            return positive_color
        elif value < 0:
            return negative_color
        else:
            return neutral_color
    except (TypeError, ValueError):
        return neutral_color
