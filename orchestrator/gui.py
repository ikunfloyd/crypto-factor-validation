#!/usr/bin/env python3
"""Orchestrator图形界面"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, Any
from datetime import datetime
from loguru import logger


class OrchestratorGUI:
    """Orchestrator图形界面控制器"""
    
    def __init__(self, orchestrator=None):
        """
        初始化GUI
        
        Args:
            orchestrator: Orchestrator实例（可选）
        """
        self.orchestrator = orchestrator
        self.root = None
        self.status_labels = {}
        self.running = False
        
    def create_window(self):
        """创建主窗口"""
        self.root = tk.Tk()
        self.root.title("DOGE-USDT Trading System - Orchestrator")
        self.root.geometry("800x600")
        
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 标题
        title_label = ttk.Label(
            main_frame, 
            text="量化交易系统控制面板",
            font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # 系统状态区域
        status_frame = ttk.LabelFrame(main_frame, text="系统状态", padding="10")
        status_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # 进程状态
        processes = ['io_gateway', 'quant_core', 'execution_guard']
        for i, proc_name in enumerate(processes):
            label = ttk.Label(status_frame, text=f"{proc_name}:")
            label.grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            
            status_label = ttk.Label(status_frame, text="未运行", foreground="red")
            status_label.grid(row=i, column=1, sticky=tk.W, padx=5, pady=2)
            self.status_labels[proc_name] = status_label
        
        # 控制按钮区域
        control_frame = ttk.LabelFrame(main_frame, text="系统控制", padding="10")
        control_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # 启动按钮
        start_btn = ttk.Button(
            control_frame,
            text="启动系统",
            command=self.start_system
        )
        start_btn.grid(row=0, column=0, padx=5, pady=5)
        
        # 停止按钮
        stop_btn = ttk.Button(
            control_frame,
            text="停止系统",
            command=self.stop_system
        )
        stop_btn.grid(row=0, column=1, padx=5, pady=5)
        
        # 创建检查点按钮
        checkpoint_btn = ttk.Button(
            control_frame,
            text="创建检查点",
            command=self.create_checkpoint
        )
        checkpoint_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # 日志区域
        log_frame = ttk.LabelFrame(main_frame, text="系统日志", padding="10")
        log_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        # 日志文本框
        self.log_text = tk.Text(log_frame, height=15, width=80)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 滚动条
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        logger.info("GUI window created")
    
    def start_system(self):
        """启动系统"""
        try:
            if self.orchestrator:
                self.log_message("正在启动系统...")
                # 这里应该调用orchestrator的启动方法
                # 由于orchestrator.run()是异步的，需要特殊处理
                self.log_message("系统启动功能需要配合Orchestrator实例")
            else:
                self.log_message("错误: 未连接到Orchestrator实例")
                messagebox.showwarning("警告", "未连接到Orchestrator实例")
        except Exception as e:
            self.log_message(f"启动失败: {e}")
            logger.error(f"Failed to start system: {e}")
    
    def stop_system(self):
        """停止系统"""
        try:
            if self.orchestrator:
                response = messagebox.askyesno("确认", "确定要停止系统吗？")
                if response:
                    self.log_message("正在停止系统...")
                    self.orchestrator.shutdown()
                    self.update_process_status()
            else:
                self.log_message("错误: 未连接到Orchestrator实例")
        except Exception as e:
            self.log_message(f"停止失败: {e}")
            logger.error(f"Failed to stop system: {e}")
    
    def create_checkpoint(self):
        """创建检查点"""
        try:
            if self.orchestrator:
                self.log_message("正在创建检查点...")
                checkpoint_path = self.orchestrator.create_checkpoint()
                self.log_message(f"检查点已创建: {checkpoint_path}")
            else:
                self.log_message("错误: 未连接到Orchestrator实例")
        except Exception as e:
            self.log_message(f"创建检查点失败: {e}")
            logger.error(f"Failed to create checkpoint: {e}")
    
    def update_process_status(self):
        """更新进程状态显示"""
        if not self.orchestrator:
            return
        
        try:
            for proc_name, status_label in self.status_labels.items():
                if proc_name in self.orchestrator.processes:
                    proc = self.orchestrator.processes[proc_name]
                    if proc.is_alive():
                        status_label.config(text=f"运行中 (PID: {proc.pid})", foreground="green")
                    else:
                        status_label.config(text="已停止", foreground="red")
                else:
                    status_label.config(text="未启动", foreground="gray")
        except Exception as e:
            logger.error(f"Failed to update process status: {e}")
    
    def log_message(self, message: str):
        """在日志区域显示消息"""
        if self.log_text:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.log_text.see(tk.END)  # 自动滚动到底部
    
    def run(self):
        """运行GUI"""
        if not self.root:
            self.create_window()
        
        # 定期更新状态
        def update_status():
            self.update_process_status()
            if self.running:
                self.root.after(1000, update_status)  # 每秒更新一次
        
        self.running = True
        update_status()
        
        logger.info("Starting GUI main loop")
        self.log_message("GUI已启动")
        self.root.mainloop()
    
    def close(self):
        """关闭GUI"""
        self.running = False
        if self.root:
            self.root.quit()
            self.root.destroy()
        logger.info("GUI closed")


def test_gui():
    """测试GUI（独立运行）"""
    logger.info("Testing Orchestrator GUI...")
    
    # 创建GUI实例（不连接orchestrator）
    gui = OrchestratorGUI()
    gui.log_message("这是一个测试GUI窗口")
    gui.log_message("在实际使用中，需要传入Orchestrator实例")
    
    # 运行GUI
    gui.run()


if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    test_gui()
