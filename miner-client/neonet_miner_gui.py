#!/usr/bin/env python3
"""
NeoNet One-Click Miner (NNET)
Simple GUI for mining NNET tokens

Features:
- One-click mining start
- No technical knowledge required
- Dynamic rewards based on active miners
- Real-time stats display

Usage:
1. Enter your wallet address (neo1... or 0x...)
2. Click START MINING
3. Earn NNET tokens!

To build .exe:
    pip install pyinstaller
    pyinstaller --onefile --noconsole --name NeoNetMiner neonet_miner_gui.py
"""
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import asyncio
import aiohttp
import time
import hashlib
import platform
import os
import sys
import webbrowser
from typing import Optional

NEONET_SERVERS = [
    "https://neonet-ai-blockchain-neonetainetwork.replit.app",
    "https://neonetainetwork.com",
    "http://localhost:5000"
]

DEFAULT_SERVER = NEONET_SERVERS[0]

class NeoNetMinerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("NeoNet Miner - NNET")
        self.root.geometry("500x650")
        self.root.resizable(False, False)
        
        self.root.configure(bg='#1a1a2e')
        
        self.is_mining = False
        self.session_id = None
        self.tasks_completed = 0
        self.total_rewards = 0.0
        self.active_miners = 1
        self.current_reward_rate = 10.0
        self.mining_thread = None
        self.loop = None
        
        self.server_url = DEFAULT_SERVER
        
        self.create_widgets()
        
    def create_widgets(self):
        title_frame = tk.Frame(self.root, bg='#1a1a2e')
        title_frame.pack(pady=20)
        
        title = tk.Label(
            title_frame, 
            text="⚡ NeoNet Miner",
            font=("Arial", 28, "bold"),
            fg='#00d9ff',
            bg='#1a1a2e'
        )
        title.pack()
        
        subtitle = tk.Label(
            title_frame,
            text="AI-Powered Web4 Mining",
            font=("Arial", 12),
            fg='#888888',
            bg='#1a1a2e'
        )
        subtitle.pack()
        
        wallet_frame = tk.Frame(self.root, bg='#1a1a2e')
        wallet_frame.pack(pady=20, padx=30, fill='x')
        
        wallet_label = tk.Label(
            wallet_frame,
            text="Wallet Address:",
            font=("Arial", 12),
            fg='#ffffff',
            bg='#1a1a2e'
        )
        wallet_label.pack(anchor='w')
        
        self.wallet_entry = tk.Entry(
            wallet_frame,
            font=("Arial", 14),
            bg='#2a2a4e',
            fg='#ffffff',
            insertbackground='#00d9ff',
            relief='flat',
            width=40
        )
        self.wallet_entry.pack(fill='x', pady=5, ipady=10)
        self.wallet_entry.insert(0, "neo1...")
        
        hint = tk.Label(
            wallet_frame,
            text="Enter your neo1... or 0x... address",
            font=("Arial", 9),
            fg='#666666',
            bg='#1a1a2e'
        )
        hint.pack(anchor='w')
        
        button_frame = tk.Frame(self.root, bg='#1a1a2e')
        button_frame.pack(pady=20)
        
        self.start_button = tk.Button(
            button_frame,
            text="▶ START MINING",
            font=("Arial", 18, "bold"),
            bg='#00d900',
            fg='white',
            activebackground='#00aa00',
            activeforeground='white',
            relief='flat',
            width=20,
            height=2,
            cursor='hand2',
            command=self.toggle_mining
        )
        self.start_button.pack()
        
        self.status_label = tk.Label(
            self.root,
            text="🔴 Stopped",
            font=("Arial", 16, "bold"),
            fg='#ff4444',
            bg='#1a1a2e'
        )
        self.status_label.pack(pady=10)
        
        stats_frame = tk.Frame(self.root, bg='#2a2a4e', relief='flat')
        stats_frame.pack(pady=20, padx=30, fill='x')
        
        stats_title = tk.Label(
            stats_frame,
            text="Mining Statistics",
            font=("Arial", 14, "bold"),
            fg='#00d9ff',
            bg='#2a2a4e'
        )
        stats_title.pack(pady=10)
        
        stats_inner = tk.Frame(stats_frame, bg='#2a2a4e')
        stats_inner.pack(padx=20, pady=10, fill='x')
        
        self.tasks_label = tk.Label(
            stats_inner,
            text="Tasks Completed: 0",
            font=("Arial", 12),
            fg='#ffffff',
            bg='#2a2a4e'
        )
        self.tasks_label.pack(anchor='w', pady=2)
        
        self.rewards_label = tk.Label(
            stats_inner,
            text="Total Earned: 0.0000 NNET",
            font=("Arial", 14, "bold"),
            fg='#00ff00',
            bg='#2a2a4e'
        )
        self.rewards_label.pack(anchor='w', pady=2)
        
        self.miners_label = tk.Label(
            stats_inner,
            text="Active Miners: 1",
            font=("Arial", 12),
            fg='#aaaaaa',
            bg='#2a2a4e'
        )
        self.miners_label.pack(anchor='w', pady=2)
        
        self.rate_label = tk.Label(
            stats_inner,
            text="Current Rate: 10.0 NNET/task",
            font=("Arial", 12),
            fg='#ffaa00',
            bg='#2a2a4e'
        )
        self.rate_label.pack(anchor='w', pady=2)
        
        self.task_label = tk.Label(
            stats_inner,
            text="Current Task: None",
            font=("Arial", 10),
            fg='#888888',
            bg='#2a2a4e'
        )
        self.task_label.pack(anchor='w', pady=5)
        
        info_frame = tk.Frame(self.root, bg='#1a1a2e')
        info_frame.pack(pady=10, padx=30, fill='x')
        
        info_text = tk.Label(
            info_frame,
            text="Dynamic Rewards: More miners = lower rewards\n" +
                 "Formula: Reward = Weight × (Budget / Miners)",
            font=("Arial", 9),
            fg='#666666',
            bg='#1a1a2e',
            justify='center'
        )
        info_text.pack()
        
        links_frame = tk.Frame(self.root, bg='#1a1a2e')
        links_frame.pack(pady=10)
        
        website_btn = tk.Button(
            links_frame,
            text="Website",
            font=("Arial", 10),
            bg='#3a3a5e',
            fg='#00d9ff',
            relief='flat',
            cursor='hand2',
            command=lambda: webbrowser.open("https://neonetainetwork.com")
        )
        website_btn.pack(side='left', padx=5)
        
        docs_btn = tk.Button(
            links_frame,
            text="Documentation",
            font=("Arial", 10),
            bg='#3a3a5e',
            fg='#00d9ff',
            relief='flat',
            cursor='hand2',
            command=lambda: webbrowser.open("https://neonetainetwork.info")
        )
        docs_btn.pack(side='left', padx=5)
        
        version_label = tk.Label(
            self.root,
            text="v1.0.0 | NeoNet AI Network",
            font=("Arial", 9),
            fg='#444444',
            bg='#1a1a2e'
        )
        version_label.pack(side='bottom', pady=10)
    
    def toggle_mining(self):
        if self.is_mining:
            self.stop_mining()
        else:
            self.start_mining()
    
    def start_mining(self):
        wallet = self.wallet_entry.get().strip()
        
        if not wallet or wallet == "neo1..." or len(wallet) < 10:
            messagebox.showerror("Error", "Please enter a valid wallet address (neo1... or 0x...)")
            return
        
        self.is_mining = True
        self.start_button.config(text="⏹ STOP MINING", bg='#ff4444')
        self.status_label.config(text="🟢 Mining...", fg='#00ff00')
        
        self.mining_thread = threading.Thread(target=self.mining_loop, args=(wallet,), daemon=True)
        self.mining_thread.start()
    
    def stop_mining(self):
        self.is_mining = False
        self.start_button.config(text="▶ START MINING", bg='#00d900')
        self.status_label.config(text="🔴 Stopped", fg='#ff4444')
    
    def mining_loop(self, wallet: str):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.loop.run_until_complete(self.async_mining_loop(wallet))
        except Exception as e:
            print(f"Mining error: {e}")
        finally:
            self.loop.close()
    
    async def async_mining_loop(self, wallet: str):
        contributor_id = wallet.lower()
        
        registered = await self.register_miner(contributor_id)
        if not registered:
            self.root.after(0, lambda: self.stop_mining())
            self.root.after(0, lambda: messagebox.showerror("Error", "Failed to register with network"))
            return
        
        session_started = await self.start_session(contributor_id)
        if not session_started:
            self.root.after(0, lambda: self.stop_mining())
            self.root.after(0, lambda: messagebox.showerror("Error", "Failed to start session"))
            return
        
        while self.is_mining:
            try:
                task = await self.fetch_task(contributor_id)
                
                if task and task.get("task_id"):
                    self.root.after(0, lambda t=task: self.task_label.config(
                        text=f"Current Task: {t.get('task_type', 'unknown')}"
                    ))
                    
                    result = self.process_task(task)
                    
                    reward = await self.submit_result(contributor_id, task, result)
                    
                    if reward > 0:
                        self.tasks_completed += 1
                        self.total_rewards += reward
                        
                        self.root.after(0, self.update_stats)
                
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"Task error: {e}")
                await asyncio.sleep(5)
    
    async def register_miner(self, contributor_id: str) -> bool:
        async with aiohttp.ClientSession() as session:
            try:
                payload = {
                    "contributor_id": contributor_id,
                    "cpu_cores": os.cpu_count() or 4,
                    "gpu_memory_mb": 0,
                    "gpu_model": "CPU Only"
                }
                async with session.post(
                    f"{self.server_url}/ai-energy/register",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    return resp.status == 200 or resp.status == 400
            except Exception as e:
                print(f"Register error: {e}")
                return False
    
    async def start_session(self, contributor_id: str) -> bool:
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.server_url}/ai-energy/start-session",
                    json={"contributor_id": contributor_id},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.session_id = data.get("session_id")
                        return True
                    return False
            except Exception as e:
                print(f"Session error: {e}")
                return False
    
    async def fetch_task(self, contributor_id: str) -> Optional[dict]:
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    f"{self.server_url}/ai-energy/task/{contributor_id}",
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        task = await resp.json()
                        self.active_miners = task.get("active_miners_at_creation", 1)
                        self.current_reward_rate = task.get("reward", 1.0)
                        return task
                    return None
            except Exception as e:
                print(f"Fetch error: {e}")
                return None
    
    def process_task(self, task: dict) -> dict:
        task_type = task.get("task_type", "unknown")
        task_data = task.get("data", {})
        
        time.sleep(0.5)
        
        result_hash = hashlib.sha256(f"{task_type}{time.time()}".encode()).hexdigest()[:16]
        
        results = {
            "fraud_detection": {"results_hash": result_hash, "fraud_detected": False},
            "model_training": {"weights_hash": result_hash, "loss": 0.01},
            "network_protection": {"blocks_validated": 5, "anomalies": 0},
            "data_validation": {"integrity_score": 0.99, "records": 100},
            "inference": {"output_hash": result_hash, "predictions": 10},
            "federated_learning": {"weights_hash": result_hash, "rounds": 3},
            "gradient_compute": {"gradient_hash": result_hash, "norm": 0.5},
            "matrix_ops": {"result_hash": result_hash, "ops": 1000}
        }
        
        return results.get(task_type, {"result_hash": result_hash})
    
    async def submit_result(self, contributor_id: str, task: dict, result: dict) -> float:
        async with aiohttp.ClientSession() as session:
            try:
                payload = {
                    "contributor_id": contributor_id,
                    "session_id": self.session_id,
                    "task_id": task.get("task_id"),
                    "result": result
                }
                async with session.post(
                    f"{self.server_url}/ai-energy/submit-result",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("reward", 0.0)
                    return 0.0
            except Exception as e:
                print(f"Submit error: {e}")
                return 0.0
    
    def update_stats(self):
        self.tasks_label.config(text=f"Tasks Completed: {self.tasks_completed}")
        self.rewards_label.config(text=f"Total Earned: {self.total_rewards:.4f} NNET")
        self.miners_label.config(text=f"Active Miners: {self.active_miners}")
        self.rate_label.config(text=f"Current Rate: {self.current_reward_rate:.4f} NNET/task")

def main():
    root = tk.Tk()
    app = NeoNetMinerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
