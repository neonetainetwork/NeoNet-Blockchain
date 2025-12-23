#!/usr/bin/env python3
"""
NeoNet Energy Provider (NNET)
Simple GUI for providing energy to NeoNet network

HOW IT WORKS:
- You DON'T mine or compute anything
- You just provide "energy" (stay connected)
- AI does all the work (training, protection, validation)
- You get rewarded for being online

Usage:
1. Enter your wallet address (neo1... or 0x...)
2. Click START
3. Stay connected and earn NNET!

To build .exe:
    pip install pyinstaller
    pyinstaller --onefile --noconsole --name NeoNetEnergy neonet_miner_gui.py
"""
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import asyncio
import aiohttp
import time
import os
import webbrowser
from typing import Optional

NEONET_SERVERS = [
    "https://neonet-ai-blockchain-neonetainetwork.replit.app",
    "https://neonetainetwork.com",
    "http://localhost:5000"
]

DEFAULT_SERVER = NEONET_SERVERS[0]

class NeoNetEnergyGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("NeoNet Energy Provider - NNET")
        self.root.geometry("500x700")
        self.root.resizable(False, False)
        
        self.root.configure(bg='#1a1a2e')
        
        self.is_providing = False
        self.session_id = None
        self.blocks_contributed = 0
        self.total_rewards = 0.0
        self.active_providers = 1
        self.current_reward_rate = 10.0
        self.energy_thread = None
        self.loop = None
        self.uptime_seconds = 0
        
        self.server_url = DEFAULT_SERVER
        
        self.create_widgets()
        
    def create_widgets(self):
        title_frame = tk.Frame(self.root, bg='#1a1a2e')
        title_frame.pack(pady=20)
        
        title = tk.Label(
            title_frame, 
            text="NeoNet Energy Provider",
            font=("Arial", 26, "bold"),
            fg='#00d9ff',
            bg='#1a1a2e'
        )
        title.pack()
        
        subtitle = tk.Label(
            title_frame,
            text="AI Does The Work - You Provide Energy",
            font=("Arial", 12),
            fg='#888888',
            bg='#1a1a2e'
        )
        subtitle.pack()
        
        info_frame = tk.Frame(self.root, bg='#2a2a4e', relief='flat')
        info_frame.pack(pady=10, padx=30, fill='x')
        
        info_text = tk.Label(
            info_frame,
            text="Your computer does NOTHING heavy.\n" +
                 "Just stay connected. AI handles:\n" +
                 "- Training models\n" +
                 "- Protecting network\n" +
                 "- Validating transactions\n" +
                 "You get rewarded for being online!",
            font=("Arial", 10),
            fg='#aaaaaa',
            bg='#2a2a4e',
            justify='left',
            padx=15,
            pady=10
        )
        info_text.pack()
        
        wallet_frame = tk.Frame(self.root, bg='#1a1a2e')
        wallet_frame.pack(pady=15, padx=30, fill='x')
        
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
        
        button_frame = tk.Frame(self.root, bg='#1a1a2e')
        button_frame.pack(pady=15)
        
        self.start_button = tk.Button(
            button_frame,
            text="PROVIDE ENERGY",
            font=("Arial", 18, "bold"),
            bg='#00d900',
            fg='white',
            activebackground='#00aa00',
            activeforeground='white',
            relief='flat',
            width=20,
            height=2,
            cursor='hand2',
            command=self.toggle_providing
        )
        self.start_button.pack()
        
        self.status_label = tk.Label(
            self.root,
            text="Offline",
            font=("Arial", 16, "bold"),
            fg='#ff4444',
            bg='#1a1a2e'
        )
        self.status_label.pack(pady=10)
        
        stats_frame = tk.Frame(self.root, bg='#2a2a4e', relief='flat')
        stats_frame.pack(pady=15, padx=30, fill='x')
        
        stats_title = tk.Label(
            stats_frame,
            text="Your Contribution",
            font=("Arial", 14, "bold"),
            fg='#00d9ff',
            bg='#2a2a4e'
        )
        stats_title.pack(pady=10)
        
        stats_inner = tk.Frame(stats_frame, bg='#2a2a4e')
        stats_inner.pack(padx=20, pady=10, fill='x')
        
        self.uptime_label = tk.Label(
            stats_inner,
            text="Uptime: 0:00:00",
            font=("Arial", 12),
            fg='#ffffff',
            bg='#2a2a4e'
        )
        self.uptime_label.pack(anchor='w', pady=2)
        
        self.blocks_label = tk.Label(
            stats_inner,
            text="Blocks Contributed: 0",
            font=("Arial", 12),
            fg='#ffffff',
            bg='#2a2a4e'
        )
        self.blocks_label.pack(anchor='w', pady=2)
        
        self.rewards_label = tk.Label(
            stats_inner,
            text="Total Earned: 0.0000 NNET",
            font=("Arial", 14, "bold"),
            fg='#00ff00',
            bg='#2a2a4e'
        )
        self.rewards_label.pack(anchor='w', pady=2)
        
        self.providers_label = tk.Label(
            stats_inner,
            text="Active Providers: 1",
            font=("Arial", 12),
            fg='#aaaaaa',
            bg='#2a2a4e'
        )
        self.providers_label.pack(anchor='w', pady=2)
        
        self.rate_label = tk.Label(
            stats_inner,
            text="Current Rate: 10.0 NNET/block",
            font=("Arial", 12),
            fg='#ffaa00',
            bg='#2a2a4e'
        )
        self.rate_label.pack(anchor='w', pady=2)
        
        self.ai_status_label = tk.Label(
            stats_inner,
            text="AI Status: Idle",
            font=("Arial", 10),
            fg='#888888',
            bg='#2a2a4e'
        )
        self.ai_status_label.pack(anchor='w', pady=5)
        
        network_frame = tk.Frame(self.root, bg='#1a1a2e')
        network_frame.pack(pady=10, padx=30, fill='x')
        
        network_info = tk.Label(
            network_frame,
            text="Network Speed: ~3,000 TPS | Block Time: ~3 sec",
            font=("Arial", 10),
            fg='#666666',
            bg='#1a1a2e',
            justify='center'
        )
        network_info.pack()
        
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
            text="v1.1.0 | NeoNet AI Network",
            font=("Arial", 9),
            fg='#444444',
            bg='#1a1a2e'
        )
        version_label.pack(side='bottom', pady=10)
    
    def toggle_providing(self):
        if self.is_providing:
            self.stop_providing()
        else:
            self.start_providing()
    
    def start_providing(self):
        wallet = self.wallet_entry.get().strip()
        
        if not wallet or wallet == "neo1..." or len(wallet) < 10:
            messagebox.showerror("Error", "Please enter a valid wallet address (neo1... or 0x...)")
            return
        
        self.is_providing = True
        self.uptime_seconds = 0
        self.start_button.config(text="DISCONNECT", bg='#ff4444')
        self.status_label.config(text="Providing Energy...", fg='#00ff00')
        
        self.energy_thread = threading.Thread(target=self.energy_loop, args=(wallet,), daemon=True)
        self.energy_thread.start()
    
    def stop_providing(self):
        self.is_providing = False
        self.start_button.config(text="PROVIDE ENERGY", bg='#00d900')
        self.status_label.config(text="Offline", fg='#ff4444')
        self.ai_status_label.config(text="AI Status: Idle")
    
    def energy_loop(self, wallet: str):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.loop.run_until_complete(self.async_energy_loop(wallet))
        except Exception as e:
            print(f"Energy error: {e}")
        finally:
            self.loop.close()
    
    async def async_energy_loop(self, wallet: str):
        contributor_id = wallet.lower()
        
        registered = await self.register_provider(contributor_id)
        if not registered:
            self.root.after(0, lambda: self.stop_providing())
            self.root.after(0, lambda: messagebox.showerror("Error", "Failed to connect to network"))
            return
        
        session_started = await self.start_session(contributor_id)
        if not session_started:
            self.root.after(0, lambda: self.stop_providing())
            self.root.after(0, lambda: messagebox.showerror("Error", "Failed to start session"))
            return
        
        ai_activities = [
            "Training fraud detection model...",
            "Validating transactions...",
            "Protecting network...",
            "Optimizing gas fees...",
            "Running federated learning...",
            "Processing blocks...",
            "Updating state trie...",
            "Verifying signatures..."
        ]
        activity_index = 0
        
        while self.is_providing:
            try:
                self.uptime_seconds += 3
                
                reward = await self.send_heartbeat(contributor_id)
                
                if reward > 0:
                    self.blocks_contributed += 1
                    self.total_rewards += reward
                
                activity_index = (activity_index + 1) % len(ai_activities)
                self.root.after(0, lambda a=ai_activities[activity_index]: 
                    self.ai_status_label.config(text=f"AI: {a}"))
                
                self.root.after(0, self.update_stats)
                
                await asyncio.sleep(3)
                
            except Exception as e:
                print(f"Heartbeat error: {e}")
                await asyncio.sleep(5)
    
    async def register_provider(self, contributor_id: str) -> bool:
        async with aiohttp.ClientSession() as session:
            try:
                payload = {
                    "contributor_id": contributor_id,
                    "cpu_cores": 1,
                    "gpu_memory_mb": 0,
                    "gpu_model": "Energy Only"
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
    
    async def send_heartbeat(self, contributor_id: str) -> float:
        async with aiohttp.ClientSession() as session:
            try:
                payload = {
                    "contributor_id": contributor_id,
                    "session_id": self.session_id
                }
                async with session.post(
                    f"{self.server_url}/ai-energy/heartbeat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.active_providers = data.get("active_providers", 1)
                        self.current_reward_rate = data.get("reward_per_block", 10.0)
                        return data.get("reward", 0.0)
                    return 0.0
            except Exception as e:
                print(f"Heartbeat error: {e}")
                return 0.0
    
    def update_stats(self):
        hours = self.uptime_seconds // 3600
        minutes = (self.uptime_seconds % 3600) // 60
        seconds = self.uptime_seconds % 60
        self.uptime_label.config(text=f"Uptime: {hours}:{minutes:02d}:{seconds:02d}")
        self.blocks_label.config(text=f"Blocks Contributed: {self.blocks_contributed}")
        self.rewards_label.config(text=f"Total Earned: {self.total_rewards:.4f} NNET")
        self.providers_label.config(text=f"Active Providers: {self.active_providers}")
        self.rate_label.config(text=f"Current Rate: {self.current_reward_rate:.4f} NNET/block")

def main():
    root = tk.Tk()
    app = NeoNetEnergyGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
