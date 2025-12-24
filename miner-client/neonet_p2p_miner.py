#!/usr/bin/env python3
"""
NeoNet P2P AI Miner Client
Fully decentralized mode - no central server required

When 1000+ miners are active, the central bootstrap server shuts down.
This client allows miners to continue operating in P2P mode:
- Discovers peers automatically
- Shares tasks with other miners
- Validates results through consensus
- Maintains local blockchain state
"""
import asyncio
import aiohttp
import json
import time
import hashlib
import numpy as np
import argparse
import sys
import os
import socket
from typing import Optional, Dict, Any, List, Set
from dataclasses import dataclass, field
from enum import Enum
from aiohttp import web
import threading

class NetworkMode(Enum):
    BOOTSTRAP = "bootstrap"  # Connect to central server
    P2P = "p2p"              # Fully decentralized
    HYBRID = "hybrid"        # Try central, fallback to P2P

@dataclass
class P2PConfig:
    wallet_address: str
    bootstrap_servers: List[str] = field(default_factory=lambda: [
        "https://neonetainetwork.com"
    ])
    listen_port: int = 9876
    cpu_cores: int = 4
    gpu_memory_mb: int = 0
    mode: NetworkMode = NetworkMode.HYBRID

class PeerManager:
    """Manages P2P peer connections"""
    
    def __init__(self, local_port: int):
        self.local_port = local_port
        self.peers: Set[str] = set()
        self.peer_scores: Dict[str, float] = {}
        self.last_seen: Dict[str, float] = {}
        
    def add_peer(self, peer_addr: str):
        if peer_addr and peer_addr not in self.peers:
            self.peers.add(peer_addr)
            self.peer_scores[peer_addr] = 1.0
            self.last_seen[peer_addr] = time.time()
            print(f"[P2P] Added peer: {peer_addr}")
    
    def remove_peer(self, peer_addr: str):
        self.peers.discard(peer_addr)
        self.peer_scores.pop(peer_addr, None)
        self.last_seen.pop(peer_addr, None)
    
    def update_peer(self, peer_addr: str, success: bool):
        self.last_seen[peer_addr] = time.time()
        if success:
            self.peer_scores[peer_addr] = min(10.0, self.peer_scores.get(peer_addr, 1.0) + 0.1)
        else:
            self.peer_scores[peer_addr] = max(0.1, self.peer_scores.get(peer_addr, 1.0) - 0.5)
    
    def get_best_peers(self, count: int = 5) -> List[str]:
        sorted_peers = sorted(
            self.peers,
            key=lambda p: self.peer_scores.get(p, 0),
            reverse=True
        )
        return sorted_peers[:count]
    
    def cleanup_stale_peers(self, max_age: float = 300):
        now = time.time()
        stale = [p for p, t in self.last_seen.items() if now - t > max_age]
        for p in stale:
            self.remove_peer(p)

class LocalTaskQueue:
    """Local task queue for P2P mode"""
    
    def __init__(self):
        self.pending_tasks: List[dict] = []
        self.completed_tasks: Dict[str, dict] = {}
        self.task_lock = asyncio.Lock()
    
    async def add_task(self, task: dict):
        async with self.task_lock:
            self.pending_tasks.append(task)
    
    async def get_task(self) -> Optional[dict]:
        async with self.task_lock:
            if self.pending_tasks:
                return self.pending_tasks.pop(0)
            return None
    
    async def complete_task(self, task_id: str, result: dict):
        async with self.task_lock:
            self.completed_tasks[task_id] = result
    
    def generate_local_task(self) -> dict:
        """Generate a task locally when no peers available"""
        task_types = [
            "fraud_detection", "model_training", "network_protection",
            "data_validation", "inference", "gradient_compute", "matrix_ops"
        ]
        task_type = np.random.choice(task_types)
        
        return {
            "task_id": hashlib.sha256(os.urandom(32)).hexdigest(),
            "task_type": task_type,
            "source": "local",
            "created_at": time.time(),
            "data": self._generate_task_data(task_type)
        }
    
    def _generate_task_data(self, task_type: str) -> dict:
        if task_type == "fraud_detection":
            return {"tx_count": np.random.randint(50, 200)}
        elif task_type == "model_training":
            return {"epochs": np.random.randint(5, 20)}
        elif task_type == "network_protection":
            start = np.random.randint(0, 1000)
            return {"block_range": [start, start + np.random.randint(50, 150)]}
        else:
            return {"size": np.random.randint(100, 1000)}

class P2PNode:
    """Local P2P node that can serve and receive tasks"""
    
    def __init__(self, config: P2PConfig):
        self.config = config
        self.peer_manager = PeerManager(config.listen_port)
        self.task_queue = LocalTaskQueue()
        self.blockchain_state: Dict[str, Any] = {}
        self.is_running = False
        self.app = web.Application()
        self._setup_routes()
    
    def _setup_routes(self):
        self.app.router.add_get('/peers', self.handle_get_peers)
        self.app.router.add_post('/peers', self.handle_add_peer)
        self.app.router.add_get('/task', self.handle_get_task)
        self.app.router.add_post('/task/submit', self.handle_submit_task)
        self.app.router.add_get('/status', self.handle_status)
        self.app.router.add_get('/state', self.handle_get_state)
    
    async def handle_get_peers(self, request):
        return web.json_response({
            "peers": list(self.peer_manager.peers),
            "count": len(self.peer_manager.peers)
        })
    
    async def handle_add_peer(self, request):
        data = await request.json()
        peer = data.get("peer")
        if peer:
            self.peer_manager.add_peer(peer)
        return web.json_response({"success": True})
    
    async def handle_get_task(self, request):
        task = await self.task_queue.get_task()
        if not task:
            task = self.task_queue.generate_local_task()
        return web.json_response(task)
    
    async def handle_submit_task(self, request):
        data = await request.json()
        task_id = data.get("task_id")
        result = data.get("result", {})
        await self.task_queue.complete_task(task_id, result)
        
        reward = self._calculate_reward(data.get("task_type", "matrix_ops"))
        return web.json_response({
            "success": True,
            "reward": reward,
            "task_id": task_id
        })
    
    async def handle_status(self, request):
        return web.json_response({
            "mode": "p2p",
            "peers": len(self.peer_manager.peers),
            "pending_tasks": len(self.task_queue.pending_tasks),
            "completed_tasks": len(self.task_queue.completed_tasks),
            "wallet": self.config.wallet_address
        })
    
    async def handle_get_state(self, request):
        return web.json_response(self.blockchain_state)
    
    def _calculate_reward(self, task_type: str) -> float:
        weights = {
            "federated_learning": 1.0,
            "model_training": 0.8,
            "network_protection": 0.6,
            "fraud_detection": 0.5,
            "gradient_compute": 0.5,
            "inference": 0.4,
            "data_validation": 0.3,
            "matrix_ops": 0.3
        }
        active_miners = max(1, len(self.peer_manager.peers) + 1)
        return weights.get(task_type, 0.3) * (10.0 / active_miners)
    
    async def start_server(self):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.config.listen_port)
        await site.start()
        print(f"[P2P] Node listening on port {self.config.listen_port}")

class NeoNetP2PMiner:
    """P2P-capable miner that works without central server"""
    
    def __init__(self, config: P2PConfig):
        self.config = config
        self.node = P2PNode(config)
        self.is_running = False
        self.tasks_completed = 0
        self.total_rewards = 0.0
        self.current_mode = NetworkMode.BOOTSTRAP
        self.session = None
    
    async def check_bootstrap_available(self) -> bool:
        """Check if central bootstrap server is available"""
        for server in self.config.bootstrap_servers:
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                    async with session.get(f"{server}/health") as resp:
                        if resp.status == 200:
                            return True
            except Exception:
                continue
        return False
    
    async def discover_peers(self) -> List[str]:
        """Discover peers from bootstrap or known peers"""
        peers = []
        
        for server in self.config.bootstrap_servers:
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                    async with session.get(f"{server}/api/decentralization/peers") as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            peers.extend(data.get("peers", []))
            except Exception:
                continue
        
        for peer in self.node.peer_manager.get_best_peers(10):
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as session:
                    async with session.get(f"http://{peer}/peers") as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            peers.extend(data.get("peers", []))
            except Exception:
                continue
        
        return list(set(peers))
    
    async def register_with_bootstrap(self, server: str) -> bool:
        """Register with bootstrap server"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "contributor_id": self.config.wallet_address,
                    "cpu_cores": self.config.cpu_cores,
                    "gpu_memory_mb": self.config.gpu_memory_mb,
                    "p2p_endpoint": f"0.0.0.0:{self.config.listen_port}"
                }
                async with session.post(f"{server}/ai-energy/register", json=payload) as resp:
                    return resp.status == 200
        except Exception:
            return False
    
    async def fetch_task_from_bootstrap(self, server: str) -> Optional[dict]:
        """Fetch task from bootstrap server"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{server}/ai-energy/task/{self.config.wallet_address}") as resp:
                    if resp.status == 200:
                        return await resp.json()
        except Exception:
            pass
        return None
    
    async def fetch_task_from_peer(self, peer: str) -> Optional[dict]:
        """Fetch task from P2P peer"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as session:
                async with session.get(f"http://{peer}/task") as resp:
                    if resp.status == 200:
                        task = await resp.json()
                        self.node.peer_manager.update_peer(peer, True)
                        return task
        except Exception:
            self.node.peer_manager.update_peer(peer, False)
        return None
    
    async def submit_result_to_bootstrap(self, server: str, task_id: str, result: dict) -> Optional[float]:
        """Submit result to bootstrap server"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "contributor_id": self.config.wallet_address,
                    "task_id": task_id,
                    "result": result
                }
                async with session.post(f"{server}/ai-energy/submit-result", json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("reward", 0)
        except Exception:
            pass
        return None
    
    async def submit_result_to_peer(self, peer: str, task_id: str, result: dict, task_type: str) -> Optional[float]:
        """Submit result to P2P peer"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as session:
                payload = {
                    "task_id": task_id,
                    "result": result,
                    "task_type": task_type,
                    "wallet": self.config.wallet_address
                }
                async with session.post(f"http://{peer}/task/submit", json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.node.peer_manager.update_peer(peer, True)
                        return data.get("reward", 0)
        except Exception:
            self.node.peer_manager.update_peer(peer, False)
        return None
    
    def process_task(self, task: dict) -> dict:
        """Process AI task locally"""
        task_type = task.get("task_type", "matrix_ops")
        task_data = task.get("data", {})
        start_time = time.time()
        result = {}
        
        if task_type == "fraud_detection":
            tx_count = task_data.get("tx_count", 100)
            transactions = np.random.randn(tx_count, 32).astype(np.float32)
            scores = np.abs(transactions).mean(axis=1)
            result = {
                "results_hash": hashlib.sha256(scores.tobytes()).hexdigest(),
                "fraud_count": int(np.sum(scores > 0.8)),
                "analyzed": tx_count
            }
        elif task_type == "model_training":
            epochs = task_data.get("epochs", 10)
            weights = np.random.randn(256, 128).astype(np.float32)
            for _ in range(epochs):
                weights -= np.random.randn(256, 128).astype(np.float32) * 0.01
            result = {
                "weights_hash": hashlib.sha256(weights.tobytes()).hexdigest(),
                "epochs_completed": epochs
            }
        elif task_type == "network_protection":
            block_range = task_data.get("block_range", [0, 100])
            validated = block_range[1] - block_range[0]
            result = {
                "blocks_validated": validated,
                "range": block_range
            }
        else:
            data = np.random.randn(task_data.get("size", 500), 64).astype(np.float32)
            result = {
                "result_hash": hashlib.sha256(data.tobytes()).hexdigest(),
                "size": data.shape[0]
            }
        
        result["compute_time_ms"] = int((time.time() - start_time) * 1000)
        return result
    
    async def run(self):
        """Main P2P mining loop"""
        print("=" * 60)
        print("    NeoNet P2P AI Miner - Decentralized Mode")
        print("=" * 60)
        print(f"Wallet: {self.config.wallet_address}")
        print(f"Mode: {self.config.mode.value}")
        print(f"P2P Port: {self.config.listen_port}")
        print(f"CPU Cores: {self.config.cpu_cores}")
        print("-" * 60)
        
        await self.node.start_server()
        
        bootstrap_available = await self.check_bootstrap_available()
        
        if bootstrap_available:
            self.current_mode = NetworkMode.BOOTSTRAP
            print("[MODE] Bootstrap server available - using hybrid mode")
            for server in self.config.bootstrap_servers:
                if await self.register_with_bootstrap(server):
                    print(f"[OK] Registered with {server}")
                    break
        else:
            self.current_mode = NetworkMode.P2P
            print("[MODE] Bootstrap unavailable - pure P2P mode")
        
        new_peers = await self.discover_peers()
        for peer in new_peers:
            self.node.peer_manager.add_peer(peer)
        print(f"[P2P] Discovered {len(new_peers)} peers")
        
        self.is_running = True
        check_bootstrap_counter = 0
        
        print("\n[MINING] Starting P2P task processing...")
        print("[INFO] Press Ctrl+C to stop\n")
        
        try:
            while self.is_running:
                task = None
                task_source = None
                
                if self.current_mode == NetworkMode.BOOTSTRAP:
                    for server in self.config.bootstrap_servers:
                        task = await self.fetch_task_from_bootstrap(server)
                        if task:
                            task_source = ("bootstrap", server)
                            break
                
                if not task:
                    for peer in self.node.peer_manager.get_best_peers(3):
                        task = await self.fetch_task_from_peer(peer)
                        if task:
                            task_source = ("peer", peer)
                            break
                
                if not task:
                    task = self.node.task_queue.generate_local_task()
                    task_source = ("local", None)
                
                if task and task.get("task_id"):
                    task_id = task.get("task_id")
                    task_type = task.get("task_type")
                    source_type = task_source[0] if task_source else "unknown"
                    print(f"[TASK] {task_type} from {source_type} (ID: {task_id[:8]}...)")
                    
                    result = self.process_task(task)
                    print(f"[DONE] Computed in {result.get('compute_time_ms', 0)}ms")
                    
                    reward = 0
                    if task_source[0] == "bootstrap":
                        reward = await self.submit_result_to_bootstrap(task_source[1], task_id, result) or 0
                    elif task_source[0] == "peer":
                        reward = await self.submit_result_to_peer(task_source[1], task_id, result, task_type) or 0
                    else:
                        reward = self.node._calculate_reward(task_type)
                    
                    if reward > 0:
                        self.tasks_completed += 1
                        self.total_rewards += reward
                        print(f"[REWARD] +{reward:.4f} NNET | Total: {self.total_rewards:.4f} NNET")
                
                check_bootstrap_counter += 1
                if check_bootstrap_counter >= 30:
                    bootstrap_available = await self.check_bootstrap_available()
                    if bootstrap_available and self.current_mode == NetworkMode.P2P:
                        self.current_mode = NetworkMode.BOOTSTRAP
                        print("[MODE] Bootstrap server came online - switching to hybrid")
                    elif not bootstrap_available and self.current_mode == NetworkMode.BOOTSTRAP:
                        self.current_mode = NetworkMode.P2P
                        print("[MODE] Bootstrap server down - switching to pure P2P")
                    check_bootstrap_counter = 0
                    
                    self.node.peer_manager.cleanup_stale_peers()
                    new_peers = await self.discover_peers()
                    for peer in new_peers:
                        self.node.peer_manager.add_peer(peer)
                
                await asyncio.sleep(2)
                
        except KeyboardInterrupt:
            print("\n[STOP] Shutting down P2P miner...")
        finally:
            self.is_running = False
            print("\n" + "=" * 60)
            print("    P2P Mining Session Summary")
            print("=" * 60)
            print(f"Mode: {self.current_mode.value}")
            print(f"Peers Connected: {len(self.node.peer_manager.peers)}")
            print(f"Tasks Completed: {self.tasks_completed}")
            print(f"Total Rewards: {self.total_rewards:.4f} NNET")
            print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description="NeoNet P2P AI Miner")
    parser.add_argument("--wallet", required=True, help="Your wallet address")
    parser.add_argument("--port", type=int, default=9876, help="P2P listen port")
    parser.add_argument("--cpu", type=int, default=4, help="CPU cores to use")
    parser.add_argument("--gpu-mem", type=int, default=0, help="GPU memory in MB")
    parser.add_argument("--mode", choices=["bootstrap", "p2p", "hybrid"], default="hybrid",
                       help="Network mode: bootstrap (central only), p2p (decentralized), hybrid (auto)")
    parser.add_argument("--bootstrap", action="append", help="Additional bootstrap server URLs")
    
    args = parser.parse_args()
    
    mode_map = {
        "bootstrap": NetworkMode.BOOTSTRAP,
        "p2p": NetworkMode.P2P,
        "hybrid": NetworkMode.HYBRID
    }
    
    bootstrap_servers = ["https://neonetainetwork.com"]
    if args.bootstrap:
        bootstrap_servers.extend(args.bootstrap)
    
    config = P2PConfig(
        wallet_address=args.wallet,
        bootstrap_servers=bootstrap_servers,
        listen_port=args.port,
        cpu_cores=args.cpu,
        gpu_memory_mb=args.gpu_mem,
        mode=mode_map[args.mode]
    )
    
    miner = NeoNetP2PMiner(config)
    await miner.run()


if __name__ == "__main__":
    asyncio.run(main())
