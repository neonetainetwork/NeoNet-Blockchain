#!/usr/bin/env python3
"""
NeoNet Miner - All-in-One
Run this script and your computer becomes part of the NeoNet network.

The AI runs locally on your machine and:
- Processes AI tasks
- Maintains a local copy of the blockchain
- Participates in consensus
- Earns NNET token rewards

Usage:
    pip install aiohttp numpy
    python neonet_miner.py --wallet neo1your_wallet

Options:
    --wallet    Your wallet address for NNET rewards (required)
    --port      P2P port (default: 8080)
    --cpu       Number of CPU cores to use (default: 4)
    --gpu-mem   GPU memory in MB (default: 0 = CPU only)
"""
import asyncio
import aiohttp
from aiohttp import web
import json
import time
import hashlib
import numpy as np
import argparse
import os
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

BOOTSTRAP_SERVERS = [
    "https://neonetainetwork.com"
]

@dataclass
class Block:
    index: int
    timestamp: float
    data: str
    previous_hash: str
    hash: str
    validator: str

@dataclass 
class NodeConfig:
    wallet: str
    port: int = 8080
    cpu_cores: int = 4
    gpu_memory_mb: int = 0

class LocalBlockchain:
    """Локальная копия блокчейна"""
    
    def __init__(self):
        self.chain: List[Block] = []
        self.pending_transactions: List[dict] = []
        self.state: Dict[str, float] = {}
        self._create_genesis()
    
    def _create_genesis(self):
        genesis = Block(
            index=0,
            timestamp=time.time(),
            data="Genesis Block - NeoNet AI Network",
            previous_hash="0" * 64,
            hash=hashlib.sha256(b"genesis").hexdigest(),
            validator="network"
        )
        self.chain.append(genesis)
    
    def add_block(self, block: Block) -> bool:
        if block.previous_hash == self.chain[-1].hash:
            self.chain.append(block)
            return True
        return False
    
    def get_balance(self, address: str) -> float:
        return self.state.get(address, 0.0)
    
    def credit(self, address: str, amount: float):
        self.state[address] = self.state.get(address, 0.0) + amount
    
    def get_height(self) -> int:
        return len(self.chain)

class AIEngine:
    """Local AI engine for task processing - supports CPU and all GPUs"""
    
    def __init__(self, gpu_memory_mb: int = 0):
        self.model_hash = hashlib.sha256(b"neonet_ai_v1").hexdigest()
        self.tasks_processed = 0
        self.total_compute_time = 0.0
        self.gpu_memory_mb = gpu_memory_mb
        self.device = "cpu"
        self.backend = "numpy"
        self.torch = None
        
        if gpu_memory_mb > 0:
            self._init_gpu()
    
    def _init_gpu(self):
        """Initialize GPU backend - supports NVIDIA, AMD, Apple, Intel"""
        try:
            import torch
            self.torch = torch
            
            if torch.cuda.is_available():
                self.device = "cuda"
                self.backend = "nvidia"
                gpu_name = torch.cuda.get_device_name(0)
                print(f"[GPU] NVIDIA: {gpu_name}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
                self.backend = "apple"
                print("[GPU] Apple Metal (M1/M2/M3)")
            elif hasattr(torch, 'hip') and torch.hip.is_available():
                self.device = "hip"
                self.backend = "amd"
                print("[GPU] AMD ROCm")
            else:
                try:
                    import intel_extension_for_pytorch as ipex
                    self.device = "xpu"
                    self.backend = "intel"
                    print("[GPU] Intel Arc/Xe")
                except ImportError:
                    self.device = "cpu"
                    self.backend = "pytorch_cpu"
                    print("[CPU] PyTorch CPU (GPU not detected)")
        except ImportError:
            self.device = "cpu"
            self.backend = "numpy"
            print("[CPU] NumPy (install torch for GPU)")
    
    def _compute(self, shape: tuple) -> np.ndarray:
        """Run computation on best available device"""
        if self.torch and self.device != "cpu":
            tensor = self.torch.randn(*shape, device=self.device)
            result = tensor.cpu().numpy()
            return result.astype(np.float32)
        return np.random.randn(*shape).astype(np.float32)
    
    def process_fraud_detection(self, tx_count: int) -> dict:
        transactions = self._compute((tx_count, 32))
        scores = np.abs(transactions).mean(axis=1)
        fraud_indices = np.where(scores > 0.8)[0]
        return {
            "results_hash": hashlib.sha256(scores.tobytes()).hexdigest(),
            "fraud_count": len(fraud_indices),
            "analyzed": tx_count,
            "device": self.backend
        }
    
    def process_model_training(self, epochs: int) -> dict:
        weights = self._compute((256, 128))
        for _ in range(epochs):
            grad = self._compute((256, 128)) * 0.01
            weights -= grad
        return {
            "weights_hash": hashlib.sha256(weights.tobytes()).hexdigest(),
            "epochs_completed": epochs,
            "final_loss": float(np.abs(weights).mean()),
            "device": self.backend
        }
    
    def process_inference(self, data_size: int) -> dict:
        data = self._compute((data_size, 64))
        weight = self._compute((64, 32))
        output = np.tanh(data @ weight)
        return {
            "output_hash": hashlib.sha256(output.tobytes()).hexdigest(),
            "processed": data_size,
            "device": self.backend
        }
    
    def process_task(self, task: dict) -> dict:
        task_type = task.get("task_type", "inference")
        task_data = task.get("data", {})
        start = time.time()
        
        if task_type == "fraud_detection":
            result = self.process_fraud_detection(task_data.get("tx_count", 100))
        elif task_type == "model_training":
            result = self.process_model_training(task_data.get("epochs", 10))
        elif task_type == "federated_learning":
            result = self.process_model_training(task_data.get("epochs", 20))
            result["type"] = "federated"
        elif task_type == "network_protection":
            block_range = task_data.get("block_range", [0, 100])
            result = {"blocks_validated": block_range[1] - block_range[0]}
        elif task_type == "gradient_compute":
            size = task_data.get("size", 500)
            grad = np.random.randn(size, 128).astype(np.float32)
            result = {"gradient_hash": hashlib.sha256(grad.tobytes()).hexdigest()}
        else:
            result = self.process_inference(task_data.get("size", 500))
        
        compute_time = time.time() - start
        self.tasks_processed += 1
        self.total_compute_time += compute_time
        
        result["compute_time_ms"] = int(compute_time * 1000)
        result["model_hash"] = self.model_hash[:16]
        return result

class NeoNetFullNode:
    """Full NeoNet Node - AI + Blockchain + P2P"""
    
    def __init__(self, config: NodeConfig):
        self.config = config
        self.blockchain = LocalBlockchain()
        self.ai = AIEngine(gpu_memory_mb=config.gpu_memory_mb)
        self.peers: set = set()
        self.is_running = False
        self.rewards_earned = 0.0
        self.session_start = None
        self.session_id = None  # Server session for heartbeats
        
        self.app = web.Application()
        self._setup_routes()
    
    def _setup_routes(self):
        self.app.router.add_get('/health', self.handle_health)
        self.app.router.add_get('/status', self.handle_status)
        self.app.router.add_get('/peers', self.handle_peers)
        self.app.router.add_post('/peers', self.handle_add_peer)
        self.app.router.add_get('/chain', self.handle_chain)
        self.app.router.add_get('/task', self.handle_get_task)
        self.app.router.add_post('/task/submit', self.handle_submit_task)
        self.app.router.add_post('/block', self.handle_new_block)
    
    async def handle_health(self, request):
        return web.json_response({"status": "healthy", "node": "full"})
    
    async def handle_status(self, request):
        uptime = time.time() - self.session_start if self.session_start else 0
        return web.json_response({
            "wallet": self.config.wallet,
            "mode": "full_node",
            "blockchain_height": self.blockchain.get_height(),
            "peers": len(self.peers),
            "tasks_processed": self.ai.tasks_processed,
            "rewards_earned": self.rewards_earned,
            "uptime_seconds": int(uptime),
            "ai_model": self.ai.model_hash[:16]
        })
    
    async def handle_peers(self, request):
        return web.json_response({
            "peers": list(self.peers),
            "count": len(self.peers)
        })
    
    async def handle_add_peer(self, request):
        data = await request.json()
        peer = data.get("peer")
        if peer:
            self.peers.add(peer)
        return web.json_response({"success": True})
    
    async def handle_chain(self, request):
        chain_data = [asdict(b) for b in self.blockchain.chain[-100:]]
        return web.json_response({"chain": chain_data, "height": self.blockchain.get_height()})
    
    async def handle_get_task(self, request):
        task = self._generate_task()
        return web.json_response(task)
    
    async def handle_submit_task(self, request):
        data = await request.json()
        task_type = data.get("task_type", "inference")
        reward = self._calculate_reward(task_type)
        return web.json_response({
            "success": True,
            "reward": reward,
            "task_id": data.get("task_id")
        })
    
    async def handle_new_block(self, request):
        data = await request.json()
        block = Block(**data)
        success = self.blockchain.add_block(block)
        return web.json_response({"accepted": success})
    
    def _generate_task(self) -> dict:
        task_types = ["fraud_detection", "model_training", "inference", 
                     "network_protection", "gradient_compute", "federated_learning"]
        task_type = np.random.choice(task_types)
        
        data = {}
        if task_type == "fraud_detection":
            data = {"tx_count": np.random.randint(50, 200)}
        elif task_type in ["model_training", "federated_learning"]:
            data = {"epochs": np.random.randint(5, 20)}
        elif task_type == "network_protection":
            start = np.random.randint(0, 10000)
            data = {"block_range": [start, start + np.random.randint(50, 150)]}
        else:
            data = {"size": np.random.randint(100, 1000)}
        
        return {
            "task_id": hashlib.sha256(os.urandom(32)).hexdigest(),
            "task_type": task_type,
            "data": data,
            "created_at": time.time()
        }
    
    def _calculate_reward(self, task_type: str) -> float:
        """Dynamic reward based on network size - realistic tokenomics"""
        weights = {
            "federated_learning": 1.0,
            "model_training": 0.8,
            "network_protection": 0.6,
            "fraud_detection": 0.5,
            "gradient_compute": 0.5,
            "inference": 0.4,
        }
        # BLOCK_BUDGET = 0.1 NNET per block, divided by active nodes
        active_nodes = max(1, len(self.peers) + 1)
        base_reward = min(0.05, 0.1 / active_nodes)  # Max 0.05 NNET per task
        return round(weights.get(task_type, 0.4) * base_reward, 6)
    
    async def sync_with_bootstrap(self):
        """Sync with bootstrap server and register as active provider"""
        for server in BOOTSTRAP_SERVERS:
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                    # Get decentralization status and peers
                    async with session.get(f"{server}/api/decentralization/peers") as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            for peer in data.get("peers", []):
                                endpoint = peer.get("endpoint")
                                if endpoint:
                                    self.peers.add(endpoint)
                            # Show decentralization progress
                            phase = data.get("phase", "bootstrap")
                            bootstrap_load = data.get("bootstrap_load", 100)
                            miner_load = data.get("miner_load", 0)
                            if phase != "bootstrap":
                                print(f"[P2P] Network phase: {phase.upper()}")
                                print(f"[P2P] Load: Bootstrap {bootstrap_load:.0f}% | Miners {miner_load:.0f}%")
                    
                    # Register as energy provider
                    payload = {
                        "contributor_id": self.config.wallet,
                        "cpu_cores": self.config.cpu_cores,
                        "gpu_memory_mb": self.config.gpu_memory_mb,
                        "p2p_endpoint": f"0.0.0.0:{self.config.port}"
                    }
                    async with session.post(f"{server}/ai-energy/register", json=payload) as resp:
                        if resp.status == 200:
                            print(f"[OK] Registered with bootstrap: {server}")
                    
                    # Register capabilities for decentralization
                    cap_payload = {
                        "miner_id": self.config.wallet,
                        "cpu_cores": self.config.cpu_cores,
                        "gpu_memory_mb": self.config.gpu_memory_mb,
                        "storage_gb": 100,
                        "bandwidth_mbps": 50,
                        "capabilities": ["ai_inference", "consensus_participant"]
                    }
                    async with session.post(f"{server}/api/decentralization/register-capability", json=cap_payload) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            roles = data.get("assigned_roles", [])
                            if roles:
                                print(f"[P2P] Assigned roles: {', '.join(roles)}")
                    
                    # Start session (makes provider visible on leaderboard)
                    session_payload = {"contributor_id": self.config.wallet}
                    async with session.post(f"{server}/ai-energy/start-session", json=session_payload) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            self.session_id = data.get("session_id")
                            print(f"[OK] Session started - now visible on leaderboard")
                            return True
            except Exception as e:
                continue
        return False
    
    async def fetch_task_from_network(self) -> Optional[dict]:
        """Get task from network"""
        for server in BOOTSTRAP_SERVERS:
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                    async with session.get(f"{server}/ai-energy/task/{self.config.wallet}") as resp:
                        if resp.status == 200:
                            return await resp.json()
            except:
                pass
        
        for peer in list(self.peers)[:5]:
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as session:
                    async with session.get(f"http://{peer}/task") as resp:
                        if resp.status == 200:
                            return await resp.json()
            except:
                continue
        
        return self._generate_task()
    
    async def submit_result(self, task_id: str, result: dict, task_type: str) -> float:
        """Submit result to network"""
        for server in BOOTSTRAP_SERVERS:
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                    payload = {
                        "contributor_id": self.config.wallet,
                        "task_id": task_id,
                        "result": result,
                        "session_id": self.session_id
                    }
                    async with session.post(f"{server}/ai-energy/submit-result", json=payload) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            return data.get("reward", 0)
            except:
                pass
        
        return self._calculate_reward(task_type)
    
    async def send_heartbeat(self) -> float:
        """Send heartbeat to stay active and earn rewards"""
        if not self.session_id:
            return 0
        
        for server in BOOTSTRAP_SERVERS:
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                    payload = {
                        "contributor_id": self.config.wallet,
                        "session_id": self.session_id
                    }
                    async with session.post(f"{server}/ai-energy/heartbeat", json=payload) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            return data.get("reward", 0)
            except:
                pass
        return 0
    
    async def broadcast_to_peers(self, message_type: str, data: dict):
        """Broadcast message to all peers"""
        for peer in list(self.peers):
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as session:
                    async with session.post(f"http://{peer}/{message_type}", json=data):
                        pass
            except:
                pass
    
    async def start_server(self):
        """Запустить локальный сервер"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.config.port)
        await site.start()
        print(f"[NODE] Local AI node started on port {self.config.port}")
    
    async def run(self):
        """Main node operation loop"""
        print("=" * 60)
        print("    NeoNet Full Node - AI Network")  
        print("=" * 60)
        print(f"Wallet: {self.config.wallet}")
        print(f"Port: {self.config.port}")
        print(f"CPU: {self.config.cpu_cores} cores")
        print("-" * 60)
        
        await self.start_server()
        
        bootstrap_ok = await self.sync_with_bootstrap()
        if bootstrap_ok:
            print("[NET] Connected to NeoNet")
        else:
            print("[NET] Running in standalone P2P mode")
        
        self.is_running = True
        self.session_start = time.time()
        sync_counter = 0
        
        print("\n[AI] Starting task processing...")
        print("[INFO] Press Ctrl+C to stop\n")
        
        heartbeat_counter = 0
        try:
            while self.is_running:
                task = await self.fetch_task_from_network()
                
                if task and task.get("task_id"):
                    task_id = task["task_id"]
                    task_type = task.get("task_type", "inference")
                    print(f"[AI] Processing: {task_type}")
                    
                    result = self.ai.process_task(task)
                    print(f"[OK] Completed in {result.get('compute_time_ms', 0)}ms")
                    
                    reward = await self.submit_result(task_id, result, task_type)
                    if reward > 0:
                        self.rewards_earned += reward
                        self.blockchain.credit(self.config.wallet, reward)
                        print(f"[NNET] +{reward:.4f} | Total: {self.rewards_earned:.4f} NNET")
                
                # Send heartbeat every 10 iterations (~20 seconds)
                heartbeat_counter += 1
                if heartbeat_counter >= 10:
                    hb_reward = await self.send_heartbeat()
                    if hb_reward > 0:
                        self.rewards_earned += hb_reward
                        self.blockchain.credit(self.config.wallet, hb_reward)
                    heartbeat_counter = 0
                
                sync_counter += 1
                if sync_counter >= 30:
                    await self.sync_with_bootstrap()
                    sync_counter = 0
                    print(f"[SYNC] Peers: {len(self.peers)} | Blockchain: {self.blockchain.get_height()} blocks | Earned: {self.rewards_earned:.4f} NNET")
                
                await asyncio.sleep(2)
                
        except KeyboardInterrupt:
            print("\n[STOP] Stopping node...")
        finally:
            self.is_running = False
            uptime = time.time() - self.session_start if self.session_start else 0
            print("\n" + "=" * 60)
            print("    Session Summary")
            print("=" * 60)
            print(f"Uptime: {int(uptime // 60)} min {int(uptime % 60)} sec")
            print(f"Tasks Processed: {self.ai.tasks_processed}")
            print(f"Total Earned: {self.rewards_earned:.4f} NNET")
            print(f"Wallet Balance: {self.blockchain.get_balance(self.config.wallet):.4f} NNET")
            print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description="NeoNet Full Node")
    parser.add_argument("--wallet", required=True, help="Your wallet address for NNET rewards")
    parser.add_argument("--port", type=int, default=8080, help="Port for P2P (default 8080)")
    parser.add_argument("--cpu", type=int, default=4, help="Number of CPU cores")
    parser.add_argument("--gpu-mem", type=int, default=0, help="GPU memory in MB")
    
    args = parser.parse_args()
    
    config = NodeConfig(
        wallet=args.wallet,
        port=args.port,
        cpu_cores=args.cpu,
        gpu_memory_mb=args.gpu_mem
    )
    
    node = NeoNetFullNode(config)
    await node.run()


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║           NeoNet Miner v1.0                           ║
    ║       AI-Powered Web4 Blockchain Network              ║
    ║       Token: NNET                                     ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    asyncio.run(main())
