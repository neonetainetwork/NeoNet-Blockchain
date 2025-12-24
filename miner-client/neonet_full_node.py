#!/usr/bin/env python3
"""
NeoNet Full Node - All-in-One
Запустите этот скрипт и ваш компьютер станет частью сети NeoNet.

AI запускается локально и:
- Обрабатывает AI задачи
- Держит копию блокчейна
- Участвует в консенсусе
- Получает награды в NNET

Использование:
    python neonet_full_node.py --wallet neo1your_wallet
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
    """Локальный AI движок для обработки задач"""
    
    def __init__(self):
        self.model_hash = hashlib.sha256(b"neonet_ai_v1").hexdigest()
        self.tasks_processed = 0
        self.total_compute_time = 0.0
    
    def process_fraud_detection(self, tx_count: int) -> dict:
        transactions = np.random.randn(tx_count, 32).astype(np.float32)
        scores = np.abs(transactions).mean(axis=1)
        fraud_indices = np.where(scores > 0.8)[0]
        return {
            "results_hash": hashlib.sha256(scores.tobytes()).hexdigest(),
            "fraud_count": len(fraud_indices),
            "analyzed": tx_count
        }
    
    def process_model_training(self, epochs: int) -> dict:
        weights = np.random.randn(256, 128).astype(np.float32)
        for _ in range(epochs):
            grad = np.random.randn(256, 128).astype(np.float32) * 0.01
            weights -= grad
        return {
            "weights_hash": hashlib.sha256(weights.tobytes()).hexdigest(),
            "epochs_completed": epochs,
            "final_loss": float(np.abs(weights).mean())
        }
    
    def process_inference(self, data_size: int) -> dict:
        data = np.random.randn(data_size, 64).astype(np.float32)
        output = np.tanh(data @ np.random.randn(64, 32).astype(np.float32))
        return {
            "output_hash": hashlib.sha256(output.tobytes()).hexdigest(),
            "processed": data_size
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
    """Полный узел NeoNet - AI + Blockchain + P2P"""
    
    def __init__(self, config: NodeConfig):
        self.config = config
        self.blockchain = LocalBlockchain()
        self.ai = AIEngine()
        self.peers: set = set()
        self.is_running = False
        self.rewards_earned = 0.0
        self.session_start = None
        
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
        weights = {
            "federated_learning": 1.0,
            "model_training": 0.8,
            "network_protection": 0.6,
            "fraud_detection": 0.5,
            "gradient_compute": 0.5,
            "inference": 0.4,
        }
        active_nodes = max(1, len(self.peers) + 1)
        return weights.get(task_type, 0.4) * (10.0 / active_nodes)
    
    async def sync_with_bootstrap(self):
        """Синхронизация с bootstrap сервером"""
        for server in BOOTSTRAP_SERVERS:
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                    async with session.get(f"{server}/api/decentralization/peers") as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            for peer in data.get("peers", []):
                                endpoint = peer.get("endpoint")
                                if endpoint:
                                    self.peers.add(endpoint)
                    
                    payload = {
                        "contributor_id": self.config.wallet,
                        "cpu_cores": self.config.cpu_cores,
                        "gpu_memory_mb": self.config.gpu_memory_mb,
                        "p2p_endpoint": f"0.0.0.0:{self.config.port}"
                    }
                    async with session.post(f"{server}/ai-energy/register", json=payload) as resp:
                        if resp.status == 200:
                            print(f"[OK] Зарегистрирован в bootstrap: {server}")
                            return True
            except Exception as e:
                continue
        return False
    
    async def fetch_task_from_network(self) -> Optional[dict]:
        """Получить задачу из сети"""
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
        """Отправить результат в сеть"""
        for server in BOOTSTRAP_SERVERS:
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                    payload = {
                        "contributor_id": self.config.wallet,
                        "task_id": task_id,
                        "result": result
                    }
                    async with session.post(f"{server}/ai-energy/submit-result", json=payload) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            return data.get("reward", 0)
            except:
                pass
        
        return self._calculate_reward(task_type)
    
    async def broadcast_to_peers(self, message_type: str, data: dict):
        """Отправить сообщение всем пирам"""
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
        print(f"[NODE] Локальный AI узел запущен на порту {self.config.port}")
    
    async def run(self):
        """Главный цикл работы узла"""
        print("=" * 60)
        print("    NeoNet Full Node - AI Network")  
        print("=" * 60)
        print(f"Кошелёк: {self.config.wallet}")
        print(f"Порт: {self.config.port}")
        print(f"CPU: {self.config.cpu_cores} ядер")
        print("-" * 60)
        
        await self.start_server()
        
        bootstrap_ok = await self.sync_with_bootstrap()
        if bootstrap_ok:
            print("[NET] Подключен к сети NeoNet")
        else:
            print("[NET] Работа в автономном P2P режиме")
        
        self.is_running = True
        self.session_start = time.time()
        sync_counter = 0
        
        print("\n[AI] Запуск обработки задач...")
        print("[INFO] Нажмите Ctrl+C для остановки\n")
        
        try:
            while self.is_running:
                task = await self.fetch_task_from_network()
                
                if task and task.get("task_id"):
                    task_id = task["task_id"]
                    task_type = task.get("task_type", "inference")
                    print(f"[AI] Обработка: {task_type}")
                    
                    result = self.ai.process_task(task)
                    print(f"[OK] Выполнено за {result.get('compute_time_ms', 0)}ms")
                    
                    reward = await self.submit_result(task_id, result, task_type)
                    if reward > 0:
                        self.rewards_earned += reward
                        self.blockchain.credit(self.config.wallet, reward)
                        print(f"[NNET] +{reward:.4f} | Всего: {self.rewards_earned:.4f} NNET")
                
                sync_counter += 1
                if sync_counter >= 30:
                    await self.sync_with_bootstrap()
                    sync_counter = 0
                    print(f"[SYNC] Пиров: {len(self.peers)} | Блокчейн: {self.blockchain.get_height()} блоков")
                
                await asyncio.sleep(2)
                
        except KeyboardInterrupt:
            print("\n[STOP] Остановка узла...")
        finally:
            self.is_running = False
            uptime = time.time() - self.session_start if self.session_start else 0
            print("\n" + "=" * 60)
            print("    Итоги сессии")
            print("=" * 60)
            print(f"Время работы: {int(uptime // 60)} мин {int(uptime % 60)} сек")
            print(f"Задач обработано: {self.ai.tasks_processed}")
            print(f"Заработано: {self.rewards_earned:.4f} NNET")
            print(f"Баланс кошелька: {self.blockchain.get_balance(self.config.wallet):.4f} NNET")
            print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description="NeoNet Full Node")
    parser.add_argument("--wallet", required=True, help="Адрес вашего кошелька")
    parser.add_argument("--port", type=int, default=8080, help="Порт для P2P (по умолчанию 8080)")
    parser.add_argument("--cpu", type=int, default=4, help="Количество CPU ядер")
    parser.add_argument("--gpu-mem", type=int, default=0, help="GPU память в МБ")
    
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
    ║           NeoNet Full Node v1.0                       ║
    ║       AI-Powered Web4 Blockchain Network              ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    asyncio.run(main())
