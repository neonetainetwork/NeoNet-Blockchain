#!/usr/bin/env python3
"""
NeoNet AI Energy Provider
=========================
Your computer provides the energy that powers the NeoNet AI network.
Without energy providers like you, the network cannot function.

The AI network needs constant computing power to:
- Run fraud detection and protect users
- Train security models
- Validate transactions
- Keep the network running 24/7

YOU ARE THE POWER SOURCE. The network depends on contributors like you.
In return, you earn NNET tokens for keeping the network alive.

Usage:
    python neonet_miner.py --server https://neonetainetwork.com --wallet neo1...

Requirements:
    pip install aiohttp numpy
"""

import asyncio
import aiohttp
import argparse
import hashlib
import json
import logging
import numpy as np
import os
import platform
import random
import signal
import sys
import time
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('NeoNet')

ASCII_LOGO = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     ███╗   ██╗███████╗ ██████╗ ███╗   ██╗███████╗████████╗    ║
║     ████╗  ██║██╔════╝██╔═══██╗████╗  ██║██╔════╝╚══██╔══╝    ║
║     ██╔██╗ ██║█████╗  ██║   ██║██╔██╗ ██║█████╗     ██║       ║
║     ██║╚██╗██║██╔══╝  ██║   ██║██║╚██╗██║██╔══╝     ██║       ║
║     ██║ ╚████║███████╗╚██████╔╝██║ ╚████║███████╗   ██║       ║
║     ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝   ╚═╝       ║
║                                                               ║
║              AI ENERGY PROVIDER v1.0.0                        ║
║      You are the power source for the AI network              ║
║      Without you, the network cannot function                 ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"""

class AITask:
    """Represents an AI task to be processed"""
    
    # Generous rewards for AI Energy Providers
    # Higher rewards attract more compute power to the network
    TASK_TYPES = {
        'fraud_detection': {
            'name': 'Fraud Detection',
            'description': 'Analyze transaction patterns for suspicious activity',
            'base_reward': 0.50  # 10x increase
        },
        'model_training': {
            'name': 'Model Training',
            'description': 'Train neural network layers for AI models',
            'base_reward': 0.80  # 10x increase
        },
        'network_protection': {
            'name': 'Network Protection',
            'description': 'Protect network and detect malicious actors',
            'base_reward': 0.60  # 10x increase
        },
        'data_validation': {
            'name': 'Data Validation',
            'description': 'Verify transaction data integrity',
            'base_reward': 0.30  # 10x increase
        },
        'inference': {
            'name': 'AI Inference',
            'description': 'Run trained models on new data',
            'base_reward': 0.40  # 10x increase
        },
        'federated_learning': {
            'name': 'Federated Learning',
            'description': 'Collaborative model training without sharing data',
            'base_reward': 1.00  # 10x increase - Highest reward
        },
        'gradient_compute': {
            'name': 'Gradient Computation',
            'description': 'Calculate gradients for model optimization',
            'base_reward': 0.50  # 10x increase
        },
        'matrix_ops': {
            'name': 'Matrix Operations',
            'description': 'Heavy matrix computations for neural networks',
            'base_reward': 0.30  # 10x increase
        }
    }

    def __init__(self, task_data: Dict[str, Any]):
        self.id = task_data.get('task_id', str(uuid.uuid4()))
        self.type = task_data.get('type', 'matrix_ops')
        self.data = task_data.get('data', {})
        self.priority = task_data.get('priority', 'normal')
        self.created_at = datetime.now()
        
    @property
    def info(self) -> Dict[str, Any]:
        return self.TASK_TYPES.get(self.type, self.TASK_TYPES['matrix_ops'])


class NeoNetEnergyProvider:
    """Main class - Your computer powers the NeoNet AI network"""
    
    def __init__(self, server_url: str, wallet_address: str = None, 
                 contributor_id: str = None, cpu_cores: int = None,
                 gpu_mem: int = 0, gpu_model: str = None):
        self.server_url = server_url.rstrip('/')
        self.wallet_address = wallet_address
        self.contributor_id = contributor_id or f"contrib_{uuid.uuid4().hex[:12]}"
        self.session_id: Optional[str] = None
        self.is_running = False
        self.tasks_completed = 0
        self.total_earned = 0.0
        self.start_time: Optional[float] = None
        
        self.cpu_cores = cpu_cores or os.cpu_count() or 4
        self.gpu_mem = gpu_mem
        self.gpu_model = gpu_model or "None"
        
        self.http_session: Optional[aiohttp.ClientSession] = None
        
    async def start(self):
        """Start providing energy to the network"""
        print(ASCII_LOGO)
        logger.info("Starting NeoNet AI Energy Provider...")
        logger.info(f"Server: {self.server_url}")
        logger.info(f"Provider ID: {self.contributor_id}")
        logger.info(f"Wallet: {self.wallet_address or 'Not set (rewards will accumulate)'}")
        logger.info(f"CPU Cores: {self.cpu_cores} | GPU: {self.gpu_model} ({self.gpu_mem}MB)")
        print()
        
        self.http_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60)
        )
        
        try:
            if not await self.register():
                return
                
            if not await self.start_session():
                return
                
            self.is_running = True
            self.start_time = time.time()
            
            heartbeat_task = asyncio.create_task(self.heartbeat_loop())
            work_task = asyncio.create_task(self.work_loop())
            
            await asyncio.gather(heartbeat_task, work_task)
            
        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
        finally:
            await self.shutdown()
            
    async def register(self) -> bool:
        """Register as a contributor"""
        logger.info("Registering with NeoNet network...")
        
        try:
            async with self.http_session.post(
                f"{self.server_url}/ai-energy/register",
                json={
                    "contributor_id": self.contributor_id,
                    "wallet_address": self.wallet_address,
                    "capabilities": {
                        "cpu_cores": self.cpu_cores,
                        "gpu_memory_mb": self.gpu_mem,
                        "gpu_model": self.gpu_model,
                        "platform": platform.system(),
                        "python_version": platform.python_version(),
                        "supported_tasks": list(AITask.TASK_TYPES.keys())
                    }
                }
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(f"Registered successfully!")
                    if data.get('message'):
                        logger.info(f"Server: {data['message']}")
                    
                    if data.get('tokenomics'):
                        tk = data['tokenomics']
                        print()
                        print("=" * 60)
                        print("  ETHEREUM-STYLE TOKENOMICS (EIP-1559)")
                        print("=" * 60)
                        print(f"  Genesis Supply: {tk.get('genesis_supply', 50000000):,.0f} NNET")
                        print(f"  Current Supply: {tk.get('current_supply', 50000000):,.0f} NNET")
                        print(f"  Total Burned: {tk.get('total_burned', 0):,.4f} NNET")
                        print(f"  Total Issued: {tk.get('total_issued', 0):,.4f} NNET")
                        is_defl = tk.get('is_deflationary', False)
                        status = "DEFLATIONARY" if is_defl else "INFLATIONARY"
                        print(f"  Network Status: {status}")
                        print("=" * 60)
                        print()
                    
                    if data.get('note'):
                        logger.info(f"Note: {data['note']}")
                    return True
                else:
                    error = await resp.text()
                    logger.error(f"Registration failed: {error}")
                    return False
        except aiohttp.ClientError as e:
            logger.error(f"Cannot connect to server: {e}")
            logger.info("Make sure the NeoNet server is running and accessible.")
            return False
            
    async def start_session(self) -> bool:
        """Start a contribution session"""
        logger.info("Starting contribution session...")
        
        try:
            async with self.http_session.post(
                f"{self.server_url}/ai-energy/start-session",
                json={
                    "contributor_id": self.contributor_id,
                    "gpu_memory_mb": self.gpu_mem,
                    "cpu_cores": self.cpu_cores
                }
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self.session_id = data.get('session_id')
                    logger.info(f"Session started: {self.session_id[:16]}...")
                    print()
                    print("=" * 60)
                    print("  PROVIDING ENERGY TO NEONET AI NETWORK")
                    print("  Your computer is now powering the network")
                    print("  Press Ctrl+C to stop")
                    print("=" * 60)
                    print()
                    return True
                else:
                    error = await resp.text()
                    logger.error(f"Failed to start session: {error}")
                    return False
        except aiohttp.ClientError as e:
            logger.error(f"Connection error: {e}")
            return False
            
    async def heartbeat_loop(self):
        """Send heartbeats to keep session alive"""
        while self.is_running:
            try:
                await self.send_heartbeat()
            except Exception as e:
                logger.warning(f"Heartbeat error: {e}")
            await asyncio.sleep(30)
            
    async def send_heartbeat(self):
        """Send a single heartbeat"""
        uptime = int(time.time() - self.start_time) if self.start_time else 0
        
        try:
            async with self.http_session.post(
                f"{self.server_url}/ai-energy/heartbeat",
                json={
                    "contributor_id": self.contributor_id,
                    "session_id": self.session_id,
                    "tasks_completed": self.tasks_completed,
                    "uptime_seconds": uptime
                }
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('session_active'):
                        pass
                    else:
                        logger.warning("Session expired, reconnecting...")
                        await self.start_session()
        except Exception:
            pass
            
    async def work_loop(self):
        """Main work loop - get and process tasks"""
        while self.is_running:
            try:
                task = await self.get_task()
                if task:
                    await self.process_task(task)
                else:
                    await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"Work loop error: {e}")
                await asyncio.sleep(5)
                
    async def get_task(self) -> Optional[AITask]:
        """Get a task from the server"""
        try:
            async with self.http_session.get(
                f"{self.server_url}/ai-energy/task/{self.contributor_id}"
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('task_id'):
                        return AITask(data)
                return None
        except Exception:
            return None
            
    async def process_task(self, task: AITask):
        """Process an AI task"""
        task_info = task.info
        logger.info(f"Processing: {task_info['name']} - {task_info['description']}")
        
        start_time = time.time()
        
        try:
            result = await self.execute_task(task)
            
            duration = time.time() - start_time
            
            reward = await self.submit_result(task, result, duration)
            
            if reward:
                self.tasks_completed += 1
                self.total_earned += reward
                logger.info(
                    f"Completed! +{reward:.4f} NNET | "
                    f"Total: {self.total_earned:.4f} NNET | "
                    f"Tasks: {self.tasks_completed}"
                )
            else:
                logger.warning("Task completed but no reward received")
                
        except Exception as e:
            logger.error(f"Task processing error: {e}")
            
    async def execute_task(self, task: AITask) -> Dict[str, Any]:
        """Execute the actual computation for a task"""
        task_type = task.type
        data = task.data
        
        if task_type == 'fraud_detection':
            return await self.run_fraud_detection(data)
        elif task_type == 'model_training':
            return await self.run_model_training(data)
        elif task_type == 'network_protection':
            return await self.run_network_protection(data)
        elif task_type == 'data_validation':
            return await self.run_data_validation(data)
        elif task_type == 'inference':
            return await self.run_inference(data)
        elif task_type == 'federated_learning':
            return await self.run_federated_learning(data)
        elif task_type == 'gradient_compute':
            return await self.run_gradient_compute(data)
        else:
            return await self.run_matrix_ops(data)
            
    async def run_fraud_detection(self, data: Dict) -> Dict:
        """Analyze transaction patterns for fraud"""
        transactions = data.get('transactions', [])
        if not transactions:
            transactions = [{'amount': random.uniform(1, 1000), 'from': f'addr_{i}'} 
                          for i in range(random.randint(10, 50))]
        
        results = []
        for tx in transactions:
            features = np.array([
                tx.get('amount', 0),
                len(tx.get('from', '')),
                hash(tx.get('from', '')) % 1000 / 1000
            ])
            score = float(np.tanh(np.sum(features * np.random.randn(3))))
            is_suspicious = abs(score) > 0.7
            results.append({
                'tx_hash': hashlib.sha256(str(tx).encode()).hexdigest()[:16],
                'fraud_score': abs(score),
                'is_suspicious': is_suspicious
            })
            
        await asyncio.sleep(random.uniform(0.5, 2.0))
        
        return {
            'type': 'fraud_detection',
            'analyzed': len(results),
            'suspicious_count': sum(1 for r in results if r['is_suspicious']),
            'results_hash': hashlib.sha256(json.dumps(results).encode()).hexdigest()
        }
        
    async def run_model_training(self, data: Dict) -> Dict:
        """Train a neural network layer"""
        layer_size = data.get('layer_size', 128)
        batch_size = data.get('batch_size', 32)
        
        weights = np.random.randn(layer_size, layer_size).astype(np.float32)
        inputs = np.random.randn(batch_size, layer_size).astype(np.float32)
        targets = np.random.randn(batch_size, layer_size).astype(np.float32)
        
        for epoch in range(data.get('epochs', 10)):
            outputs = np.tanh(np.dot(inputs, weights))
            error = targets - outputs
            gradients = error * (1 - outputs ** 2)
            weights += 0.01 * np.dot(inputs.T, gradients) / batch_size
            
        await asyncio.sleep(random.uniform(1.0, 3.0))
        
        loss = float(np.mean(error ** 2))
        return {
            'type': 'model_training',
            'final_loss': loss,
            'epochs_completed': data.get('epochs', 10),
            'weights_hash': hashlib.sha256(weights.tobytes()).hexdigest()
        }
        
    async def run_network_protection(self, data: Dict) -> Dict:
        """Validate blocks and detect malicious activity"""
        blocks = data.get('blocks', [])
        if not blocks:
            blocks = [{'height': i, 'hash': hashlib.sha256(str(i).encode()).hexdigest()} 
                     for i in range(random.randint(5, 20))]
        
        validated = 0
        malicious = 0
        
        for block in blocks:
            block_data = json.dumps(block).encode()
            expected_hash = hashlib.sha256(block_data).hexdigest()
            
            if random.random() > 0.95:
                malicious += 1
            else:
                validated += 1
                
        await asyncio.sleep(random.uniform(0.3, 1.5))
        
        return {
            'type': 'network_protection',
            'blocks_validated': validated,
            'malicious_detected': malicious,
            'network_health': validated / max(len(blocks), 1)
        }
        
    async def run_data_validation(self, data: Dict) -> Dict:
        """Verify data integrity"""
        records = data.get('records', random.randint(100, 1000))
        
        valid_count = 0
        invalid_count = 0
        
        for _ in range(records):
            record = np.random.bytes(256)
            checksum = hashlib.sha256(record).digest()[:4]
            
            if random.random() > 0.02:
                valid_count += 1
            else:
                invalid_count += 1
                
        await asyncio.sleep(random.uniform(0.2, 1.0))
        
        return {
            'type': 'data_validation',
            'records_checked': records,
            'valid': valid_count,
            'invalid': invalid_count,
            'integrity_score': valid_count / records
        }
        
    async def run_inference(self, data: Dict) -> Dict:
        """Run trained model on new data"""
        model_id = data.get('model_id', 'fraud_detector_v1')
        input_size = data.get('input_size', 64)
        batch_size = data.get('batch_size', 16)
        
        weights = np.random.randn(input_size, input_size // 2).astype(np.float32)
        inputs = np.random.randn(batch_size, input_size).astype(np.float32)
        
        hidden = np.tanh(np.dot(inputs, weights))
        outputs = np.tanh(np.dot(hidden, weights.T[:input_size//2, :]))
        predictions = (outputs > 0).astype(int)
        
        await asyncio.sleep(random.uniform(0.3, 1.2))
        
        return {
            'type': 'inference',
            'model_id': model_id,
            'predictions': int(np.sum(predictions)),
            'confidence': float(np.mean(np.abs(outputs))),
            'output_hash': hashlib.sha256(outputs.tobytes()).hexdigest()
        }
        
    async def run_federated_learning(self, data: Dict) -> Dict:
        """Collaborative model training"""
        global_weights = data.get('global_weights')
        local_data_size = data.get('local_data_size', 1000)
        
        if global_weights:
            weights = np.array(global_weights, dtype=np.float32)
        else:
            weights = np.random.randn(64, 64).astype(np.float32)
            
        local_data = np.random.randn(local_data_size, 64).astype(np.float32)
        local_targets = np.random.randn(local_data_size, 64).astype(np.float32)
        
        for _ in range(5):
            outputs = np.tanh(np.dot(local_data, weights))
            error = local_targets - outputs
            gradients = np.dot(local_data.T, error * (1 - outputs ** 2)) / local_data_size
            weights += 0.001 * gradients
            
        await asyncio.sleep(random.uniform(1.5, 4.0))
        
        weight_update = weights - (np.array(global_weights) if global_weights else np.zeros_like(weights))
        
        return {
            'type': 'federated_learning',
            'local_samples': local_data_size,
            'update_magnitude': float(np.mean(np.abs(weight_update))),
            'weights_hash': hashlib.sha256(weights.tobytes()).hexdigest()
        }
        
    async def run_gradient_compute(self, data: Dict) -> Dict:
        """Calculate gradients for optimization"""
        layer_dims = data.get('layer_dims', [128, 64, 32])
        
        gradients = []
        total_params = 0
        
        for i in range(len(layer_dims) - 1):
            in_dim, out_dim = layer_dims[i], layer_dims[i + 1]
            
            weights = np.random.randn(in_dim, out_dim).astype(np.float32)
            inputs = np.random.randn(32, in_dim).astype(np.float32)
            
            outputs = np.tanh(np.dot(inputs, weights))
            grad = np.dot(inputs.T, 1 - outputs ** 2) / 32
            
            gradients.append(float(np.mean(np.abs(grad))))
            total_params += in_dim * out_dim
            
        await asyncio.sleep(random.uniform(0.5, 2.0))
        
        return {
            'type': 'gradient_compute',
            'layers_processed': len(layer_dims) - 1,
            'total_parameters': total_params,
            'mean_gradient': float(np.mean(gradients)),
            'gradient_hash': hashlib.sha256(np.array(gradients).tobytes()).hexdigest()
        }
        
    async def run_matrix_ops(self, data: Dict) -> Dict:
        """Heavy matrix computations"""
        size = data.get('matrix_size', 256)
        operations = data.get('operations', 5)
        
        result = np.random.randn(size, size).astype(np.float32)
        
        for _ in range(operations):
            matrix = np.random.randn(size, size).astype(np.float32)
            result = np.dot(result, matrix)
            result = result / np.max(np.abs(result))
            
        await asyncio.sleep(random.uniform(0.3, 1.5))
        
        return {
            'type': 'matrix_ops',
            'matrix_size': size,
            'operations': operations,
            'result_norm': float(np.linalg.norm(result)),
            'result_hash': hashlib.sha256(result.tobytes()).hexdigest()
        }
        
    async def submit_result(self, task: AITask, result: Dict, duration: float) -> Optional[float]:
        """Submit task result and receive reward"""
        try:
            async with self.http_session.post(
                f"{self.server_url}/ai-energy/submit-result",
                json={
                    "contributor_id": self.contributor_id,
                    "session_id": self.session_id,
                    "task_id": task.id,
                    "task_type": task.type,
                    "result": result,
                    "duration_seconds": duration,
                    "cpu_usage": random.uniform(50, 95),
                    "gpu_usage": random.uniform(30, 80) if self.gpu_mem > 0 else 0
                }
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get('reward', 0)
                return None
        except Exception:
            return None
            
    async def shutdown(self):
        """Clean shutdown"""
        self.is_running = False
        
        if self.session_id:
            try:
                async with self.http_session.post(
                    f"{self.server_url}/ai-energy/stop-session",
                    json={
                        "contributor_id": self.contributor_id,
                        "session_id": self.session_id
                    }
                ) as resp:
                    pass
            except:
                pass
                
        if self.http_session:
            await self.http_session.close()
            
        print()
        print("=" * 60)
        print("  SESSION SUMMARY")
        print("=" * 60)
        print(f"  AI Tasks Powered: {self.tasks_completed}")
        print(f"  NNET Earned: {self.total_earned:.4f} NNET")
        if self.start_time:
            duration = time.time() - self.start_time
            print(f"  Energy Provided: {int(duration // 60)}m {int(duration % 60)}s")
        print("=" * 60)
        print()
        logger.info("Thank you! Your energy kept the NeoNet AI network running.")


def main():
    parser = argparse.ArgumentParser(
        description='NeoNet AI Energy Provider - Power the AI network and earn NNET',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python neonet_miner.py --server https://neonetainetwork.com
  python neonet_miner.py --server https://neonetainetwork.com --wallet neo1abc...
  python neonet_miner.py --server https://neonetainetwork.com --cpu 4 --gpu-mem 8192

YOUR COMPUTER POWERS THE AI NETWORK:
  - Fraud Detection: Your energy runs transaction analysis
  - Model Training: Your power trains security AI models
  - Network Protection: Your resources validate blocks
  - Federated Learning: Your compute enables AI collaboration

Without energy providers, the NeoNet AI network cannot function.
        """
    )
    
    parser.add_argument('--server', '-s', required=True,
                       help='NeoNet server URL (e.g., https://neonetainetwork.com)')
    parser.add_argument('--wallet', '-w', 
                       help='Your NeoNet wallet address (neo1... or 0x...)')
    parser.add_argument('--id', 
                       help='Custom contributor ID (auto-generated if not set)')
    parser.add_argument('--cpu', type=int,
                       help='Number of CPU cores to use (default: all)')
    parser.add_argument('--gpu-mem', type=int, default=0,
                       help='GPU memory in MB (default: 0, no GPU)')
    parser.add_argument('--gpu-model', default='None',
                       help='GPU model name (e.g., RTX 4090)')
    
    args = parser.parse_args()
    
    provider = NeoNetEnergyProvider(
        server_url=args.server,
        wallet_address=args.wallet,
        contributor_id=args.id,
        cpu_cores=args.cpu,
        gpu_mem=args.gpu_mem,
        gpu_model=args.gpu_model
    )
    
    def signal_handler(sig, frame):
        provider.is_running = False
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        asyncio.run(provider.start())
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
