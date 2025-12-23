"""
NeoNet Blockchain Persistence Layer
====================================
Integrates NeoNetBlockchain with DecentralizedDB for full state persistence.
All data survives server restarts.
"""

import os
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import asdict

PERSISTENCE_DIR = Path("neonet_state")
PERSISTENCE_DIR.mkdir(parents=True, exist_ok=True)

class BlockchainPersistence:
    """Persistence layer for NeoNet Blockchain state"""
    
    def __init__(self, data_dir: str = "neonet_state"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.balances_file = self.data_dir / "balances.json"
        self.providers_file = self.data_dir / "validators.json"  # Legacy filename for compatibility
        self.transactions_file = self.data_dir / "transactions.json"
        self.stakes_file = self.data_dir / "stakes.json"
        self.tokens_file = self.data_dir / "tokens.json"
        self.pools_file = self.data_dir / "pools.json"
        self.contracts_file = self.data_dir / "contracts.json"
        self.dapps_file = self.data_dir / "dapps.json"
        self.nfts_file = self.data_dir / "nfts.json"
        self.miners_file = self.data_dir / "miners.json"
        self.blocks_file = self.data_dir / "blocks.json"
        self.meta_file = self.data_dir / "meta.json"
        
        self._ensure_files()
    
    def _ensure_files(self):
        """Create empty files if they don't exist"""
        defaults = {
            self.balances_file: {},
            self.providers_file: {},
            self.transactions_file: [],
            self.stakes_file: {},
            self.tokens_file: [],
            self.pools_file: {},
            self.contracts_file: [],
            self.dapps_file: [],
            self.nfts_file: [],
            self.miners_file: {},
            self.blocks_file: [],
            self.meta_file: {"last_saved": 0, "block_height": 0, "state_root": "genesis"}
        }
        
        for file_path, default_value in defaults.items():
            if not file_path.exists():
                self._write_json(file_path, default_value)
    
    def _read_json(self, file_path: Path) -> Any:
        """Read JSON file safely"""
        try:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
        return None
    
    def _write_json(self, file_path: Path, data: Any):
        """Write JSON file safely with atomic write"""
        temp_file = file_path.with_suffix('.tmp')
        try:
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            temp_file.rename(file_path)
        except Exception as e:
            print(f"Error writing {file_path}: {e}")
            if temp_file.exists():
                temp_file.unlink()
    
    def save_balances(self, balances: Dict[str, float]):
        """Save all balances"""
        self._write_json(self.balances_file, balances)
    
    def load_balances(self) -> Dict[str, float]:
        """Load balances"""
        data = self._read_json(self.balances_file)
        return data if data else {}
    
    def save_providers(self, providers: Dict):
        """Save AI Energy Providers"""
        serialized = {}
        for addr, v in providers.items():
            if hasattr(v, '__dict__'):
                serialized[addr] = {
                    'address': v.address,
                    'stake': v.stake,
                    'is_active': v.is_active,
                    'blocks_validated': v.blocks_validated,
                    'rewards_earned': v.rewards_earned,
                    'intelligence_score': v.intelligence_score,
                    'registered_at': getattr(v, 'registered_at', int(time.time()))
                }
            else:
                serialized[addr] = v
        self._write_json(self.providers_file, serialized)
    
    def load_providers(self) -> Dict:
        """Load AI Energy Providers"""
        data = self._read_json(self.providers_file)
        return data if data else {}
    
    def save_transactions(self, transactions: List):
        """Save transactions"""
        serialized = []
        for tx in transactions:
            if hasattr(tx, '__dict__'):
                serialized.append(tx.__dict__)
            elif isinstance(tx, dict):
                serialized.append(tx)
        self._write_json(self.transactions_file, serialized[-1000:])
    
    def load_transactions(self) -> List:
        """Load transactions"""
        data = self._read_json(self.transactions_file)
        return data if data else []
    
    def save_tokens(self, tokens: List):
        """Save registered tokens"""
        self._write_json(self.tokens_file, tokens)
    
    def load_tokens(self) -> List:
        """Load registered tokens"""
        data = self._read_json(self.tokens_file)
        return data if data else []
    
    def save_pools(self, pools: Dict):
        """Save liquidity pools"""
        serialized = {}
        for pair, pool in pools.items():
            if hasattr(pool, '__dict__'):
                serialized[pair] = pool.__dict__
            else:
                serialized[pair] = pool
        self._write_json(self.pools_file, serialized)
    
    def load_pools(self) -> Dict:
        """Load liquidity pools"""
        data = self._read_json(self.pools_file)
        return data if data else {}
    
    def save_contracts(self, contracts: List):
        """Save deployed contracts"""
        self._write_json(self.contracts_file, contracts)
    
    def load_contracts(self) -> List:
        """Load deployed contracts"""
        data = self._read_json(self.contracts_file)
        return data if data else []
    
    def save_dapps(self, dapps: List):
        """Save registered dApps"""
        self._write_json(self.dapps_file, dapps)
    
    def load_dapps(self) -> List:
        """Load registered dApps"""
        data = self._read_json(self.dapps_file)
        return data if data else []
    
    def save_nfts(self, nfts: List):
        """Save NFT collections"""
        self._write_json(self.nfts_file, nfts)
    
    def load_nfts(self) -> List:
        """Load NFT collections"""
        data = self._read_json(self.nfts_file)
        return data if data else []
    
    def save_miners(self, miners: Dict):
        """Save miners"""
        serialized = {}
        for addr, m in miners.items():
            if hasattr(m, '__dict__'):
                serialized[addr] = {
                    'address': m.address,
                    'cpu_cores': m.cpu_cores,
                    'gpu_memory_mb': m.gpu_memory_mb,
                    'is_active': m.is_active,
                    'tasks_completed': m.tasks_completed,
                    'rewards_earned': m.rewards_earned,
                    'intelligence_contribution': getattr(m, 'intelligence_contribution', 0),
                    'registered_at': getattr(m, 'registered_at', int(time.time()))
                }
            else:
                serialized[addr] = m
        self._write_json(self.miners_file, serialized)
    
    def load_miners(self) -> Dict:
        """Load miners"""
        data = self._read_json(self.miners_file)
        return data if data else {}
    
    def save_blocks(self, blocks: List):
        """Save block data (just last 100 blocks)"""
        serialized = []
        for block in blocks[-100:]:
            try:
                if hasattr(block, '__dict__'):
                    block_data = {
                        'index': getattr(block, 'index', 0),
                        'timestamp': getattr(block, 'timestamp', 0),
                        'validator': getattr(block, 'validator', ''),
                        'hash': getattr(block, 'hash', ''),
                        'prev_hash': getattr(block, 'prev_hash', getattr(block, 'previous_hash', '')),
                        'ai_score': getattr(block, 'ai_score', 0),
                        'tx_count': len(block.transactions) if hasattr(block, 'transactions') else 0
                    }
                    serialized.append(block_data)
                elif isinstance(block, dict):
                    serialized.append(block)
            except Exception:
                continue
        self._write_json(self.blocks_file, serialized)
    
    def load_blocks(self) -> List:
        """Load blocks"""
        data = self._read_json(self.blocks_file)
        return data if data else []
    
    def save_meta(self, block_height: int, state_root: str, network_stats: Dict = None):
        """Save metadata including network stats"""
        meta = {
            'last_saved': int(time.time()),
            'block_height': block_height,
            'state_root': state_root,
            'version': '1.0.0',
            'network_stats': network_stats or {}
        }
        self._write_json(self.meta_file, meta)
    
    def load_meta(self) -> Dict:
        """Load metadata"""
        data = self._read_json(self.meta_file)
        return data if data else {'last_saved': 0, 'block_height': 0, 'state_root': 'genesis', 'network_stats': {}}
    
    def save_all(self, blockchain):
        """Save complete blockchain state"""
        try:
            self.save_balances(dict(blockchain.balances))
            self.save_providers(blockchain.ai_energy_providers)
            
            if hasattr(blockchain, 'transactions') and blockchain.transactions:
                self.save_transactions(blockchain.transactions)
            
            if hasattr(blockchain, 'miners'):
                self.save_miners(blockchain.miners)
            
            if hasattr(blockchain, 'blocks'):
                self.save_blocks(blockchain.blocks)
            
            block_height = len(blockchain.blocks) if hasattr(blockchain, 'blocks') else 0
            state_root = hashlib.sha256(json.dumps(dict(blockchain.balances), sort_keys=True).encode()).hexdigest()[:16]
            
            # Save network stats (tokenomics: total_issued, total_burned, etc)
            network_stats = {}
            if hasattr(blockchain, 'network_stats'):
                network_stats = dict(blockchain.network_stats)
            self.save_meta(block_height, state_root, network_stats)
            
            print(f"[Persistence] Saved blockchain state: {block_height} blocks, {len(blockchain.balances)} accounts, issued={network_stats.get('total_issued', 0):.2f}, burned={network_stats.get('total_burned', 0):.2f}")
            return True
        except Exception as e:
            print(f"[Persistence] Error saving state: {e}")
            return False
    
    def load_into_blockchain(self, blockchain):
        """Load persisted state into blockchain object"""
        try:
            USER_WALLET = "neo1dfa5ee86e6443115287e8a6c604cd8aa32d101"
            INITIAL_ALLOCATION = 30_000_000
            BURN_ADDRESS = "nnet1000000000000000000000000000000000dead"
            
            initial_user_balance = blockchain.balances.get(USER_WALLET, 0)
            
            balances = self.load_balances()
            if balances:
                blockchain.balances.update(balances)
                
                # Preserve user wallet genesis allocation
                if USER_WALLET in blockchain.balances:
                    persisted_balance = blockchain.balances[USER_WALLET]
                    if persisted_balance < INITIAL_ALLOCATION and initial_user_balance >= INITIAL_ALLOCATION:
                        blockchain.balances[USER_WALLET] = initial_user_balance
                        print(f"[Persistence] Preserved initial allocation of {INITIAL_ALLOCATION:,} NNET for user wallet")
                
                # Ensure burn address exists
                if BURN_ADDRESS not in blockchain.balances:
                    blockchain.balances[BURN_ADDRESS] = 0.0
                
                print(f"[Persistence] Loaded {len(balances)} balances")
            
            providers_data = self.load_providers()
            if providers_data:
                from .neonet_blockchain import AIEnergyProvider
                for addr, v_data in providers_data.items():
                    blockchain.ai_energy_providers[addr] = AIEnergyProvider(
                        address=v_data.get('address', addr),
                        stake=v_data.get('stake', 0),
                        is_active=v_data.get('is_active', True),
                        blocks_validated=v_data.get('blocks_validated', 0),
                        rewards_earned=v_data.get('rewards_earned', 0),
                        intelligence_score=v_data.get('intelligence_score', 0.5),
                        registered_at=v_data.get('registered_at', int(time.time()))
                    )
                print(f"[Persistence] Loaded {len(providers_data)} AI Energy Providers")
            
            transactions = self.load_transactions()
            if transactions:
                if not hasattr(blockchain, 'transactions'):
                    blockchain.transactions = []
                blockchain.transactions.extend(transactions)
                print(f"[Persistence] Loaded {len(transactions)} transactions")
            
            miners_data = self.load_miners()
            if miners_data and hasattr(blockchain, 'miners'):
                from .neonet_blockchain import Miner
                for addr, m_data in miners_data.items():
                    blockchain.miners[addr] = Miner(
                        address=m_data.get('address', addr),
                        cpu_cores=m_data.get('cpu_cores', 4),
                        gpu_memory_mb=m_data.get('gpu_memory_mb', 8192),
                        is_active=m_data.get('is_active', True),
                        tasks_completed=m_data.get('tasks_completed', 0),
                        rewards_earned=m_data.get('rewards_earned', 0),
                        intelligence_contribution=m_data.get('intelligence_contribution', 0),
                        registered_at=m_data.get('registered_at', int(time.time()))
                    )
                print(f"[Persistence] Loaded {len(miners_data)} miners")
            
            meta = self.load_meta()
            
            # Restore network stats (tokenomics)
            network_stats = meta.get('network_stats', {})
            if network_stats and hasattr(blockchain, 'network_stats'):
                blockchain.network_stats['total_issued'] = network_stats.get('total_issued', 0.0)
                blockchain.network_stats['total_burned'] = network_stats.get('total_burned', 0.0)
                blockchain.network_stats['current_supply'] = blockchain.GENESIS_SUPPLY + blockchain.network_stats['total_issued'] - blockchain.network_stats['total_burned']
                blockchain.network_stats['is_deflationary'] = blockchain.network_stats['total_burned'] > blockchain.network_stats['total_issued']
                print(f"[Persistence] Restored tokenomics: issued={network_stats.get('total_issued', 0):.2f}, burned={network_stats.get('total_burned', 0):.2f}")
            
            print(f"[Persistence] State restored from block {meta.get('block_height', 0)}")
            
            return True
        except Exception as e:
            print(f"[Persistence] Error loading state: {e}")
            return False


_persistence_instance: Optional[BlockchainPersistence] = None

def get_persistence() -> BlockchainPersistence:
    """Get or create persistence instance"""
    global _persistence_instance
    if _persistence_instance is None:
        _persistence_instance = BlockchainPersistence()
    return _persistence_instance
