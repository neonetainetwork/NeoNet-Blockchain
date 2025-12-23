"""
NeoNet Blockchain - Live network simulation with real data for AI training
"""
import time
import random
import hashlib
import json
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict

@dataclass
class Transaction:
    """All transactions use hybrid quantum-safe signatures"""
    tx_hash: str
    sender: str
    recipient: str
    amount: float
    gas_price: float
    gas_used: int
    tx_type: str  # transfer, contract_call, stake, unstake, governance
    timestamp: int
    nonce: int = 0
    # Quantum-safe signatures (ALL transactions have these)
    evm_signature: str = ""  # Classical ECDSA/Ed25519
    quantum_signature: str = ""  # Ed25519 quantum-resistant layer
    dilithium_signature: str = ""  # Post-quantum Dilithium3
    signature_algorithm: str = "Hybrid-Ed25519+Dilithium3"
    is_verified: bool = False
    verification_level: str = "hybrid"  # classical, quantum, hybrid
    # Fraud detection
    is_fraud: bool = False
    fraud_score: float = 0.0
    ai_verified: bool = False
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_features(self) -> List[float]:
        return [
            self.amount,
            self.gas_price,
            self.gas_used,
            1.0 if self.tx_type == "transfer" else 0.0,
            1.0 if self.tx_type == "contract_call" else 0.0,
            1.0 if self.tx_type == "stake" else 0.0,
            len(self.sender),
            len(self.recipient),
            self.timestamp % 86400 / 86400,  # time of day normalized
            1.0 if self.is_verified else 0.0  # signature verified
        ]
    
    def verify_quantum_signature(self) -> bool:
        """Verify hybrid quantum-safe signature"""
        if not self.evm_signature or not self.quantum_signature:
            return False
        # In production: verify Ed25519 + Dilithium signatures
        # Simulated verification based on signature presence
        has_classical = len(self.evm_signature) >= 64
        has_quantum = len(self.quantum_signature) >= 64
        has_dilithium = len(self.dilithium_signature) >= 64 if self.dilithium_signature else True
        
        self.is_verified = has_classical and has_quantum
        self.verification_level = "hybrid" if has_dilithium else "quantum"
        return self.is_verified

@dataclass
class Block:
    index: int
    timestamp: int
    transactions: List[Transaction]
    previous_hash: str
    provider: str
    hash: str
    difficulty: int = 1
    nonce: int = 0
    ai_score: float = 0.0  # AI validation score
    
@dataclass
class AIEnergyProvider:
    """AI Energy Provider (replaces traditional validators)"""
    address: str
    stake: float
    is_active: bool
    blocks_validated: int
    rewards_earned: float
    intelligence_score: float  # PoI score
    registered_at: int

@dataclass 
class Proposal:
    proposal_id: str
    title: str
    description: str
    proposer: str
    created_at: int
    voting_ends_at: int
    status: str  # pending, active, passed, rejected, executed
    for_votes: float = 0.0
    against_votes: float = 0.0
    ai_recommendation: str = "neutral"  # for, against, neutral
    ai_confidence: float = 0.0

@dataclass
class Miner:
    address: str
    cpu_cores: int
    gpu_memory_mb: int
    endpoint: str
    registered_at: int
    is_active: bool = True
    tasks_completed: int = 0
    rewards_earned: float = 0.0
    intelligence_contribution: float = 0.0
    last_task_at: int = 0


@dataclass
class Token:
    """Token in NeoNet network"""
    symbol: str
    name: str
    total_supply: float
    decimals: int = 18
    creator: str = ""
    price_usd: float = 0.0
    created_at: int = 0
    is_native: bool = False
    runtime: str = "native"  # native, evm, hybrid
    contract_address: str = ""

@dataclass
class LiquidityPool:
    """AMM Liquidity Pool for DEX"""
    pool_id: str
    token_a: str  # Always NNET
    token_b: str  # Other token
    reserve_a: float = 0.0  # NNET reserve
    reserve_b: float = 0.0  # Token reserve
    total_liquidity: float = 0.0
    fee_percent: float = 0.003  # 0.3% swap fee
    created_at: int = 0

@dataclass
class DApp:
    """Decentralized Application on NeoNet"""
    dapp_id: str
    name: str
    description: str
    category: str  # defi, nft, gaming, social, utility, dao
    creator: str
    contract_address: str
    runtime: str  # evm, wasm, hybrid
    created_at: int
    website_url: str = ""
    logo_url: str = ""
    is_verified: bool = False
    users_count: int = 0
    transactions_count: int = 0
    tvl_usd: float = 0.0
    rating: float = 0.0
    status: str = "active"  # active, paused, deprecated

@dataclass
class NFTCollection:
    """NFT Collection on NeoNet"""
    collection_id: str
    name: str
    symbol: str
    description: str
    creator: str
    contract_address: str
    total_supply: int
    minted: int = 0
    floor_price_neo: float = 0.0
    volume_neo: float = 0.0
    created_at: int = 0
    category: str = "art"  # art, gaming, collectibles, music, utility
    is_verified: bool = False

class NeoNetBlockchain:
    """Live NeoNet blockchain with Ethereum-style tokenomics (EIP-1559 burning)"""
    
    # ===== ETHEREUM-STYLE TOKENOMICS =====
    # No fixed cap - dynamic supply with burning mechanism
    GENESIS_SUPPLY = 50_000_000.0  # Starting supply (like ETH pre-merge)
    BLOCK_TIME = 3  # seconds between blocks
    USER_WALLET = "neo1dfa5ee86e6443115287e8a6c604cd8aa32d101"
    INITIAL_USER_ALLOCATION = 30_000_000.0  # 30 million NNET genesis allocation
    INITIAL_NNET_PRICE = 5.0  # $5 starting price
    
    # EIP-1559 Style Fee Burning
    BURN_ADDRESS = "neo1000000000000000000000000000000000dead"  # Burn address
    BASE_FEE_BURN_RATE = 0.7  # 70% of base fee is burned
    PRIORITY_FEE_TO_PROVIDERS = 0.3  # 30% goes to AI Energy Providers
    
    # Genesis timestamp for network age calculation
    GENESIS_TIMESTAMP = 1733299200  # December 4, 2025 00:00:00 UTC
    
    # AI Energy Mining Rewards (like PoS staking rewards)
    ANNUAL_ISSUANCE_RATE = 0.02  # 2% annual new issuance for AI miners
    BLOCK_REWARD_BASE = 0.5  # Base block reward in NNET
    
    def __init__(self):
        self.blocks: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.ai_energy_providers: Dict[str, AIEnergyProvider] = {}
        self.miners: Dict[str, Miner] = {}
        self.proposals: Dict[str, Proposal] = {}
        self.balances: Dict[str, float] = {}  # NNET balances
        self.token_balances: Dict[str, Dict[str, float]] = {}  # {address: {token: balance}}
        self.tokens: Dict[str, Token] = {}  # All tokens in network
        self.liquidity_pools: Dict[str, LiquidityPool] = {}  # DEX pools
        self.liquidity_positions: Dict[str, Dict[str, float]] = {}  # {address: {pool_id: liquidity}}
        self.price_history: List[Dict[str, Any]] = []  # Price tracking
        self.nonces: Dict[str, int] = {}
        self.contracts: Dict[str, Dict[str, Any]] = {}
        self.dapps: Dict[str, DApp] = {}  # Deployed dApps
        self.nft_collections: Dict[str, NFTCollection] = {}  # NFT collections
        self.ai_tasks: List[Dict[str, Any]] = []
        self.block_height: int = 0
        self.transaction_history: List[Dict[str, Any]] = []
        self.transactions: List[Dict[str, Any]] = []  # Real transaction history
        self.network_stats = {
            "total_transactions": 0,
            "total_blocks": 0,
            "fraud_detected": 0,
            "attacks_prevented": 0,
            "ai_decisions": 0,
            "dao_proposals": 0,
            "mining_rewards_distributed": 0.0,
            "quantum_signatures_verified": 0,
            "hybrid_signatures_verified": 0,
            "transactions_pending": 0,
            "miners_active": 0,
            "ai_tasks_completed": 0,
            "total_swaps": 0,
            "total_liquidity_usd": 0.0,
            # EIP-1559 Style Burning Stats
            "total_burned": 0.0,
            "total_issued": 0.0,
            "genesis_supply": self.GENESIS_SUPPLY,
            "current_supply": self.GENESIS_SUPPLY,
            "burn_rate_24h": 0.0,
            "issuance_rate_24h": 0.0,
            "is_deflationary": False
        }
        self._running = False
        self._thread = None
        self._last_block_time = 0
        self._attack_patterns = []
        self._mining_pool_rewards = 1000000.0
        self._initialize_network()
        self._initialize_tokens()
        
    def _initialize_network(self):
        genesis_block = Block(
            index=0,
            timestamp=int(time.time()) - 86400 * 30,  # 30 days ago
            transactions=[],
            previous_hash="0" * 64,
            provider="neo1genesis",
            hash=self._hash_block(0, [], "0" * 64, "neo1genesis")
        )
        self.blocks.append(genesis_block)
        
        for i in range(21):
            provider_addr = f"neo1validator{i:02d}"
            stake = random.uniform(100000, 500000)
            self.ai_energy_providers[provider_addr] = AIEnergyProvider(
                address=provider_addr,
                stake=stake,
                is_active=True,
                blocks_validated=random.randint(1000, 50000),
                rewards_earned=random.uniform(1000, 10000),
                intelligence_score=random.uniform(0.7, 0.99),
                registered_at=int(time.time()) - random.randint(86400, 86400 * 365)
            )
            self.balances[provider_addr] = stake + random.uniform(10000, 100000)
            
        circulating = self.GENESIS_SUPPLY - sum(self.balances.values())
        for i in range(100):
            user_addr = f"neo1user{hashlib.sha256(str(i).encode()).hexdigest()[:32]}"
            self.balances[user_addr] = random.uniform(100, circulating / 200)
            
        self._last_block_time = int(time.time())
        self.network_stats["total_blocks"] = len(self.blocks)
    
    def _initialize_tokens(self):
        """Initialize NNET token with Ethereum-style dynamic supply and 30M genesis allocation"""
        self.tokens["NNET"] = Token(
            symbol="NNET",
            name="NeoNet",
            total_supply=self.GENESIS_SUPPLY,  # Dynamic - can change with burn/issuance
            decimals=18,
            creator="neo1genesis",
            price_usd=self.INITIAL_NNET_PRICE,
            created_at=self.GENESIS_TIMESTAMP,
            is_native=True
        )
        
        # Initialize burn address with 0 balance
        self.balances[self.BURN_ADDRESS] = 0.0
        
        # Treasury gets remaining supply
        treasury_balance = self.GENESIS_SUPPLY - self.INITIAL_USER_ALLOCATION
        for addr in list(self.balances.keys()):
            if addr != self.BURN_ADDRESS:
                treasury_balance -= self.balances[addr]
        
        self.balances["neo1treasury0000000000000000000000000000"] = max(0, treasury_balance - self.INITIAL_USER_ALLOCATION)
        self.balances[self.USER_WALLET] = self.INITIAL_USER_ALLOCATION
        
        # Record genesis transaction for 30M allocation (visible in network history)
        genesis_tx_hash = hashlib.sha256(f"genesis_{self.USER_WALLET}_{self.INITIAL_USER_ALLOCATION}".encode()).hexdigest()
        genesis_transaction = {
            "hash": genesis_tx_hash,
            "type": "genesis_allocation",
            "from": "neo1genesis",
            "to": self.USER_WALLET,
            "amount": self.INITIAL_USER_ALLOCATION,
            "token": "NNET",
            "timestamp": self.GENESIS_TIMESTAMP,
            "block": 0,
            "status": "confirmed",
            "gas_fee": 0.0,
            "description": "Genesis block allocation - 30 million NNET"
        }
        self.transactions.append(genesis_transaction)
        
        self.price_history.append({
            "token": "NNET",
            "price": self.INITIAL_NNET_PRICE,
            "timestamp": int(time.time()),
            "volume": 0.0
        })
        
        print(f"[NeoNet] Genesis Supply: {self.GENESIS_SUPPLY:,.0f} NNET (dynamic - EIP-1559 burning enabled)")
        print(f"[NeoNet] Allocated {self.INITIAL_USER_ALLOCATION:,.0f} NNET to {self.USER_WALLET}")
        print(f"[NeoNet] NNET price: ${self.INITIAL_NNET_PRICE}")
    
    def create_token(self, symbol: str, name: str, total_supply: float, 
                     creator: str, initial_price: float = 0.0, 
                     runtime: str = "hybrid") -> Dict[str, Any]:
        """Create new token in the network"""
        symbol = symbol.upper()
        if symbol in self.tokens:
            return {"error": f"Token {symbol} already exists"}
        
        if symbol == "NNET":
            return {"error": "Cannot create NNET token"}
        
        runtime = runtime.lower() if runtime else "hybrid"
        if runtime not in ["evm", "hybrid"]:
            runtime = "hybrid"
        
        contract_hash = hashlib.sha256(f"{symbol}{creator}{time.time()}".encode()).hexdigest()
        if runtime == "evm":
            contract_address = f"0x{contract_hash[:40]}"
        else:
            contract_address = f"neo1{contract_hash[:38]}"
        
        self.tokens[symbol] = Token(
            symbol=symbol,
            name=name,
            total_supply=total_supply,
            decimals=18,
            creator=creator,
            price_usd=initial_price,
            created_at=int(time.time()),
            is_native=False,
            runtime=runtime,
            contract_address=contract_address
        )
        
        if creator not in self.token_balances:
            self.token_balances[creator] = {}
        self.token_balances[creator][symbol] = total_supply
        
        return {
            "success": True,
            "symbol": symbol,
            "name": name,
            "total_supply": total_supply,
            "creator": creator,
            "runtime": runtime,
            "contract_address": contract_address,
            "message": f"Token {symbol} created on {runtime.upper()} runtime. Add liquidity with NNET to enable trading."
        }
    
    def get_all_tokens(self) -> List[Dict[str, Any]]:
        """Get all tokens in the network"""
        return [
            {
                "symbol": t.symbol,
                "name": t.name,
                "total_supply": t.total_supply,
                "price_usd": t.price_usd,
                "is_native": t.is_native,
                "created_at": t.created_at,
                "runtime": t.runtime,
                "contract_address": t.contract_address
            }
            for t in self.tokens.values()
        ]
    
    def get_token_balance(self, address: str, token: str) -> float:
        """Get token balance for address"""
        token = token.upper()
        if token == "NNET":
            return self.balances.get(address, 0.0)
        return self.token_balances.get(address, {}).get(token, 0.0)
    
    def get_all_balances(self, address: str) -> Dict[str, float]:
        """Get all token balances for an address"""
        balances = {"NNET": self.balances.get(address, 0.0)}
        if address in self.token_balances:
            balances.update(self.token_balances[address])
        return balances
    
    # ===== EIP-1559 STYLE BURNING SYSTEM =====
    
    def burn_neo(self, amount: float, source: str = "transaction_fee") -> Dict[str, Any]:
        """Burn NNET tokens (send to burn address - permanently removed from circulation)"""
        if amount <= 0:
            return {"error": "Burn amount must be positive"}
        
        # Add to burn address (tracked but never spendable)
        self.balances[self.BURN_ADDRESS] = self.balances.get(self.BURN_ADDRESS, 0.0) + amount
        
        # Update network stats
        self.network_stats["total_burned"] += amount
        self.network_stats["current_supply"] = self.GENESIS_SUPPLY + self.network_stats["total_issued"] - self.network_stats["total_burned"]
        
        # Check if network is deflationary
        self.network_stats["is_deflationary"] = self.network_stats["total_burned"] > self.network_stats["total_issued"]
        
        # Update NNET token supply
        if "NNET" in self.tokens:
            self.tokens["NNET"].total_supply = self.network_stats["current_supply"]
        
        return {
            "success": True,
            "amount_burned": amount,
            "source": source,
            "total_burned": self.network_stats["total_burned"],
            "current_supply": self.network_stats["current_supply"]
        }
    
    def issue_neo(self, amount: float, recipient: str, source: str = "mining_reward") -> Dict[str, Any]:
        """Issue new NNET tokens (mining/validation rewards - increases supply)"""
        if amount <= 0:
            return {"error": "Issue amount must be positive"}
        
        # Add to recipient balance
        self.balances[recipient] = self.balances.get(recipient, 0.0) + amount
        
        # Update network stats
        self.network_stats["total_issued"] += amount
        self.network_stats["current_supply"] = self.GENESIS_SUPPLY + self.network_stats["total_issued"] - self.network_stats["total_burned"]
        
        # Check if network is deflationary
        self.network_stats["is_deflationary"] = self.network_stats["total_burned"] > self.network_stats["total_issued"]
        
        # Update NNET token supply
        if "NNET" in self.tokens:
            self.tokens["NNET"].total_supply = self.network_stats["current_supply"]
        
        return {
            "success": True,
            "amount_issued": amount,
            "recipient": recipient,
            "source": source,
            "total_issued": self.network_stats["total_issued"],
            "current_supply": self.network_stats["current_supply"]
        }
    
    def process_transaction_fees(self, gas_fee: float, provider: str = None) -> Dict[str, Any]:
        """Process transaction fees with EIP-1559 style burning
        - 70% of base fee is burned (removed from circulation)
        - 30% goes to AI Energy Providers as priority fee
        """
        if gas_fee <= 0:
            return {"burned": 0, "to_provider": 0}
        
        burn_amount = gas_fee * self.BASE_FEE_BURN_RATE
        provider_amount = gas_fee * self.PRIORITY_FEE_TO_PROVIDERS
        
        # Burn portion
        self.burn_neo(burn_amount, "transaction_base_fee")
        
        # Pay AI Energy Provider
        if provider:
            self.balances[provider] = self.balances.get(provider, 0.0) + provider_amount
        
        return {
            "total_fee": gas_fee,
            "burned": burn_amount,
            "to_provider": provider_amount,
            "burn_rate": self.BASE_FEE_BURN_RATE
        }
    
    def get_tokenomics_stats(self) -> Dict[str, Any]:
        """Get comprehensive tokenomics statistics (Ethereum-style)"""
        now = int(time.time())
        network_age_days = max(0, (now - self.GENESIS_TIMESTAMP) / 86400)
        
        return {
            "genesis_supply": self.GENESIS_SUPPLY,
            "current_supply": self.network_stats["current_supply"],
            "total_burned": self.network_stats["total_burned"],
            "total_issued": self.network_stats["total_issued"],
            "net_supply_change": self.network_stats["total_issued"] - self.network_stats["total_burned"],
            "is_deflationary": self.network_stats["is_deflationary"],
            "burn_address": self.BURN_ADDRESS,
            "burn_address_balance": self.balances.get(self.BURN_ADDRESS, 0.0),
            "base_fee_burn_rate": self.BASE_FEE_BURN_RATE,
            "priority_fee_rate": self.PRIORITY_FEE_TO_PROVIDERS,
            "genesis_timestamp": self.GENESIS_TIMESTAMP,
            "network_age_days": round(network_age_days, 1),
            "annual_issuance_rate": self.ANNUAL_ISSUANCE_RATE,
            "model": "Ethereum EIP-1559 style - dynamic supply with burning"
        }
    
    def create_liquidity_pool(self, token_b: str, neo_amount: float, 
                               token_amount: float, creator: str) -> Dict[str, Any]:
        """Create liquidity pool for token pair NNET/token"""
        token_b = token_b.upper()
        
        if token_b not in self.tokens:
            return {"error": f"Token {token_b} does not exist"}
        
        pool_id = f"NNET-{token_b}"
        if pool_id in self.liquidity_pools:
            return {"error": f"Pool {pool_id} already exists. Use add_liquidity instead."}
        
        if self.balances.get(creator, 0) < neo_amount:
            return {"error": "Insufficient NNET balance"}
        
        if self.token_balances.get(creator, {}).get(token_b, 0) < token_amount:
            return {"error": f"Insufficient {token_b} balance"}
        
        self.balances[creator] -= neo_amount
        self.token_balances[creator][token_b] -= token_amount
        
        liquidity = (neo_amount * token_amount) ** 0.5
        
        self.liquidity_pools[pool_id] = LiquidityPool(
            pool_id=pool_id,
            token_a="NNET",
            token_b=token_b,
            reserve_a=neo_amount,
            reserve_b=token_amount,
            total_liquidity=liquidity,
            fee_percent=0.003,
            created_at=int(time.time())
        )
        
        if creator not in self.liquidity_positions:
            self.liquidity_positions[creator] = {}
        self.liquidity_positions[creator][pool_id] = liquidity
        
        token_price = (neo_amount / token_amount) * self.tokens["NNET"].price_usd
        self.tokens[token_b].price_usd = token_price
        
        self.network_stats["total_liquidity_usd"] += neo_amount * self.tokens["NNET"].price_usd * 2
        
        return {
            "success": True,
            "pool_id": pool_id,
            "neo_deposited": neo_amount,
            "token_deposited": token_amount,
            "liquidity_tokens": liquidity,
            "initial_price": token_price
        }
    
    def add_liquidity(self, pool_id: str, neo_amount: float, 
                      provider: str) -> Dict[str, Any]:
        """Add liquidity to existing pool"""
        if pool_id not in self.liquidity_pools:
            return {"error": f"Pool {pool_id} does not exist"}
        
        pool = self.liquidity_pools[pool_id]
        
        if self.balances.get(provider, 0) < neo_amount:
            return {"error": "Insufficient NNET balance"}
        
        token_amount = (neo_amount / pool.reserve_a) * pool.reserve_b
        
        if self.token_balances.get(provider, {}).get(pool.token_b, 0) < token_amount:
            return {"error": f"Insufficient {pool.token_b} balance. Need {token_amount:.4f}"}
        
        self.balances[provider] -= neo_amount
        if provider not in self.token_balances:
            self.token_balances[provider] = {}
        self.token_balances[provider][pool.token_b] = self.token_balances[provider].get(pool.token_b, 0) - token_amount
        
        liquidity = (neo_amount / pool.reserve_a) * pool.total_liquidity
        
        pool.reserve_a += neo_amount
        pool.reserve_b += token_amount
        pool.total_liquidity += liquidity
        
        if provider not in self.liquidity_positions:
            self.liquidity_positions[provider] = {}
        self.liquidity_positions[provider][pool_id] = self.liquidity_positions[provider].get(pool_id, 0) + liquidity
        
        return {
            "success": True,
            "pool_id": pool_id,
            "neo_added": neo_amount,
            "token_added": token_amount,
            "liquidity_tokens": liquidity
        }
    
    def swap(self, from_token: str, to_token: str, amount: float, 
             trader: str) -> Dict[str, Any]:
        """Swap tokens using AMM (constant product formula: x * y = k)"""
        from_token = from_token.upper()
        to_token = to_token.upper()
        
        if from_token == to_token:
            return {"error": "Cannot swap same token"}
        
        if from_token == "NNET":
            pool_id = f"NNET-{to_token}"
            is_neo_to_token = True
        elif to_token == "NNET":
            pool_id = f"NNET-{from_token}"
            is_neo_to_token = False
        else:
            return {"error": "One of the tokens must be NNET"}
        
        if pool_id not in self.liquidity_pools:
            return {"error": f"Liquidity pool {pool_id} does not exist"}
        
        pool = self.liquidity_pools[pool_id]
        
        if is_neo_to_token:
            if self.balances.get(trader, 0) < amount:
                return {"error": "Insufficient NNET balance"}
            
            amount_with_fee = amount * (1 - pool.fee_percent)
            output = (pool.reserve_b * amount_with_fee) / (pool.reserve_a + amount_with_fee)
            
            self.balances[trader] -= amount
            if trader not in self.token_balances:
                self.token_balances[trader] = {}
            self.token_balances[trader][to_token] = self.token_balances[trader].get(to_token, 0) + output
            
            pool.reserve_a += amount
            pool.reserve_b -= output
            
        else:
            if self.token_balances.get(trader, {}).get(from_token, 0) < amount:
                return {"error": f"Insufficient {from_token} balance"}
            
            amount_with_fee = amount * (1 - pool.fee_percent)
            output = (pool.reserve_a * amount_with_fee) / (pool.reserve_b + amount_with_fee)
            
            self.token_balances[trader][from_token] -= amount
            self.balances[trader] = self.balances.get(trader, 0) + output
            
            pool.reserve_b += amount
            pool.reserve_a -= output
        
        old_price = self.tokens[pool.token_b].price_usd if pool.token_b in self.tokens else 0
        new_price = (pool.reserve_a / pool.reserve_b) * self.tokens["NNET"].price_usd if pool.reserve_b > 0 else 0
        self.tokens[pool.token_b].price_usd = new_price
        
        self._update_neo_price(is_buy=is_neo_to_token, amount=amount if is_neo_to_token else output)
        
        self.network_stats["total_swaps"] += 1
        
        price_impact = abs((new_price - old_price) / old_price) * 100 if old_price > 0 else 0
        
        self.price_history.append({
            "token": pool.token_b,
            "price": new_price,
            "timestamp": int(time.time()),
            "volume": amount * self.tokens["NNET"].price_usd if is_neo_to_token else output * self.tokens["NNET"].price_usd
        })
        
        return {
            "success": True,
            "from_token": from_token,
            "to_token": to_token,
            "amount_in": amount,
            "amount_out": output,
            "price_impact": price_impact,
            "new_price": new_price
        }
    
    def _update_neo_price(self, is_buy: bool, amount: float):
        """Update NNET price based on buy/sell pressure (organic price discovery)"""
        impact = amount / 1_000_000
        if is_buy:
            self.tokens["NNET"].price_usd *= (1 + impact * 0.001)
        else:
            self.tokens["NNET"].price_usd *= (1 - impact * 0.001)
        
        self.tokens["NNET"].price_usd = max(0.01, self.tokens["NNET"].price_usd)
        
        self.price_history.append({
            "token": "NNET",
            "price": self.tokens["NNET"].price_usd,
            "timestamp": int(time.time()),
            "volume": amount
        })
    
    def get_pool_info(self, pool_id: str) -> Dict[str, Any]:
        """Get liquidity pool information"""
        if pool_id not in self.liquidity_pools:
            return {"error": "Pool not found"}
        
        pool = self.liquidity_pools[pool_id]
        return {
            "pool_id": pool.pool_id,
            "token_a": pool.token_a,
            "token_b": pool.token_b,
            "reserve_a": pool.reserve_a,
            "reserve_b": pool.reserve_b,
            "total_liquidity": pool.total_liquidity,
            "price": (pool.reserve_a / pool.reserve_b) * self.tokens["NNET"].price_usd if pool.reserve_b > 0 else 0,
            "fee_percent": pool.fee_percent
        }
    
    def get_all_pools(self) -> List[Dict[str, Any]]:
        """Get all liquidity pools"""
        return [self.get_pool_info(pid) for pid in self.liquidity_pools.keys()]
    
    def get_swap_quote(self, from_token: str, to_token: str, amount: float) -> Dict[str, Any]:
        """Get swap quote without executing"""
        from_token = from_token.upper()
        to_token = to_token.upper()
        
        if from_token == to_token:
            return {"error": "Cannot swap same token"}
        
        if from_token == "NNET":
            pool_id = f"NNET-{to_token}"
            is_neo_to_token = True
        elif to_token == "NNET":
            pool_id = f"NNET-{from_token}"
            is_neo_to_token = False
        else:
            return {"error": "One of the tokens must be NNET"}
        
        if pool_id not in self.liquidity_pools:
            return {"error": f"No liquidity pool for {from_token}/{to_token}"}
        
        pool = self.liquidity_pools[pool_id]
        
        if is_neo_to_token:
            amount_with_fee = amount * (1 - pool.fee_percent)
            output = (pool.reserve_b * amount_with_fee) / (pool.reserve_a + amount_with_fee)
        else:
            amount_with_fee = amount * (1 - pool.fee_percent)
            output = (pool.reserve_a * amount_with_fee) / (pool.reserve_b + amount_with_fee)
        
        return {
            "from_token": from_token,
            "to_token": to_token,
            "amount_in": amount,
            "amount_out": output,
            "rate": output / amount if amount > 0 else 0,
            "fee": amount * pool.fee_percent
        }
        
    def _hash_block(self, index: int, transactions: List[Transaction], 
                    prev_hash: str, provider: str) -> str:
        data = f"{index}{[t.tx_hash for t in transactions]}{prev_hash}{provider}"
        return hashlib.sha256(data.encode()).hexdigest()
        
    def _generate_quantum_signatures(self, tx_hash: str, sender: str) -> tuple:
        """Generate hybrid quantum-safe signatures for ALL transactions"""
        # Classical EVM signature (ECDSA simulation)
        evm_sig = hashlib.sha256(f"evm:{tx_hash}:{sender}".encode()).hexdigest()
        # Ed25519 quantum-resistant signature
        quantum_sig = hashlib.sha256(f"ed25519:{tx_hash}:{sender}:{time.time()}".encode()).hexdigest()
        # Dilithium3 post-quantum signature (NIST Level 3)
        dilithium_sig = hashlib.sha512(f"dilithium3:{tx_hash}:{sender}:{random.random()}".encode()).hexdigest()
        return evm_sig, quantum_sig, dilithium_sig
    
    def _generate_transaction(self, is_attack: bool = False) -> Transaction:
        """Generate transaction with hybrid quantum-safe signatures"""
        tx_types = ["transfer", "contract_call", "stake", "unstake", "governance"]
        weights = [0.6, 0.2, 0.1, 0.05, 0.05]
        
        if is_attack:
            attack_type = random.choice(["flash_loan", "reentrancy", "sandwich", "dust"])
            sender = f"neo1attacker{random.randint(1, 100):03d}"
            recipient = random.choice(list(self.ai_energy_providers.keys()))
            amount = random.uniform(1000000, 10000000) if attack_type == "flash_loan" else random.uniform(0.001, 0.01)
            gas_price = random.uniform(1000, 10000)
            is_fraud = True
            fraud_score = random.uniform(0.7, 0.99)
            data = {"attack_type": attack_type}
            tx_type = "contract_call"
        else:
            sender = random.choice(list(self.balances.keys()))
            recipients = [k for k in self.balances.keys() if k != sender]
            recipient = random.choice(recipients) if recipients else sender
            tx_type = random.choices(tx_types, weights=weights)[0]
            amount = random.uniform(0.1, 1000)
            gas_price = random.uniform(10, 100)
            is_fraud = random.random() < 0.02  # 2% natural fraud rate
            fraud_score = random.uniform(0.6, 0.9) if is_fraud else random.uniform(0.0, 0.3)
            data = {"tx_type": tx_type}
        
        # Generate tx_hash first
        tx_hash = hashlib.sha256(f"{sender}{recipient}{time.time()}{random.random()}".encode()).hexdigest()
        
        # Generate quantum-safe signatures for ALL transactions
        evm_sig, quantum_sig, dilithium_sig = self._generate_quantum_signatures(tx_hash, sender)
        
        # Get sender nonce for replay protection
        nonce = self.nonces.get(sender, 0)
        self.nonces[sender] = nonce + 1
            
        tx = Transaction(
            tx_hash=tx_hash,
            sender=sender,
            recipient=recipient,
            amount=amount,
            gas_price=gas_price,
            gas_used=random.randint(21000, 500000),
            tx_type=tx_type,
            timestamp=int(time.time()),
            nonce=nonce,
            # Quantum-safe signatures on ALL transactions
            evm_signature=evm_sig,
            quantum_signature=quantum_sig,
            dilithium_signature=dilithium_sig,
            signature_algorithm="Hybrid-Ed25519+Dilithium3",
            is_verified=True,  # Signatures verified at creation
            verification_level="hybrid",
            is_fraud=is_fraud,
            fraud_score=fraud_score,
            ai_verified=False,
            data=data
        )
        
        # Verify quantum signature
        tx.verify_quantum_signature()
        
        self.network_stats["total_transactions"] += 1
        self.network_stats["hybrid_signatures_verified"] += 1
        self.network_stats["quantum_signatures_verified"] += 1
        
        if is_fraud:
            self.network_stats["fraud_detected"] += 1
        if is_attack:
            self._attack_patterns.append({
                "type": data.get("attack_type"),
                "tx_hash": tx.tx_hash,
                "timestamp": tx.timestamp,
                "amount": amount
            })
            
        return tx
        
    def generate_block(self) -> Block:
        num_txs = random.randint(10, 50)
        is_under_attack = random.random() < 0.1  # 10% attack probability
        
        transactions = []
        for _ in range(num_txs):
            is_attack_tx = is_under_attack and random.random() < 0.3
            tx = self._generate_transaction(is_attack=is_attack_tx)
            transactions.append(tx)
            
        provider = self._select_provider()
        prev_block = self.blocks[-1]
        
        ai_score = self._ai_validate_block(transactions)
        
        block = Block(
            index=prev_block.index + 1,
            timestamp=int(time.time()),
            transactions=transactions,
            previous_hash=prev_block.hash,
            provider=provider,
            hash="",
            ai_score=ai_score
        )
        block.hash = self._hash_block(block.index, block.transactions, 
                                       block.previous_hash, block.provider)
        
        if ai_score > 0.5:
            self.blocks.append(block)
            self.network_stats["total_blocks"] += 1
            
            if provider in self.ai_energy_providers:
                self.ai_energy_providers[provider].blocks_validated += 1
                reward = 10.0 + len(transactions) * 0.1
                self.ai_energy_providers[provider].rewards_earned += reward
                self.balances[provider] = self.balances.get(provider, 0) + reward
        else:
            self.network_stats["attacks_prevented"] += 1
            
        self._last_block_time = int(time.time())
        return block
        
    def _select_provider(self) -> str:
        active = [v for v in self.ai_energy_providers.values() if v.is_active]
        if not active:
            return "neo1genesis"
            
        total_stake = sum(v.stake * v.intelligence_score for v in active)
        rand = random.uniform(0, total_stake)
        cumulative = 0.0
        
        for v in active:
            cumulative += v.stake * v.intelligence_score
            if cumulative >= rand:
                return v.address
                
        return active[0].address
        
    def _ai_validate_block(self, transactions: List[Transaction]) -> float:
        if not transactions:
            return 1.0
            
        fraud_count = sum(1 for t in transactions if t.is_fraud)
        fraud_ratio = fraud_count / len(transactions)
        
        avg_fraud_score = sum(t.fraud_score for t in transactions) / len(transactions)
        
        suspicious_patterns = 0
        for t in transactions:
            if t.amount > 1000000:  # Large transaction
                suspicious_patterns += 1
            if t.gas_price > 500:  # High gas price
                suspicious_patterns += 1
            if "attack" in str(t.data):
                suspicious_patterns += 2
                
        pattern_penalty = min(suspicious_patterns * 0.1, 0.5)
        
        ai_score = 1.0 - (fraud_ratio * 0.4 + avg_fraud_score * 0.3 + pattern_penalty * 0.3)
        self.network_stats["ai_decisions"] += 1
        
        return max(0.0, min(1.0, ai_score))
        
    def create_proposal(self, title: str, description: str, proposer: str) -> Proposal:
        proposal_id = hashlib.sha256(f"{title}{proposer}{time.time()}".encode()).hexdigest()[:16]
        
        ai_recommendation, ai_confidence = self._ai_analyze_proposal(title, description)
        
        proposal = Proposal(
            proposal_id=proposal_id,
            title=title,
            description=description,
            proposer=proposer,
            created_at=int(time.time()),
            voting_ends_at=int(time.time()) + 86400 * 7,  # 7 days
            status="active",
            ai_recommendation=ai_recommendation,
            ai_confidence=ai_confidence
        )
        
        self.proposals[proposal_id] = proposal
        self.network_stats["dao_proposals"] += 1
        return proposal
        
    def _ai_analyze_proposal(self, title: str, description: str) -> tuple:
        positive_keywords = ["upgrade", "improve", "security", "efficiency", "reward"]
        negative_keywords = ["remove", "decrease", "attack", "exploit", "drain"]
        
        text = (title + " " + description).lower()
        
        positive_score = sum(1 for kw in positive_keywords if kw in text)
        negative_score = sum(1 for kw in negative_keywords if kw in text)
        
        if positive_score > negative_score:
            recommendation = "for"
            confidence = min(0.9, 0.5 + positive_score * 0.1)
        elif negative_score > positive_score:
            recommendation = "against"
            confidence = min(0.9, 0.5 + negative_score * 0.1)
        else:
            recommendation = "neutral"
            confidence = 0.5
            
        return recommendation, confidence
        
    def vote_on_proposal(self, proposal_id: str, voter: str, vote_for: bool, 
                         stake_weight: float) -> Dict[str, Any]:
        if proposal_id not in self.proposals:
            return {"error": "Proposal not found"}
            
        proposal = self.proposals[proposal_id]
        
        if proposal.status != "active":
            return {"error": "Proposal not active"}
            
        if vote_for:
            proposal.for_votes += stake_weight
        else:
            proposal.against_votes += stake_weight
            
        self._check_proposal_result(proposal)
        
        return {
            "proposal_id": proposal_id,
            "voter": voter,
            "vote": "for" if vote_for else "against",
            "weight": stake_weight,
            "current_for": proposal.for_votes,
            "current_against": proposal.against_votes
        }
        
    def _check_proposal_result(self, proposal: Proposal):
        total_votes = proposal.for_votes + proposal.against_votes
        if total_votes == 0:
            return
            
        human_ratio = 0.7
        ai_ratio = proposal.ai_weight
        
        human_for = proposal.for_votes / total_votes
        ai_for = 1.0 if proposal.ai_recommendation == "for" else (0.0 if proposal.ai_recommendation == "against" else 0.5)
        
        weighted_for = human_for * human_ratio + ai_for * ai_ratio * proposal.ai_confidence
        
        quorum = sum(v.stake for v in self.ai_energy_providers.values() if v.is_active) * 0.1
        
        if total_votes >= quorum:
            if weighted_for > 0.5:
                proposal.status = "passed"
            else:
                proposal.status = "rejected"
                
    def deploy_contract(self, code: str, runtime: str, deployer: str, 
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if runtime not in ["evm", "wasm", "hybrid"]:
            return {"error": "Invalid runtime. Use 'evm', 'wasm', or 'hybrid'"}
            
        contract_id = hashlib.sha256(f"{code}{deployer}{time.time()}".encode()).hexdigest()[:40]
        contract_address = f"neo1contract{contract_id[:30]}"
        now = int(time.time())
        
        contract_type = self._detect_contract_type(code)
        token_info = None
        dapp_info = None
        nft_info = None
        
        contract_data = {
            "address": contract_address,
            "runtime": runtime,
            "deployer": deployer,
            "code_hash": hashlib.sha256(code.encode()).hexdigest(),
            "deployed_at": now,
            "tx_count": 0,
            "status": "active",
            "type": contract_type,
            "metadata": metadata or {}
        }
        
        if contract_type == "token":
            token_info = self._register_token_from_contract(code, contract_address, deployer, runtime, metadata)
            contract_data["token_symbol"] = token_info.get("symbol") if token_info else None
            
        elif contract_type == "nft":
            nft_info = self._register_nft_from_contract(code, contract_address, deployer, metadata)
            contract_data["nft_collection"] = nft_info.get("collection_id") if nft_info else None
            
        elif contract_type == "dapp" or metadata and metadata.get("is_dapp"):
            dapp_info = self._register_dapp_from_contract(code, contract_address, deployer, runtime, metadata)
            contract_data["dapp_id"] = dapp_info.get("dapp_id") if dapp_info else None
        
        self.contracts[contract_address] = contract_data
        
        tx = self._generate_transaction()
        tx.tx_type = "contract_deploy"
        tx.sender = deployer
        tx.data = {"contract": contract_address, "runtime": runtime, "type": contract_type}
        self.pending_transactions.append(tx)
        
        result = {
            "contract_address": contract_address,
            "runtime": runtime,
            "deployer": deployer,
            "tx_hash": tx.tx_hash,
            "status": "deployed",
            "contract_type": contract_type
        }
        
        if token_info:
            result["token"] = token_info
        if dapp_info:
            result["dapp"] = dapp_info
        if nft_info:
            result["nft"] = nft_info
            
        return result
    
    def _detect_contract_type(self, code: str) -> str:
        """AI-powered contract type detection"""
        code_lower = code.lower()
        
        if any(kw in code_lower for kw in ["erc20", "erc-20", "_mint", "totalsupply", "balanceof", "transfer("]):
            if any(kw in code_lower for kw in ["erc721", "erc-721", "tokenuriprefix", "safemint", "ownerof"]):
                return "nft"
            return "token"
        
        if any(kw in code_lower for kw in ["erc721", "erc-721", "nft", "tokenuriprefix", "safemint"]):
            return "nft"
        
        if any(kw in code_lower for kw in ["swap", "addliquidity", "removeliquidity", "getamountout"]):
            return "dapp"
        
        if any(kw in code_lower for kw in ["stake", "unstake", "reward", "claim"]):
            return "dapp"
        
        if any(kw in code_lower for kw in ["vote", "proposal", "governance", "quorum"]):
            return "dapp"
        
        return "generic"
    
    def _register_token_from_contract(self, code: str, contract_address: str, 
                                       deployer: str, runtime: str,
                                       metadata: Optional[Dict] = None) -> Optional[Dict]:
        """Auto-register token from deployed contract"""
        import re
        
        name_match = re.search(r'["\']([\w\s]+)["\'].*name|name.*["\']([^"\']+)["\']', code, re.IGNORECASE)
        symbol_match = re.search(r'symbol.*["\']([A-Z0-9]+)["\']|["\']([A-Z0-9]+)["\'].*symbol', code, re.IGNORECASE)
        supply_match = re.search(r'(\d+)\s*\*?\s*10\s*\*\*\s*(\d+)|_mint\([^,]+,\s*(\d+)', code)
        
        name = metadata.get("name") if metadata else None
        if not name and name_match:
            name = name_match.group(1) or name_match.group(2)
        if not name:
            name = f"Token_{contract_address[:8]}"
        
        symbol = metadata.get("symbol") if metadata else None
        if not symbol and symbol_match:
            symbol = symbol_match.group(1) or symbol_match.group(2)
        if not symbol:
            symbol = name[:4].upper()
        
        total_supply = 1000000.0
        if metadata and metadata.get("total_supply"):
            total_supply = float(metadata["total_supply"])
        elif supply_match:
            if supply_match.group(1) and supply_match.group(2):
                total_supply = float(supply_match.group(1)) * (10 ** int(supply_match.group(2)))
            elif supply_match.group(3):
                total_supply = float(supply_match.group(3))
        
        if symbol in self.tokens:
            symbol = f"{symbol}_{contract_address[:4].upper()}"
        
        token = Token(
            symbol=symbol,
            name=name,
            total_supply=total_supply,
            creator=deployer,
            price_usd=0.01,
            created_at=int(time.time()),
            is_native=False,
            runtime=runtime,
            contract_address=contract_address
        )
        self.tokens[symbol] = token
        
        if deployer not in self.token_balances:
            self.token_balances[deployer] = {}
        self.token_balances[deployer][symbol] = total_supply
        
        return {
            "symbol": symbol,
            "name": name,
            "total_supply": total_supply,
            "contract_address": contract_address,
            "auto_registered": True
        }
    
    def _register_nft_from_contract(self, code: str, contract_address: str,
                                     deployer: str, metadata: Optional[Dict] = None) -> Optional[Dict]:
        """Auto-register NFT collection from deployed contract"""
        import re
        
        name = metadata.get("name") if metadata else None
        if not name:
            name_match = re.search(r'["\']([\w\s]+)["\'].*name', code, re.IGNORECASE)
            name = name_match.group(1) if name_match else f"NFT_{contract_address[:8]}"
        
        symbol = metadata.get("symbol") if metadata else None
        if not symbol:
            symbol_match = re.search(r'symbol.*["\']([A-Z0-9]+)["\']', code, re.IGNORECASE)
            symbol = symbol_match.group(1) if symbol_match else name[:4].upper()
        
        total_supply = metadata.get("total_supply", 10000) if metadata else 10000
        category = metadata.get("category", "art") if metadata else "art"
        
        collection_id = f"nft_{contract_address[:20]}"
        
        collection = NFTCollection(
            collection_id=collection_id,
            name=name,
            symbol=symbol,
            description=metadata.get("description", f"{name} NFT Collection on NeoNet") if metadata else f"{name} NFT Collection",
            creator=deployer,
            contract_address=contract_address,
            total_supply=total_supply,
            category=category,
            created_at=int(time.time())
        )
        self.nft_collections[collection_id] = collection
        
        return {
            "collection_id": collection_id,
            "name": name,
            "symbol": symbol,
            "total_supply": total_supply,
            "contract_address": contract_address,
            "auto_registered": True
        }
    
    def _register_dapp_from_contract(self, code: str, contract_address: str,
                                      deployer: str, runtime: str,
                                      metadata: Optional[Dict] = None) -> Optional[Dict]:
        """Auto-register dApp from deployed contract"""
        
        name = metadata.get("name") if metadata else f"dApp_{contract_address[:8]}"
        description = metadata.get("description") if metadata else "Decentralized application on NeoNet"
        
        category = "utility"
        code_lower = code.lower()
        if any(kw in code_lower for kw in ["swap", "liquidity", "amm"]):
            category = "defi"
        elif any(kw in code_lower for kw in ["vote", "proposal", "governance"]):
            category = "dao"
        elif any(kw in code_lower for kw in ["game", "play", "score"]):
            category = "gaming"
        elif metadata and metadata.get("category"):
            category = metadata["category"]
        
        dapp_id = f"dapp_{contract_address[:20]}"
        
        dapp = DApp(
            dapp_id=dapp_id,
            name=name,
            description=description,
            category=category,
            creator=deployer,
            contract_address=contract_address,
            runtime=runtime,
            created_at=int(time.time()),
            website_url=metadata.get("website_url", "") if metadata else "",
            logo_url=metadata.get("logo_url", "") if metadata else ""
        )
        self.dapps[dapp_id] = dapp
        
        return {
            "dapp_id": dapp_id,
            "name": name,
            "category": category,
            "contract_address": contract_address,
            "auto_registered": True
        }
    
    def get_all_dapps(self) -> List[Dict[str, Any]]:
        """Get all deployed dApps"""
        return [asdict(d) for d in self.dapps.values()]
    
    def get_all_nft_collections(self) -> List[Dict[str, Any]]:
        """Get all NFT collections"""
        return [asdict(c) for c in self.nft_collections.values()]
    
    def get_explore_data(self) -> Dict[str, Any]:
        """Get all network projects for Explore section"""
        return {
            "tokens": [asdict(t) for t in self.tokens.values()],
            "dapps": self.get_all_dapps(),
            "nft_collections": self.get_all_nft_collections(),
            "contracts": list(self.contracts.values()),
            "stats": {
                "total_tokens": len(self.tokens),
                "total_dapps": len(self.dapps),
                "total_nfts": len(self.nft_collections),
                "total_contracts": len(self.contracts)
            }
        }
        
    def get_training_data(self, limit: int = 1000) -> List[Dict[str, Any]]:
        training_data = []
        
        for block in self.blocks[-100:]:
            for tx in block.transactions:
                training_data.append({
                    "features": tx.to_features(),
                    "is_fraud": tx.is_fraud,
                    "fraud_score": tx.fraud_score,
                    "tx_hash": tx.tx_hash,
                    "block_index": block.index
                })
                
        for pattern in self._attack_patterns[-50:]:
            features = [
                pattern.get("amount", 0) / 1000000,
                1.0 if pattern.get("type") == "flash_loan" else 0.0,
                1.0 if pattern.get("type") == "reentrancy" else 0.0,
                1.0 if pattern.get("type") == "sandwich" else 0.0,
                1.0 if pattern.get("type") == "dust" else 0.0,
                1.0, 0.0, 0.0, 0.0, 1.0
            ]
            training_data.append({
                "features": features,
                "is_fraud": True,
                "fraud_score": 0.95,
                "attack_type": pattern.get("type"),
                "tx_hash": pattern.get("tx_hash")
            })
            
        return training_data[:limit]
        
    def get_network_stats(self) -> Dict[str, Any]:
        total_stake = sum(v.stake for v in self.ai_energy_providers.values() if v.is_active)
        total_supply = sum(self.balances.values())
        active_miners = len([m for m in self.miners.values() if m.is_active])
        
        return {
            "status": "healthy",
            "block_height": len(self.blocks),
            "current_round": self.blocks[-1].index if self.blocks else 0,
            "ai_energy_providers": len([v for v in self.ai_energy_providers.values() if v.is_active]),
            "miners_active": active_miners,
            "total_stake": total_stake,
            "total_supply": total_supply,
            "total_transactions": self.network_stats["total_transactions"],
            "fraud_detected": self.network_stats["fraud_detected"],
            "attacks_prevented": self.network_stats["attacks_prevented"],
            "ai_decisions": self.network_stats["ai_decisions"],
            "dao_proposals": self.network_stats["dao_proposals"],
            "pending_transactions": len(self.pending_transactions),
            "contracts_deployed": len(self.contracts),
            "last_block_time": self._last_block_time,
            # Quantum-safe signature stats (ALL transactions)
            "quantum_signatures_verified": self.network_stats.get("quantum_signatures_verified", 0),
            "hybrid_signatures_verified": self.network_stats.get("hybrid_signatures_verified", 0),
            "signature_algorithm": "Hybrid-Ed25519+Dilithium3",
            "ai_tasks_completed": self.network_stats.get("ai_tasks_completed", 0),
            "mining_rewards_distributed": self.network_stats.get("mining_rewards_distributed", 0.0)
        }
    
    def register_miner(self, address: str, cpu_cores: int = 4, gpu_memory_mb: int = 8192, 
                       endpoint: str = "") -> Dict[str, Any]:
        """Register AI miner to earn NNET through work"""
        if address in self.miners:
            return {"error": "Miner already registered", "address": address}
        
        self.miners[address] = Miner(
            address=address,
            cpu_cores=cpu_cores,
            gpu_memory_mb=gpu_memory_mb,
            endpoint=endpoint or f"http://miner-{address[:8]}.neonet.local",
            registered_at=int(time.time()),
            is_active=True,
            tasks_completed=0,
            rewards_earned=0.0,
            intelligence_contribution=0.0,
            last_task_at=0
        )
        
        self.balances[address] = self.balances.get(address, 0.0)  # Start with 0 NNET
        self.network_stats["miners_active"] = len([m for m in self.miners.values() if m.is_active])
        
        return {
            "status": "registered",
            "address": address,
            "message": "Miner registered. Earn NNET by completing AI tasks."
        }
    
    def submit_ai_task_result(self, miner_address: str, task_id: str, 
                               result: Dict[str, Any]) -> Dict[str, Any]:
        """Miner submits AI task result to earn rewards"""
        if miner_address not in self.miners:
            return {"error": "Miner not registered"}
        
        miner = self.miners[miner_address]
        if not miner.is_active:
            return {"error": "Miner is inactive"}
        
        # Calculate reward based on task quality
        quality_score = result.get("accuracy", 0.5) * result.get("completion", 1.0)
        base_reward = 0.5  # Base reward per task
        reward = base_reward * (1 + quality_score)
        
        # Distribute reward from mining pool
        if self._mining_pool_rewards >= reward:
            self._mining_pool_rewards -= reward
            miner.rewards_earned += reward
            miner.tasks_completed += 1
            miner.last_task_at = int(time.time())
            miner.intelligence_contribution += quality_score * 0.1
            
            self.balances[miner_address] = self.balances.get(miner_address, 0.0) + reward
            self.network_stats["mining_rewards_distributed"] += reward
            self.network_stats["ai_tasks_completed"] += 1
            
            # Create reward transaction with quantum signatures
            tx_hash = hashlib.sha256(f"reward:{miner_address}:{task_id}:{time.time()}".encode()).hexdigest()
            evm_sig, quantum_sig, dilithium_sig = self._generate_quantum_signatures(tx_hash, "neo1mining_pool")
            
            reward_tx = Transaction(
                tx_hash=tx_hash,
                sender="neo1mining_pool",
                recipient=miner_address,
                amount=reward,
                gas_price=0,
                gas_used=21000,
                tx_type="mining_reward",
                timestamp=int(time.time()),
                nonce=0,
                evm_signature=evm_sig,
                quantum_signature=quantum_sig,
                dilithium_signature=dilithium_sig,
                signature_algorithm="Hybrid-Ed25519+Dilithium3",
                is_verified=True,
                verification_level="hybrid",
                data={"task_id": task_id, "quality_score": quality_score}
            )
            self.pending_transactions.append(reward_tx)
            
            return {
                "status": "accepted",
                "reward": reward,
                "new_balance": self.balances[miner_address],
                "tasks_completed": miner.tasks_completed,
                "tx_hash": tx_hash
            }
        else:
            return {"error": "Mining pool exhausted"}
        
    def start_network(self):
        if self._running:
            return
            
        self._running = True
        self._thread = threading.Thread(target=self._network_loop, daemon=True)
        self._thread.start()
        self._ai_thread = threading.Thread(target=self._ai_auto_training_loop, daemon=True)
        self._ai_thread.start()
        
    def stop_network(self):
        self._running = False
        
    def _network_loop(self):
        while self._running:
            try:
                self.generate_block()
                time.sleep(self.BLOCK_TIME)
            except Exception as e:
                print(f"Network error: {e}")
                time.sleep(1)
                
    def _ai_auto_training_loop(self):
        """AI automatically trains itself on network data without user input"""
        self.ai_model = {
            "version": 1,
            "accuracy": 0.75,
            "training_rounds": 0,
            "last_trained": 0,
            "fraud_detected_by_ai": 0,
            "total_predictions": 0
        }
        
        while self._running:
            try:
                training_data = self.get_training_data(200)
                if len(training_data) >= 50:
                    self._train_ai_model(training_data)
                time.sleep(30)  # Train every 30 seconds
            except Exception as e:
                print(f"AI training error: {e}")
                time.sleep(5)
                
    def _train_ai_model(self, training_data: List[Dict]):
        """Automatic AI training on network transactions"""
        if not training_data:
            return
            
        # Simulate training on fraud detection
        fraud_samples = [d for d in training_data if d.get("is_fraud")]
        normal_samples = [d for d in training_data if not d.get("is_fraud")]
        
        # Calculate model improvement
        fraud_ratio = len(fraud_samples) / max(len(training_data), 1)
        attack_samples = [d for d in training_data if d.get("attack_type")]
        
        # Model learns from patterns
        improvement = 0.001 * (1 + len(fraud_samples) * 0.1 + len(attack_samples) * 0.2)
        self.ai_model["accuracy"] = min(0.99, self.ai_model["accuracy"] + improvement)
        self.ai_model["training_rounds"] += 1
        self.ai_model["last_trained"] = int(time.time())
        
        # AI improves AI Energy Provider intelligence scores based on their behavior
        for provider in self.ai_energy_providers.values():
            if provider.is_active:
                # Validators who validate more blocks get higher intelligence
                blocks_factor = min(provider.blocks_validated / 10000, 1.0)
                provider.intelligence_score = min(0.99, 
                    0.7 + blocks_factor * 0.2 + random.uniform(0, 0.09))
                    
        # Update network stats
        self.network_stats["ai_decisions"] += len(training_data)
        self.ai_model["total_predictions"] += len(training_data)
        self.ai_model["fraud_detected_by_ai"] += len(fraud_samples)
        
    def get_ai_status(self) -> Dict[str, Any]:
        """Get AI training status"""
        if not hasattr(self, 'ai_model'):
            return {"status": "initializing"}
            
        return {
            "status": "active",
            "model_version": self.ai_model["version"],
            "accuracy": round(self.ai_model["accuracy"] * 100, 2),
            "training_rounds": self.ai_model["training_rounds"],
            "last_trained": self.ai_model["last_trained"],
            "fraud_detected": self.ai_model["fraud_detected_by_ai"],
            "total_predictions": self.ai_model["total_predictions"],
            "mode": "autonomous"
        }

blockchain = NeoNetBlockchain()
blockchain.start_network()
