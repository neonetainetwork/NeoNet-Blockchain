# NeoNet Whitepaper v2.0

## AI-Powered Web4 Blockchain with Proof of Intelligence

**Version**: 2.0  
**Date**: December 2025  
**Ticker**: NNET  

---

## Abstract

NeoNet is a revolutionary Web4 blockchain platform that implements Proof of Intelligence (PoI) consensus, where AI performs all computational work while users simply provide computing power. The network features quantum-safe cryptography, dynamic reward distribution, adaptive security, and a novel decentralization model that transitions from centralized bootstrap to fully distributed operation.

---

## 1. Introduction

### 1.1 The Problem

Traditional blockchain networks face several challenges:
- **Energy Waste**: Proof of Work consumes massive amounts of electricity for hash puzzles
- **Centralization**: Mining pools control most hash power
- **Quantum Vulnerability**: Classical cryptography will be broken by quantum computers
- **Useless Computation**: Mining work produces no practical value
- **High Barrier to Entry**: Expensive hardware requirements exclude average users

### 1.2 The NeoNet Solution

NeoNet solves these problems through:
- **Proof of Intelligence**: AI does meaningful work (fraud detection, model training)
- **Inclusive Mining**: Even weak laptops can participate and earn NNET
- **Quantum-Safe Signatures**: Ed25519 + Dilithium3 hybrid protection
- **Dynamic Rewards**: Fair distribution based on active participants
- **Progressive Decentralization**: Network transitions from bootstrap to fully distributed

---

## 2. Proof of Intelligence (PoI) Consensus

### 2.1 Core Concept

In NeoNet, users don't perform computations themselves - **AI does ALL the work**:

```
Traditional Mining:
  User Hardware → Solve Hash Puzzle → Get Reward

NeoNet PoI:
  User Hardware → Provide Energy → AI Does Work → Get Reward
```

### 2.2 AI Task Types

| Task Type | Intensity | Reward | Hardware Required |
|-----------|-----------|--------|-------------------|
| fraud_check | Low | 0.50 NNET | 1+ CPU, 0 GPU |
| validate | Low | 0.30 NNET | 1+ CPU, 0 GPU |
| inference | Low | 0.40 NNET | 1+ CPU, 0 GPU |
| aggregate | Medium | 0.30 NNET | 2+ CPU, 0 GPU |
| training_light | Medium | 0.50 NNET | 4+ CPU, 2GB GPU |
| training | High | 0.80 NNET | 8+ CPU, 8GB GPU |
| federated_learning | Extreme | 1.00 NNET | 8+ CPU, 16GB GPU |

### 2.3 Task Delegation

AI automatically assigns tasks based on hardware capabilities:

```
Weak Laptop (1 CPU, no GPU):
  → fraud_check, validate, inference

Standard PC (4 CPU, 2GB GPU):
  → aggregate, training_light

Powerful GPU (8+ CPU, 8GB+ GPU):
  → training, federated_learning
```

---

## 3. Dynamic Reward System

### 3.1 Why Dynamic Rewards?

**Problem**: If you pay a fixed reward (e.g., 1 NNET per task), and 1 million miners join, the token hyperinflates.

**Solution**: "We pay not for the fact of work, but we divide a common pie. The pie is always the same size (no inflation), but there can be many eaters."

### 3.2 Formula

```
Reward = Task_Weight × (Block_Budget / Active_Miners)

Where:
- Block_Budget = 10.0 NNET (fixed "pie" per block, every ~3 seconds)
- Active_Miners = Current active energy providers (minimum 1)
- Task_Weight = 0.30 to 1.00 (based on task difficulty)

Caps:
- Minimum: 0.001 NNET (prevents dust)
- Maximum: 10.0 NNET (when alone)
```

### 3.3 Task Weights

| Task Type | Weight | Example (10 miners) |
|-----------|--------|---------------------|
| federated_learning | 1.00 | 1.00 NNET |
| model_training | 0.80 | 0.80 NNET |
| network_protection | 0.60 | 0.60 NNET |
| fraud_detection | 0.50 | 0.50 NNET |
| gradient_compute | 0.50 | 0.50 NNET |
| inference | 0.40 | 0.40 NNET |
| data_validation | 0.30 | 0.30 NNET |
| matrix_ops | 0.30 | 0.30 NNET |

### 3.4 Reward Examples

| Active Miners | Base Rate | federated_learning | fraud_detection |
|---------------|-----------|-------------------|-----------------|
| 1 | 10.00 NNET | 10.00 NNET | 5.00 NNET |
| 5 | 2.00 NNET | 2.00 NNET | 1.00 NNET |
| 10 | 1.00 NNET | 1.00 NNET | 0.50 NNET |
| 50 | 0.20 NNET | 0.20 NNET | 0.10 NNET |
| 100 | 0.10 NNET | 0.10 NNET | 0.05 NNET |
| 500 | 0.02 NNET | 0.02 NNET | 0.01 NNET |
| 1000 | 0.01 NNET | 0.01 NNET | 0.005 NNET |
| 1,000,000 | 0.00001 NNET | 0.00001 NNET | 0.000005 NNET |

### 3.5 Economic Properties

1. **Anti-Hyperinflation**: No matter how many miners join, total issuance per block stays constant
2. **Early Adopter Advantage**: First miners earn significantly more
3. **Self-Balancing**: High rewards attract miners → rewards decrease → equilibrium
4. **Fair Distribution**: Larger computing power earns proportionally more

---

## 4. Tokenomics (EIP-1559 Model)

### 4.1 Token Overview

| Parameter | Value |
|-----------|-------|
| **Token Name** | NeoNet |
| **Ticker** | NNET |
| **Genesis Supply** | 50,000,000 NNET |
| **Supply Model** | Dynamic (inflationary/deflationary) |
| **Decimals** | 18 |
| **Burn Address** | neo1000000000000000000000000000000000dead |

### 4.2 Token Creation (Minting)

New NNET tokens are **minted** (created) only through AI Energy Mining:

```python
# How minting works in NeoNet
def issue_nnet(recipient: str, amount: float, task_type: str):
    """
    Creates new NNET tokens and adds them to circulation.
    Only triggered when miners complete AI tasks.
    """
    # 1. Validate the completed AI task
    if not validate_ai_task(task_id):
        return False
    
    # 2. Calculate reward based on task type
    rewards = {
        "federated_learning": 1.00,  # Highest reward
        "model_training": 0.80,
        "network_protection": 0.60,
        "fraud_detection": 0.50,
        "gradient_compute": 0.50,
        "inference": 0.40,
        "data_validation": 0.30,
        "matrix_ops": 0.30
    }
    
    # 3. Mint new tokens (increase total supply)
    blockchain.balances[recipient] += amount
    blockchain.network_stats["total_issued"] += amount
    blockchain.network_stats["current_supply"] += amount
    
    # 4. Record in transaction history
    record_issuance(recipient, amount, task_type)
```

**Minting Rules:**
- Only AI task completion triggers minting
- No pre-mine (founders receive from genesis)
- No ICO/IEO allocations
- Annual issuance rate: ~2% (controlled by network activity)

### 4.3 Token Destruction (Burning)

NNET tokens are **burned** (permanently destroyed) through transaction fees:

```python
# How burning works in NeoNet (EIP-1559 style)
def process_transaction_fee(tx_fee: float):
    """
    Burns 70% of transaction fees, tips 30% to validators.
    """
    BURN_RATE = 0.70  # 70% burned
    TIP_RATE = 0.30   # 30% to validators
    
    burn_amount = tx_fee * BURN_RATE
    tip_amount = tx_fee * TIP_RATE
    
    # Burn tokens (send to dead address, reduce supply)
    blockchain.balances[BURN_ADDRESS] += burn_amount
    blockchain.network_stats["total_burned"] += burn_amount
    blockchain.network_stats["current_supply"] -= burn_amount
    
    # Tip goes to block validator
    blockchain.balances[validator] += tip_amount
```

**Burning Events:**
| Event | Burn Rate | Description |
|-------|-----------|-------------|
| Transaction Fee | 70% | Base fee burned, tips to validator |
| Contract Deployment | 100% | Full deployment fee burned |
| Failed Transaction | 50% | Partial burn for spam prevention |
| Governance Vote | 0% | No burn for voting |

### 4.4 Supply Formula

```
Current_Supply = Genesis_Supply + Total_Issued - Total_Burned

Where:
- Genesis_Supply = 50,000,000 NNET (fixed)
- Total_Issued = Sum of all mining rewards (increases)
- Total_Burned = Sum of all burned fees (increases)

Net_Change = Total_Issued - Total_Burned

If Net_Change > 0: Inflationary (supply grows)
If Net_Change < 0: Deflationary (supply shrinks)
If Net_Change = 0: Stable supply
```

### 4.5 Real-Time Tokenomics API

```bash
# Get current tokenomics state
curl https://api.neonet.network/api/tokenomics

# Response:
{
    "genesis_supply": 50000000.0,
    "current_supply": 50003192.47,
    "total_issued": 3192.47,
    "total_burned": 0.0,
    "net_supply_change": 3192.47,
    "is_deflationary": false,
    "base_fee_burn_rate": 0.7,
    "priority_fee_rate": 0.3
}
```

### 4.6 Genesis Distribution

| Allocation | Amount | Percentage | Vesting |
|------------|--------|------------|---------|
| Network Rewards | 35,000,000 NNET | 70% | Minted over time |
| Development Fund | 7,500,000 NNET | 15% | 2 year linear |
| Team & Advisors | 5,000,000 NNET | 10% | 1 year cliff + 2 year linear |
| Community Airdrops | 2,500,000 NNET | 5% | Immediate |

### 4.7 Deflationary Mechanics

As network usage grows, burning accelerates:

```
Scenario 1: Early Network (Low Usage)
  Mining Rewards: 1000 NNET/day
  Fee Burns: 100 NNET/day
  Net: +900 NNET/day (inflationary - incentivizes miners)

Scenario 2: Growing Network (Medium Usage)
  Mining Rewards: 1000 NNET/day
  Fee Burns: 1000 NNET/day
  Net: 0 NNET/day (stable supply)

Scenario 3: Mature Network (High Usage)
  Mining Rewards: 1000 NNET/day
  Fee Burns: 2000 NNET/day
  Net: -1000 NNET/day (deflationary - value increases)
```

### 4.8 Token Utility

| Use Case | Description |
|----------|-------------|
| **Gas Fees** | Pay for transactions and contract execution |
| **Staking** | Stake to become validator, earn rewards |
| **Governance** | Vote on proposals (1 NNET = 1 vote) |
| **AI Services** | Pay for AI inference, model training |
| **Mining Rewards** | Earn for providing compute power |

---

## 5. Quantum-Safe Cryptography

### 5.1 Hybrid Signature Scheme

ALL transactions use triple-layer signatures:

```
Transaction Signature:
  ├── ECDSA (secp256k1) - Classical EVM compatibility
  ├── Ed25519 - Quantum-resistant layer 1
  └── Dilithium3 - NIST PQC Level 3
```

### 5.2 Key Derivation

```
Mnemonic (BIP-39)
    ↓
Master Seed (512-bit)
    ├── EVM Key (BIP-44: m/44'/60'/0'/0/0)
    │     └── Address: 0x...
    └── Quantum Key (BIP-44: m/44'/9999'/0'/0/0)
          └── Address: neo1...
```

### 5.3 Why Hybrid?

- **ECDSA**: Required for Ethereum compatibility
- **Ed25519**: Fast, secure, widely deployed
- **Dilithium3**: NIST-approved post-quantum signature

---

## 6. AI Security Guard

### 6.1 Anti-Poisoning Protection (Outlier Detection)

AI monitors for model poisoning attacks:

```python
class AISecurityGuard:
    def validate_gradient(gradient, provider_id):
        # Check gradient norm
        if gradient_norm > threshold:
            reject("Abnormal gradient detected")
        
        # Check for statistical outliers
        if is_outlier(gradient, historical_data):
            reject("Outlier gradient - possible attack")
        
        # Decrease trust for suspicious providers
        decrease_trust(provider_id)
```

### 6.2 Byzantine Fault Tolerance

The system tolerates up to 33% malicious nodes:

```
Consensus Requirement:
  Valid Block = 2/3 + 1 of validators agree

Malicious Update Detection:
  - Gradient norm validation
  - Weight anomaly detection
  - Cross-validation between providers
```

### 6.3 Trust Score System

Each energy provider has a trust score:

```
Initial Trust: 0.5

Good Behavior:
  → Trust += 0.01 (gradual increase)

Bad Behavior:
  → Trust -= 0.10 (rapid decrease)

Low Trust (< 0.2):
  → Provider flagged as suspicious
  → Tasks not delegated to them
```

---

## 7. Smart Throttling (Adaptive DDoS Protection)

### 7.1 Philosophy

NeoNet doesn't block addresses - it adapts:

```
Traditional DDoS Protection:
  High Traffic → Block IP → Service Denied

NeoNet Smart Throttling:
  High Traffic → Increase Gas Fee → Attacker Pays More
```

### 7.2 Thresholds

| Requests/Minute | Status | Gas Multiplier |
|-----------------|--------|----------------|
| 0-50 | Normal | 1.0x |
| 50-100 | Elevated | 1.0x - 2.5x |
| 100-200 | Suspicious | 2.5x - 5.0x |
| 200+ | Attack | 10.0x |

### 7.3 Auto-Reset

Suspicious flags decay over time:

```
Every 60 seconds:
  - Request counters reset
  - Suspicion levels decay by 20%
  - Gas multipliers reduce toward 1.0x
```

---

## 8. Decentralized State Management

### 8.1 Ethereum-Style State Trie

NeoNet uses Merkle Patricia Trie for state:

```
State Trie Structure:
  Root Hash
    ├── Account 1: {balance, nonce, storage_root}
    ├── Account 2: {balance, nonce, storage_root}
    └── Account N: {balance, nonce, storage_root}
```

### 8.2 AI-Managed Integrity

AI continuously validates state integrity:

```
AI State Manager:
  - Detect tampering attempts
  - Validate Merkle proofs
  - Compact and optimize storage
  - Sync across energy providers
```

### 8.3 Light Client Support

Merkle proofs enable light clients:

```
Proof Request:
  Client → "Prove account X balance"

Proof Response:
  Server → {balance: 100, proof: [hash1, hash2, ...], root: "abc..."}

Verification:
  Client → Recompute root from proof → Match with trusted root
```

---

## 9. Progressive Decentralization

### 9.1 The Challenge

New blockchain networks face a bootstrap problem:
- Need servers to run initially
- Want to become fully decentralized
- Transition must be seamless

### 9.2 NeoNet Solution: Gradual Transition

The network automatically shifts from centralized bootstrap to distributed operation:

```
Phase 1: Bootstrap (0-10 miners)
  └── Bootstrap servers handle 100% of load
  └── Miners observe and earn rewards

Phase 2: Transition (10-100 miners)
  └── Load distributed: Bootstrap 50%, Miners 50%
  └── State replicated across miners

Phase 3: Distributed (100-1000 miners)
  └── Miners handle 90% of load
  └── Bootstrap acts as fallback only

Phase 4: Fully Decentralized (1000+ miners)
  └── Miners handle 100% of load
  └── Bootstrap server can be shut down
  └── Network is fully autonomous
```

### 9.3 Transition Metrics

```python
def calculate_decentralization_level():
    miners = count_active_miners()
    
    if miners < 10:
        return {"bootstrap": 100, "miners": 0, "phase": "bootstrap"}
    elif miners < 100:
        miner_share = (miners - 10) / 90 * 50
        return {"bootstrap": 100 - miner_share, "miners": miner_share, "phase": "transition"}
    elif miners < 1000:
        miner_share = 50 + (miners - 100) / 900 * 40
        return {"bootstrap": 100 - miner_share, "miners": miner_share, "phase": "distributed"}
    else:
        return {"bootstrap": 0, "miners": 100, "phase": "decentralized"}
```

### 9.4 Data Replication

As miners join, they receive:
1. **Blockchain State**: Full state trie
2. **Transaction History**: All historical transactions
3. **AI Models**: Trained fraud detection models
4. **Consensus Data**: Validator information

---

## 10. Dual Runtime (EVM + WASM)

### 10.1 Unified Execution Fabric

NeoNet supports both EVM and WASM contracts:

```
Contract Deployment:
  ├── Solidity (.sol) → EVM Bytecode → EVM Runtime
  └── Rust/AssemblyScript → WASM → WASM Runtime

Cross-VM Calls:
  EVM Contract ←→ Cross-VM Bridge ←→ WASM Contract
```

### 10.2 AI Runtime Planner

AI optimizes contract execution:

```
AI Planner:
  - Analyze contract bytecode
  - Estimate gas costs
  - Route to optimal runtime (EVM vs WASM)
  - Parallelize independent operations
```

---

## 11. Governance (DualGov)

### 11.1 Hybrid Voting

Governance combines AI and human voting:

```
Proposal Voting:
  ├── DAO Vote: 70% weight (token holders)
  └── AI Vote: 30% weight (AI analysis)

Final Decision:
  Approved if: (DAO_Yes * 0.7) + (AI_Confidence * 0.3) > 0.5
```

### 11.2 AI Analysis

AI evaluates proposals for:
- Technical feasibility
- Security implications
- Economic impact
- Alignment with network goals

---

## 12. Network Architecture

### 12.1 Component Overview

```
NeoNet Network
├── Frontend Layer
│   ├── React dApp (Web4 Wallet)
│   └── HashRouter Navigation
├── API Layer
│   ├── FastAPI (Python)
│   └── REST/WebSocket Endpoints
├── Blockchain Layer
│   ├── Rust Core (EVM+WASM)
│   └── State Management
├── Consensus Layer
│   ├── Go (libp2p)
│   └── Proof of Intelligence
└── Security Layer
    ├── AI Security Guard
    ├── Smart Throttling
    └── Quantum Signatures
```

### 12.2 Port Allocation

| Service | Port | Protocol |
|---------|------|----------|
| Frontend/API | 5000 | HTTP |
| Rust Core | 6000 | TCP |
| Go Consensus | 8080 | HTTP |
| Go P2P | 50051 | gRPC |

---

## 13. API Reference

### 13.1 Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/network/stats` | GET | Network statistics |
| `/api/tokenomics` | GET | Token economics |
| `/api/protection/status` | GET | Security status |

### 13.2 Mining Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/ai-energy/register` | POST | Register as miner |
| `/api/ai-energy/start-session` | POST | Start mining |
| `/api/ai/delegate-task` | POST | Get AI task |
| `/api/ai/task-requirements` | GET | Hardware requirements |

### 13.3 Wallet Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/wallet/balances/{address}` | GET | Get balances |
| `/api/wallet/send` | POST | Send tokens |
| `/api/tokens/transfer` | POST | Transfer tokens |

---

## 14. Security Considerations

### 14.1 Attack Vectors & Mitigations

| Attack | Mitigation |
|--------|------------|
| Model Poisoning | Outlier detection, gradient norm validation |
| Sybil Attack | Trust score system, stake requirements |
| DDoS | Smart throttling, adaptive gas fees |
| Quantum Attack | Dilithium3 signatures |
| 51% Attack | Byzantine fault tolerance (2/3 + 1) |
| Eclipse Attack | Multiple bootstrap nodes, peer diversity |

### 14.2 AI Cannot Be Hacked Because:

1. **No Single Point of Failure**: AI runs distributed across miners
2. **Gradient Validation**: All updates validated before aggregation
3. **Trust System**: Malicious providers lose trust and rewards
4. **Byzantine Tolerance**: 33% malicious nodes tolerated
5. **State Verification**: Merkle proofs verify all state changes

---

## 15. Roadmap

### Phase 1: Bootstrap (Current)
- [x] Core blockchain implementation
- [x] AI Miner with PoI consensus
- [x] Quantum-safe signatures
- [x] Dynamic reward system
- [x] Smart throttling DDoS protection

### Phase 2: Growth (Q1 2026)
- [ ] Mobile wallet app
- [ ] 100+ active miners
- [ ] Cross-chain bridges (Ethereum, BSC)
- [ ] NFT marketplace

### Phase 3: Maturity (Q2 2026)
- [ ] 1000+ active miners
- [ ] Full decentralization (bootstrap server shutdown)
- [ ] DEX with limit orders
- [ ] Governance v2

### Phase 4: Expansion (Q3 2026)
- [ ] Layer 2 scaling
- [ ] Enterprise partnerships
- [ ] AI marketplace
- [ ] Global validator network

---

## 16. Conclusion

NeoNet represents a paradigm shift in blockchain technology:

1. **AI Does the Work**: Users provide energy, AI handles computation
2. **Fair Rewards**: Dynamic distribution based on participation
3. **Future-Proof Security**: Quantum-safe from day one
4. **Inclusive Mining**: Any hardware can participate
5. **Progressive Decentralization**: Smooth transition to full distribution

Join the Web4 revolution. Provide your computing power. Earn NNET.

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **NNET** | NeoNet native token |
| **PoI** | Proof of Intelligence consensus |
| **Energy Provider** | User providing computing power |
| **Dilithium3** | NIST post-quantum signature algorithm |
| **State Trie** | Merkle Patricia Trie for account state |
| **DualGov** | Hybrid AI + DAO governance |

## Appendix B: References

1. NIST Post-Quantum Cryptography: https://csrc.nist.gov/projects/post-quantum-cryptography
2. Ethereum EIP-1559: https://eips.ethereum.org/EIPS/eip-1559
3. Ed25519: https://ed25519.cr.yp.to/
4. Merkle Patricia Trie: https://ethereum.org/en/developers/docs/data-structures-and-encoding/patricia-merkle-trie/

---

**Official Links:**
- Website: https://neonetainetwork.com
- Documentation: https://neonetainetwork.info
- Community: https://neonetainetwork.org
- GitHub (Public): https://github.com/neonetainetwork/neonet-public
- Discussions: https://github.com/neonetainetwork/neonet-public/discussions

---

*Copyright 2025 NeoNet. MIT License.*
