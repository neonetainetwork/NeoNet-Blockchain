# NeoNet - AI-Powered Web4 Blockchain

## Overview

NeoNet is a revolutionary Web4 blockchain platform featuring:

- **AI-Powered Proof of Intelligence (PoI) Consensus** - AI does ALL work, users provide energy
- **Unified Dual Runtime (EVM+WASM)** - Integrated fabric with cross-VM orchestration
- **Post-Quantum Cryptography** - Hybrid Ed25519 + Dilithium3 signatures on ALL transactions
- **Ethereum EIP-1559 Tokenomics** - Dynamic supply with 70% fee burn, 30% to miners
- **AI Energy Mining** - Earn up to 1.00 NNET per task (10x rewards!)
- **Web4 Wallet** - Dual keys: EVM (0x...) + Quantum (neo1...)
- **DualGov Governance** - AI voting (30%) + DAO voting (70%)
- **Progressive Decentralization** - Network transitions from bootstrap server to miners automatically
- **Smart Throttling** - Adaptive DDoS protection (gas adjustment, not blocking)
- **AI Cannot Be Hacked** - Multi-layer security prevents all attack vectors

## Token Economics (EIP-1559)

| Feature | Description |
|---------|-------------|
| **Ticker** | NNET |
| **Genesis Supply** | 50,000,000 NNET (dynamic - can grow or shrink) |
| **Fee Burn** | 70% of transaction fees burned |
| **Miner Tips** | 30% of fees go to AI Energy Providers |
| **Mining Rewards** | Dynamic: `Reward = Block_Budget / Active_Miners` |
| **Supply Model** | Deflationary when burn > issuance |

### AI Energy Mining Rewards

| Task Type | Reward | Hardware Required |
|-----------|--------|-------------------|
| **Federated Learning** | 1.00 NNET | 8+ CPU, 16GB GPU |
| **Model Training** | 0.80 NNET | 8+ CPU, 8GB GPU |
| **Network Protection** | 0.60 NNET | 4+ CPU |
| **Fraud Detection** | 0.50 NNET | 1+ CPU (laptops OK!) |
| **Gradient Compute** | 0.50 NNET | 4+ CPU, 2GB GPU |
| **AI Inference** | 0.40 NNET | 1+ CPU (laptops OK!) |
| **Data Validation** | 0.30 NNET | 1+ CPU (laptops OK!) |
| **Matrix Operations** | 0.30 NNET | 2+ CPU |

**Note**: Even weak laptops can participate and earn NNET tokens!

## Quick Start

### Prerequisites

- Node.js 18+
- Python 3.11+
- Rust (for core blockchain)
- Go 1.21+ (for consensus layer)

### Installation

```bash
# Clone this repository
git clone https://github.com/neonetainetwork/neonetpqc.git
cd NeoNetPQC

# Install Python dependencies
cd python-ai-service
pip install -r requirements.txt

# Install dApp dependencies
cd ../dapp
npm install

# Start backend
cd ../python-ai-service
python -m uvicorn app.main_simplified:app --host 0.0.0.0 --port 5000 --reload

# Frontend is served from backend in production mode
```

### Start AI Energy Mining

```bash
# Download the miner script
pip install aiohttp numpy

# Run the miner
python neonet_miner.py --server https://neonetainetwork.com --wallet neo1yourwalletaddress
```

## Architecture

### Network Components

```
NeoNet Architecture
├── Frontend (React + Vite + HashRouter)
│   ├── Web4 Wallet with dual keys
│   ├── Dashboard with live network stats
│   └── Developer tools for contract deployment
├── Backend (Python FastAPI)
│   ├── AI Service (Proof of Intelligence)
│   ├── Federated Learning Engine
│   ├── AI Energy Mining System (10x rewards)
│   └── EIP-1559 Fee Burning
├── Blockchain Core (Rust)
│   ├── Unified EVM+WASM Runtime
│   └── Post-Quantum Cryptography
└── Consensus Layer (Go)
    ├── P2P Networking (libp2p)
    └── Proof of Authority with AI validation
```

### Directory Structure

```
NeoNetPQC/
├── dapp/                    # React frontend
│   ├── src/
│   │   ├── App.jsx         # Main application
│   │   ├── components/     # React components
│   │   └── lib/           # Wallet & utilities
│   └── package.json
├── python-ai-service/       # Python backend
│   ├── app/
│   │   ├── main_simplified.py  # FastAPI server
│   │   ├── neonet_blockchain.py # Blockchain with EIP-1559
│   │   ├── federated_learning.py
│   │   └── poi_consensus.py
│   └── requirements.txt
├── rust-core/              # Rust blockchain core
│   ├── src/
│   │   └── unified_runtime.rs
│   └── Cargo.toml
├── go-consensus/           # Go consensus layer
│   └── main.go
└── contracts/
    ├── sol/               # Solidity contracts
    └── wasm/              # CosmWasm contracts
```

## Quantum-Safe Signatures

ALL transactions on NeoNet use hybrid quantum-safe signatures:

```
Transaction Signature = {
  evm_signature: ECDSA (classical),
  quantum_signature: Ed25519 (quantum-resistant layer),
  dilithium_signature: Dilithium3 (NIST PQC Level 3)
}
```

This protects against both classical and future quantum computer attacks.

## API Endpoints

### Health & Status
- `GET /health` - Service health check
- `GET /api/network/stats` - Network statistics
- `GET /api/network/overview` - Network overview

### Token Economics
- `GET /api/tokenomics` - EIP-1559 tokenomics info (burn rate, supply, rewards)
- `GET /api/dex/price/NNET` - Current NNET price

### AI Energy Mining
- `POST /api/ai-energy/register` - Register as AI Energy Provider
- `POST /api/ai-energy/start-session` - Start mining session
- `GET /api/ai-energy/stats` - Mining statistics
- `GET /api/ai-energy/leaderboard` - Top miners

### Wallet
- `GET /api/wallet/balances/{address}` - Get wallet balances
- `POST /api/wallet/send` - Send NNET tokens

### DEX
- `GET /api/dex/pools` - Get liquidity pools
- `POST /api/dex/swap` - Execute swap
- `GET /api/dex/quote` - Get swap quote

## URL Routes (HashRouter)

| Route | Description |
|-------|-------------|
| `/#/home` | Landing page |
| `/#/dashboard` | Network dashboard |
| `/#/wallet` | Wallet management |
| `/#/send` | Send tokens |
| `/#/dex` | Decentralized exchange |
| `/#/explore` | Explore dApps & tokens |
| `/#/mining` | AI Energy mining |
| `/#/developer` | Deploy contracts |
| `/#/info` | Documentation |
| `/#/docs` | API docs |

## Official Domains

| Service | URL |
|---------|-----|
| Main Website | https://neonetainetwork.com |
| Documentation | https://neonetainetwork.info |
| Community/DAO | https://neonetainetwork.org |
| RPC Endpoint | https://rpc.neonetainetwork.com |
| Block Explorer | https://explorer.neonetainetwork.com |

## Genesis Allocation

- **Total Genesis**: 50,000,000 NNET
- **User Allocation**: 30,000,000 NNET to `neo1dfa5ee86e6443115287e8a6c604cd8aa32d101`
- **AI Energy Pool**: Remaining supply for mining rewards

## License

MIT License - See LICENSE file for details.

## Community

- GitHub (Public): https://github.com/neonetainetwork/NeoNet-Blockchain
- Discussions: https://github.com/neonetainetwork/NeoNet-Blockchain/discussions
- Documentation: https://neonetainetwork.info
- Discord: Coming soon
- Twitter: Coming soon

## Miner Downloads

Download the miner client from our public repository:
```bash
git clone https://github.com/neonetainetwork/NeoNet-Blockchain.git
cd NeoNet-Blockchain/miner-client
pip install aiohttp numpy
python neonet_miner.py --wallet neo1yourwallet --server https://neonetainetwork.com
```
