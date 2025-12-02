# NeoNet - AI-Powered Web4 Blockchain

## Overview

NeoNet is a revolutionary Web4 blockchain platform featuring:

- **AI-Powered Proof of Intelligence (PoI) Consensus** - Validators prove intelligence through ML model training
- **Unified Dual Runtime (EVM+WASM)** - Integrated fabric with cross-VM orchestration
- **Post-Quantum Cryptography** - Ed25519 + Dilithium3 hybrid signatures for quantum resistance
- **Single Token Economy** - NEO (50,000,000 supply) - governance, staking, gas, mining rewards
- **Web4 Wallet** - Dual keys: EVM (0x...) + Quantum (neo1...)
- **DualGov Governance** - AI voting (30%) + DAO voting (70%)

## Architecture

```
                    +-------------------+
                    |    NeoNet dApp    |
                    |   (React + Vite)  |
                    +--------+----------+
                             |
                    +--------v----------+
                    |  Python AI Service |
                    | (FastAPI + PyTorch)|
                    +--------+----------+
                             |
          +------------------+------------------+
          |                                     |
+---------v---------+               +-----------v-----------+
|   Go Consensus    |<------------->|     Rust Core        |
| (P2P + Validator) |  TCP Bridge   | (Block Processing)   |
+-------------------+               +-----------------------+
          |                                     |
          |            +------------+           |
          +----------->| Blockchain |<----------+
                       | (Quantum-Safe)|
                       +------------+
```

## Components

### Rust Core (`rust-core/`)
- Deterministic block building and transaction processing
- WASM VM with gas metering for CosmWasm contracts
- EVM adapter layer for Ethereum compatibility
- Hybrid post-quantum cryptography (Ed25519 + Dilithium3)

### Go Consensus (`go-consensus/`)
- P2P networking for distributed consensus
- Proof-of-Authority validator rotation
- HTTP API for chain queries and transaction submission

### Python AI Service (`python-ai-service/`)
- RESTful API with FastAPI
- PyTorch-based fraud detection models
- Federated learning engine for distributed AI training
- Post-quantum cryptography wrappers

### Smart Contracts
- **EVM (Solidity)**: NeoToken, StakeAndVote, Oracle
- **WASM (CosmWasm)**: AI Model Registry, DualGov

## Quick Start

```bash
# Clone repository
git clone https://github.com/Timur2000kdkmd/NeoNet-Blockchain.git
cd NeoNet-Blockchain

# Start Rust Core
cd rust-core && cargo build --release && ./target/release/neonet_core

# Start Go Consensus  
cd go-consensus && go build -o neonet-consensus && ./neonet-consensus -http 8080 -port 50051

# Start AI Service
cd python-ai-service && pip install -r requirements.txt && uvicorn app.main_simplified:app --host 0.0.0.0 --port 8000

# Start dApp
cd dapp && npm install && npm run dev
```

## Security

All transactions are protected with hybrid quantum-resistant cryptography:
- **Classical**: Ed25519 (current protection)
- **Post-Quantum**: Dilithium3 (future quantum computer protection)

## Token Economics

- **Total Supply**: 50,000,000 NEO
- **No Free Tokens**: All tokens must be earned through mining, staking, or AI training
- **Mining Rewards**: 10 NEO per block
- **Staking APY**: Dynamic based on network participation

## Governance

DualGov combines human and AI decision-making:
- **DAO Voting (70%)**: Community-driven decisions
- **AI Voting (30%)**: AI recommendations based on network analysis

## License

MIT License - See LICENSE file for details.

## Community

- [Discussions](https://github.com/Timur2000kdkmd/NeoNet-Blockchain/discussions) - Ask questions, share ideas
- [Issues](https://github.com/Timur2000kdkmd/NeoNet-Blockchain/issues) - Report bugs, request features
