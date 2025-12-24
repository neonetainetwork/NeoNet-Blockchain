# NeoNet Miner Client (NNET)

One-Click AI Mining for the NeoNet Web4 Blockchain.

## Dynamic Rewards System

NeoNet uses **dynamic rewards** that adjust based on active miners:

```
Formula: Reward = Task_Weight × (Block_Budget / Active_Miners)

Block_Budget = 10.0 NNET (fixed "pie" per block)
Task_Weight = 0.30 to 1.00 (based on task difficulty)

Examples:
- 1 miner:    federated_learning = 10.0 NNET (max)
- 10 miners:  federated_learning = 1.0 NNET
- 100 miners: federated_learning = 0.1 NNET
- 1000 miners: federated_learning = 0.01 NNET
- 1M miners:  federated_learning = 0.00001 NNET
```

**Why dynamic?** Prevents hyperinflation with millions of miners while rewarding early adopters.

## One-Click GUI Miner (Recommended)

### Windows/Mac/Linux

1. Install Python 3.8+
2. Install dependencies:
   ```bash
   pip install aiohttp numpy
   ```
3. Run the GUI miner:
   ```bash
   python neonet_miner_gui.py
   ```
4. Enter your wallet address (neo1... or 0x...)
5. Click **START MINING**

### Build Standalone .exe (Windows)

```bash
pip install pyinstaller
pyinstaller --onefile --noconsole --name NeoNetMiner neonet_miner_gui.py
```

The executable will be in the `dist/` folder.

## Command Line Miner (Advanced)

For servers and advanced users:

```bash
python neonet_miner.py --wallet neo1abc123... --server https://neonetainetwork.com
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--wallet` | Your wallet address | Required |
| `--server` | NeoNet server URL | Auto-detect |
| `--cpu` | CPU cores to use | Auto-detect |
| `--gpu-memory` | GPU memory in MB | 0 |

## Task Types & Rewards

Rewards are calculated dynamically: `Task_Weight × (10.0 / Active_Miners)`

| Task Type | Weight | 1 Miner | 10 Miners | 100 Miners |
|-----------|--------|---------|-----------|------------|
| federated_learning | 1.00x | 10.00 | 1.00 | 0.10 |
| model_training | 0.80x | 8.00 | 0.80 | 0.08 |
| network_protection | 0.60x | 6.00 | 0.60 | 0.06 |
| fraud_detection | 0.50x | 5.00 | 0.50 | 0.05 |
| gradient_compute | 0.50x | 5.00 | 0.50 | 0.05 |
| inference | 0.40x | 4.00 | 0.40 | 0.04 |
| data_validation | 0.30x | 3.00 | 0.30 | 0.03 |
| matrix_ops | 0.30x | 3.00 | 0.30 | 0.03 |

## Hardware Requirements

| Level | Requirements | Typical Tasks |
|-------|--------------|---------------|
| Minimum | 1 CPU, No GPU | fraud_detection, inference, data_validation |
| Standard | 4 CPU, 2GB GPU | model_training, gradient_compute |
| High | 8+ CPU, 8GB+ GPU | federated_learning |

Any computer can mine! The network assigns appropriate tasks.

## How It Works

1. **Register** - Your wallet is registered with the network
2. **Start Session** - Begin receiving AI tasks
3. **Process Tasks** - Your computer performs AI computations
4. **Earn Rewards** - NNET tokens are minted to your wallet

All rewards are **minted as new tokens**, balanced by transaction fee burning (EIP-1559 style).

## Progressive Decentralization

As more miners join, the network becomes more decentralized:

| Miners | Network State | Server Load | Miner Load |
|--------|---------------|-------------|------------|
| 0-10 | Bootstrap | 100% | 0% |
| 10-100 | Transition | 50% | 50% |
| 100-1000 | Distributed | 10% | 90% |
| 1000+ | Decentralized | 0% | 100% |

### Automatic Server Shutdown

When the network reaches **1000+ active miners**, the central bootstrap server will **automatically shut down**:

1. Network reaches 1000+ miners
2. System waits 5 minutes for stability confirmation
3. Warning message is logged
4. After 60 seconds grace period, server shuts down
5. Network continues running entirely on P2P miner nodes

**API Endpoints:**
- `GET /api/decentralization/status` - Check current phase
- `GET /api/decentralization/auto-shutdown/status` - Check shutdown conditions
- `POST /api/decentralization/auto-shutdown/disable` - Disable auto-shutdown (manual control)

Eventually, the network runs entirely on miner nodes!

## Full Node (Рекомендуется)

**Один скрипт = полный узел сети.** AI запускается на вашем компьютере и держит сеть:

```bash
python neonet_full_node.py --wallet neo1your_wallet
```

### Что делает Full Node:
- **AI Engine** — обрабатывает все типы задач локально
- **Blockchain** — хранит локальную копию блокчейна
- **P2P Server** — принимает подключения от других узлов
- **Rewards** — получает NNET за работу

### Опции

| Флаг | Описание | По умолчанию |
|------|----------|--------------|
| `--wallet` | Адрес кошелька | Обязательно |
| `--port` | Порт для P2P | 8080 |
| `--cpu` | Ядер CPU | 4 |
| `--gpu-mem` | GPU память (МБ) | 0 |

### Пример запуска

```bash
# Базовый запуск
python neonet_full_node.py --wallet neo1abc123

# С указанием порта и ресурсов
python neonet_full_node.py --wallet neo1abc123 --port 9000 --cpu 8 --gpu-mem 4096
```

После запуска ваш компьютер:
1. Подключается к сети NeoNet
2. Получает задачи от сети или генерирует свои
3. AI обрабатывает задачи
4. Получает награды в NNET

---

## P2P Miner (Альтернативный режим)

Если нужен только майнинг без полного узла:

### Quick Start

```bash
python neonet_p2p_miner.py --wallet neo1your_wallet_address
```

### P2P Options

| Flag | Description | Default |
|------|-------------|---------|
| `--wallet` | Your wallet address | Required |
| `--port` | P2P listen port | 9876 |
| `--cpu` | CPU cores to use | 4 |
| `--gpu-mem` | GPU memory in MB | 0 |
| `--mode` | Network mode | hybrid |
| `--bootstrap` | Additional bootstrap servers | - |

### Network Modes

| Mode | Description |
|------|-------------|
| `hybrid` | Auto-switch between bootstrap and P2P (recommended) |
| `bootstrap` | Connect only to central server |
| `p2p` | Pure P2P mode, no central server |

### How P2P Mode Works

1. **Peer Discovery** - Finds other miners via bootstrap or known peers
2. **Local Node** - Runs local server on `--port` for other peers
3. **Task Distribution** - Gets/sends tasks from/to peers
4. **Automatic Fallback** - Switches to P2P when bootstrap unavailable

### Example: Pure P2P Mode

```bash
python neonet_p2p_miner.py --wallet neo1abc123 --mode p2p --port 9876
```

### Example: Add Custom Bootstrap

```bash
python neonet_p2p_miner.py --wallet neo1abc123 --bootstrap http://my-node.com:8000
```

## Running as a Service (Linux)

Create `/etc/systemd/system/neonet-miner.service`:

```ini
[Unit]
Description=NeoNet AI Miner
After=network.target

[Service]
Type=simple
User=yourusername
WorkingDirectory=/path/to/miner-client
ExecStart=/usr/bin/python3 neonet_miner.py --wallet your_wallet --server https://neonetainetwork.com
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl enable neonet-miner
sudo systemctl start neonet-miner
```

## Links

- Website: https://neonetainetwork.com
- Documentation: https://neonetainetwork.info
- GitHub: https://github.com/neonetainetwork/NeoNet-Blockchain
- Discussions: https://github.com/neonetainetwork/NeoNet-Blockchain/discussions
- Issues: https://github.com/neonetainetwork/NeoNet-Blockchain/issues

## License

MIT License - NeoNet AI Network 2025
