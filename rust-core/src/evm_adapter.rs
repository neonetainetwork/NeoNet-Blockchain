// EVM Adapter for NeoNet - Ethereum Virtual Machine compatibility
use serde::{Deserialize, Serialize};
use anyhow::{Result, anyhow};
use std::collections::HashMap;
use sha2::{Sha256, Digest};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EVMAccount {
    pub address: String,
    pub balance: u128,
    pub nonce: u64,
    pub code: Vec<u8>,
    pub storage: HashMap<String, String>,
}

pub struct EVMAdapter {
    accounts: HashMap<String, EVMAccount>,
    gas_price: u64,
    block_number: u64,
}

impl EVMAdapter {
    pub fn new() -> Self {
        EVMAdapter {
            accounts: HashMap::new(),
            gas_price: 20,
            block_number: 0,
        }
    }

    pub fn create_account(&mut self, address: String, initial_balance: u128) -> Result<()> {
        if self.accounts.contains_key(&address) {
            return Err(anyhow!("Account already exists"));
        }

        let account = EVMAccount {
            address: address.clone(),
            balance: initial_balance,
            nonce: 0,
            code: vec![],
            storage: HashMap::new(),
        };

        self.accounts.insert(address, account);
        Ok(())
    }

    pub fn deploy_contract(&mut self, deployer: &str, code: Vec<u8>) -> Result<String> {
        let deployer_account = self.accounts.get_mut(deployer)
            .ok_or_else(|| anyhow!("Deployer account not found"))?;

        let nonce = deployer_account.nonce;
        deployer_account.nonce += 1;

        let mut hasher = Sha256::new();
        hasher.update(format!("{}{}", deployer, nonce).as_bytes());
        let hash = hasher.finalize();
        let contract_address = format!("0x{}", hex::encode(&hash[..20]));

        let contract = EVMAccount {
            address: contract_address.clone(),
            balance: 0,
            nonce: 1,
            code,
            storage: HashMap::new(),
        };

        self.accounts.insert(contract_address.clone(), contract);
        Ok(contract_address)
    }

    pub fn call_contract(
        &mut self,
        from: &str,
        to: &str,
        data: Vec<u8>,
        value: u128,
        _gas_limit: u64
    ) -> Result<Vec<u8>> {
        let from_account = self.accounts.get_mut(from)
            .ok_or_else(|| anyhow!("From account not found"))?;

        if from_account.balance < value {
            return Err(anyhow!("Insufficient balance"));
        }

        from_account.balance -= value;
        from_account.nonce += 1;
        let from_balance = from_account.balance;

        let to_account = self.accounts.get_mut(to)
            .ok_or_else(|| anyhow!("Contract not found"))?;

        to_account.balance += value;

        let method_sig = if data.len() >= 4 {
            hex::encode(&data[0..4])
        } else {
            String::new()
        };

        match method_sig.as_str() {
            "70a08231" => {
                let balance_bytes = to_account.balance.to_be_bytes();
                Ok(balance_bytes.to_vec())
            },
            "a9059cbb" => {
                let mut success = vec![0u8; 31];
                success.push(1u8);
                Ok(success)
            },
            _ => {
                Ok(format!("EVM call: from balance {}, to {}, data {} bytes", 
                    from_balance, to, data.len()).into_bytes())
            }
        }
    }

    pub fn transfer(&mut self, from: &str, to: &str, amount: u128) -> Result<()> {
        {
            let from_account = self.accounts.get_mut(from)
                .ok_or_else(|| anyhow!("From account not found"))?;

            if from_account.balance < amount {
                return Err(anyhow!("Insufficient balance"));
            }

            from_account.balance -= amount;
            from_account.nonce += 1;
        }

        let to_account = self.accounts.get_mut(to)
            .ok_or_else(|| anyhow!("To account not found"))?;

        to_account.balance += amount;
        Ok(())
    }

    pub fn get_balance(&self, address: &str) -> Result<u128> {
        self.accounts.get(address)
            .map(|acc| acc.balance)
            .ok_or_else(|| anyhow!("Account not found"))
    }

    pub fn get_nonce(&self, address: &str) -> Result<u64> {
        self.accounts.get(address)
            .map(|acc| acc.nonce)
            .ok_or_else(|| anyhow!("Account not found"))
    }

    pub fn increment_block(&mut self) {
        self.block_number += 1;
    }

    pub fn get_block_number(&self) -> u64 {
        self.block_number
    }
}

impl Default for EVMAdapter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_account() {
        let mut evm = EVMAdapter::new();
        assert!(evm.create_account("0xalice".to_string(), 1000).is_ok());
        assert_eq!(evm.get_balance("0xalice").unwrap(), 1000);
    }

    #[test]
    fn test_transfer() {
        let mut evm = EVMAdapter::new();
        evm.create_account("0xalice".to_string(), 1000).unwrap();
        evm.create_account("0xbob".to_string(), 0).unwrap();
        
        assert!(evm.transfer("0xalice", "0xbob", 100).is_ok());
        assert_eq!(evm.get_balance("0xalice").unwrap(), 900);
        assert_eq!(evm.get_balance("0xbob").unwrap(), 100);
    }

    #[test]
    fn test_deploy_contract() {
        let mut evm = EVMAdapter::new();
        evm.create_account("0xdeployer".to_string(), 1000000).unwrap();
        
        let bytecode = vec![0x60, 0x60, 0x60, 0x40];
        let contract_addr = evm.deploy_contract("0xdeployer", bytecode).unwrap();
        
        assert!(contract_addr.starts_with("0x"));
        assert_eq!(evm.get_nonce("0xdeployer").unwrap(), 1);
    }
}
