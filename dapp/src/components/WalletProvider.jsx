import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { NeoWallet, createWallet, restoreWallet, validateMnemonic, injectNeoProvider } from '../lib/wallet';
import { secureStorage } from '../lib/security';

const WalletContext = createContext(null);

async function deriveKey(password, salt) {
  const enc = new TextEncoder();
  const keyMaterial = await crypto.subtle.importKey(
    'raw',
    enc.encode(password),
    { name: 'PBKDF2' },
    false,
    ['deriveBits', 'deriveKey']
  );
  return crypto.subtle.deriveKey(
    { name: 'PBKDF2', salt, iterations: 100000, hash: 'SHA-256' },
    keyMaterial,
    { name: 'AES-GCM', length: 256 },
    false,
    ['encrypt', 'decrypt']
  );
}

async function encryptData(data, password) {
  const salt = crypto.getRandomValues(new Uint8Array(16));
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const key = await deriveKey(password, salt);
  const enc = new TextEncoder();
  const encrypted = await crypto.subtle.encrypt(
    { name: 'AES-GCM', iv },
    key,
    enc.encode(JSON.stringify(data))
  );
  const combined = new Uint8Array(salt.length + iv.length + encrypted.byteLength);
  combined.set(salt, 0);
  combined.set(iv, salt.length);
  combined.set(new Uint8Array(encrypted), salt.length + iv.length);
  return btoa(String.fromCharCode(...combined));
}

async function decryptData(encryptedStr, password) {
  try {
    const combined = new Uint8Array(atob(encryptedStr).split('').map(c => c.charCodeAt(0)));
    const salt = combined.slice(0, 16);
    const iv = combined.slice(16, 28);
    const data = combined.slice(28);
    const key = await deriveKey(password, salt);
    const decrypted = await crypto.subtle.decrypt(
      { name: 'AES-GCM', iv },
      key,
      data
    );
    return JSON.parse(new TextDecoder().decode(decrypted));
  } catch (e) {
    throw new Error('Invalid password');
  }
}

function useWallet() {
  const context = useContext(WalletContext);
  if (!context) {
    throw new Error('useWallet must be used within a WalletProvider');
  }
  return context;
}

export { useWallet };

export function WalletProvider({ children }) {
  const [wallet, setWallet] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [addresses, setAddresses] = useState({ evmAddress: null, neoAddress: null });
  const [balances, setBalances] = useState({ NNET: 0 });
  const [showMnemonic, setShowMnemonic] = useState(false);
  const [mnemonic, setMnemonic] = useState('');
  const [hasPassword, setHasPassword] = useState(false);
  const [privateKeys, setPrivateKeys] = useState(null);

  const [tokens, setTokens] = useState([]);
  
  const fetchBalanceFromServer = useCallback(async (address) => {
    if (!address) return;
    try {
      const response = await fetch(`/api/wallet/balances/${address}`);
      if (response.ok) {
        const data = await response.json();
        const allBalances = data.balances || { NNET: 0 };
        setBalances(allBalances);
        if (data.tokens) {
          setTokens(data.tokens);
        }
        if (wallet) {
          wallet.setBalances(allBalances);
          if (data.tokens) wallet.tokens = data.tokens;
        }
      }
    } catch (e) {
      console.log('Balance fetch error:', e);
    }
  }, [wallet]);

  useEffect(() => {
    const savedAddresses = secureStorage.get('addresses');
    const hasEncrypted = secureStorage.get('encrypted');
    if (savedAddresses) {
      setAddresses(savedAddresses);
      setIsConnected(true);
      if (savedAddresses.neoAddress) {
        fetchBalanceFromServer(savedAddresses.neoAddress);
      }
    }
    if (hasEncrypted) {
      setHasPassword(true);
    }
  }, []);

  useEffect(() => {
    if (addresses.neoAddress && isConnected) {
      fetchBalanceFromServer(addresses.neoAddress);
      const interval = setInterval(() => {
        fetchBalanceFromServer(addresses.neoAddress);
      }, 10000);
      return () => clearInterval(interval);
    }
  }, [addresses.neoAddress, isConnected, fetchBalanceFromServer]);

  const setPassword = useCallback(async (password, walletData, walletAddresses) => {
    try {
      const dataToEncrypt = {
        mnemonic: walletData.mnemonic,
        evmPrivateKey: walletData.evmPrivateKey,
        neoPrivateKey: walletData.neoPrivateKey
      };
      const encrypted = await encryptData(dataToEncrypt, password);
      secureStorage.set('encrypted', encrypted);
      secureStorage.set('addresses', walletAddresses);
      setHasPassword(true);
      setIsConnected(true);
      setMnemonic('');
      return true;
    } catch (e) {
      throw new Error('Failed to encrypt wallet data');
    }
  }, []);

  const unlockSecrets = useCallback(async (password) => {
    try {
      const encrypted = secureStorage.get('encrypted');
      if (!encrypted) {
        throw new Error('No encrypted data found');
      }
      const decrypted = await decryptData(encrypted, password);
      setPrivateKeys(decrypted);
      setMnemonic(decrypted.mnemonic);
      return decrypted;
    } catch (e) {
      throw new Error('Invalid password');
    }
  }, []);

  const lockSecrets = useCallback(() => {
    setPrivateKeys(null);
    setMnemonic('');
  }, []);

  const clearPendingWallet = useCallback(() => {
    setWallet(null);
    setMnemonic('');
    setAddresses({ evmAddress: null, neoAddress: null });
  }, []);

  const exportWallet = useCallback(async (password) => {
    const secrets = await unlockSecrets(password);
    return {
      mnemonic: secrets.mnemonic,
      evmPrivateKey: secrets.evmPrivateKey,
      neoPrivateKey: secrets.neoPrivateKey,
      evmAddress: addresses.evmAddress,
      neoAddress: addresses.neoAddress
    };
  }, [addresses, unlockSecrets]);

  const createNewWallet = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const newWallet = await createWallet();
      setWallet(newWallet);
      setAddresses(newWallet.getAddresses());
      setMnemonic(newWallet.getMnemonic());
      setShowMnemonic(true);
      
      injectNeoProvider(newWallet);
      
      return newWallet;
    } catch (e) {
      setError(e.message);
      throw e;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const importWallet = useCallback(async (mnemonicPhrase) => {
    setIsLoading(true);
    setError(null);
    try {
      if (!validateMnemonic(mnemonicPhrase)) {
        throw new Error('Invalid mnemonic phrase');
      }
      
      const restoredWallet = await restoreWallet(mnemonicPhrase);
      setWallet(restoredWallet);
      setAddresses(restoredWallet.getAddresses());
      
      injectNeoProvider(restoredWallet);
      
      return restoredWallet;
    } catch (e) {
      setError(e.message);
      throw e;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const disconnect = useCallback(() => {
    if (wallet) {
      wallet.disconnect();
    }
    setWallet(null);
    setIsConnected(false);
    setAddresses({ evmAddress: null, neoAddress: null });
    setBalances({ NNET: 0 });
    setMnemonic('');
    setPrivateKeys(null);
    setHasPassword(false);
    secureStorage.remove('addresses');
    secureStorage.remove('encrypted');
  }, [wallet]);

  const signMessage = useCallback(async (message, mode = 'hybrid') => {
    if (!wallet || !isConnected) {
      throw new Error('Wallet not connected');
    }
    return wallet.signTransaction({ message }, mode);
  }, [wallet, isConnected]);

  const signTransaction = useCallback(async (tx, mode = 'hybrid') => {
    if (!wallet || !isConnected) {
      throw new Error('Wallet not connected');
    }
    return wallet.signTransaction(tx, mode);
  }, [wallet, isConnected]);

  const updateBalances = useCallback((newBalances) => {
    setBalances(prev => ({ ...prev, ...newBalances }));
    if (wallet) {
      wallet.setBalances(newBalances);
    }
  }, [wallet]);

  const value = {
    wallet,
    isConnected,
    isLoading,
    error,
    addresses,
    balances,
    tokens,
    mnemonic,
    showMnemonic,
    setShowMnemonic,
    createNewWallet,
    importWallet,
    disconnect,
    signMessage,
    hasPassword,
    privateKeys,
    setPassword,
    unlockSecrets,
    lockSecrets,
    clearPendingWallet,
    exportWallet,
    signTransaction,
    updateBalances,
    evmAddress: addresses.evmAddress,
    quantumAddress: addresses.neoAddress
  };

  return (
    <WalletContext.Provider value={value}>
      {children}
    </WalletContext.Provider>
  );
}

export default WalletProvider;
