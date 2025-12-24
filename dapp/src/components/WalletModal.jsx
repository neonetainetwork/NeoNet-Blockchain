import React, { useState, useEffect } from 'react';
import { useWallet } from './WalletProvider';
import { secureStorage } from '../lib/security';

const useIsMobile = () => {
  const [isMobile, setIsMobile] = useState(window.innerWidth < 480);
  
  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < 480);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);
  
  return isMobile;
};

const styles = {
  overlay: {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    background: 'rgba(0, 0, 0, 0.8)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1000
  },
  modal: {
    background: 'linear-gradient(135deg, #1a1a3a 0%, #0a0a2a 100%)',
    borderRadius: '20px',
    padding: '24px',
    maxWidth: '500px',
    width: '92%',
    maxHeight: '90vh',
    overflowY: 'auto',
    border: '1px solid rgba(0, 255, 200, 0.3)',
    boxShadow: '0 20px 60px rgba(0, 0, 0, 0.5)'
  },
  title: {
    fontSize: '24px',
    fontWeight: 'bold',
    marginBottom: '24px',
    background: 'linear-gradient(90deg, #00ffc8, #00a8ff)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    textAlign: 'center'
  },
  tabs: {
    display: 'flex',
    gap: '10px',
    marginBottom: '24px'
  },
  tab: {
    flex: 1,
    padding: '12px',
    borderRadius: '8px',
    border: '1px solid rgba(0, 255, 200, 0.2)',
    background: 'transparent',
    color: '#8892b0',
    cursor: 'pointer',
    transition: 'all 0.3s'
  },
  tabActive: {
    background: 'rgba(0, 255, 200, 0.1)',
    color: '#00ffc8',
    border: '1px solid rgba(0, 255, 200, 0.5)'
  },
  input: {
    width: '100%',
    padding: '14px 18px',
    borderRadius: '8px',
    border: '1px solid rgba(0, 255, 200, 0.2)',
    background: 'rgba(0, 0, 0, 0.3)',
    color: '#fff',
    fontSize: '14px',
    marginBottom: '16px',
    outline: 'none',
    resize: 'vertical',
    minHeight: '100px',
    fontFamily: 'monospace'
  },
  button: {
    width: '100%',
    padding: '14px',
    borderRadius: '8px',
    border: 'none',
    background: 'linear-gradient(90deg, #00ffc8, #00a8ff)',
    color: '#0a0a1a',
    fontSize: '16px',
    fontWeight: 'bold',
    cursor: 'pointer',
    transition: 'transform 0.3s, opacity 0.3s',
    marginBottom: '12px'
  },
  buttonSecondary: {
    background: 'transparent',
    border: '1px solid rgba(0, 255, 200, 0.3)',
    color: '#00ffc8'
  },
  mnemonicBox: {
    background: 'rgba(0, 0, 0, 0.4)',
    borderRadius: '12px',
    padding: '16px',
    marginBottom: '20px',
    border: '1px solid rgba(255, 200, 0, 0.3)',
    maxWidth: '100%',
    overflow: 'hidden'
  },
  mnemonicWords: {
    display: 'grid',
    gridTemplateColumns: 'repeat(4, 1fr)',
    gap: '6px',
    marginTop: '12px',
    width: '100%',
    boxSizing: 'border-box'
  },
  mnemonicWordsMobile: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: '6px',
    marginTop: '12px',
    width: '100%',
    boxSizing: 'border-box'
  },
  word: {
    background: 'rgba(0, 255, 200, 0.1)',
    padding: '6px 4px',
    borderRadius: '6px',
    fontSize: '10px',
    textAlign: 'center',
    fontFamily: 'monospace',
    whiteSpace: 'nowrap',
    overflow: 'visible',
    minWidth: '0',
    wordBreak: 'keep-all'
  },
  warning: {
    color: '#ffc800',
    fontSize: '13px',
    marginBottom: '16px',
    display: 'flex',
    alignItems: 'center',
    gap: '8px'
  },
  addressBox: {
    background: 'rgba(0, 0, 0, 0.3)',
    borderRadius: '8px',
    padding: '12px',
    marginBottom: '12px'
  },
  addressLabel: {
    fontSize: '12px',
    color: '#8892b0',
    marginBottom: '4px'
  },
  addressValue: {
    fontSize: '13px',
    fontFamily: 'monospace',
    wordBreak: 'break-all',
    color: '#00ffc8'
  },
  close: {
    position: 'absolute',
    top: '20px',
    right: '20px',
    background: 'transparent',
    border: 'none',
    color: '#8892b0',
    fontSize: '24px',
    cursor: 'pointer'
  },
  error: {
    color: '#ff4757',
    fontSize: '14px',
    marginBottom: '16px',
    textAlign: 'center'
  }
};

export function WalletModal({ isOpen, onClose }) {
  const [tab, setTab] = useState('create');
  const [importMnemonic, setImportMnemonic] = useState('');
  const [confirmed, setConfirmed] = useState(false);
  const [password, setPasswordInput] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showSetPassword, setShowSetPassword] = useState(false);
  const [showSecrets, setShowSecrets] = useState(false);
  const [unlockPassword, setUnlockPassword] = useState('');
  const [unlockError, setUnlockError] = useState('');
  const [tempWalletData, setTempWalletData] = useState(null);
  const [autoLockTimer, setAutoLockTimer] = useState(null);
  const [timeRemaining, setTimeRemaining] = useState(120);
  const isMobile = useIsMobile();
  
  const {
    createNewWallet,
    importWallet,
    isLoading,
    error,
    mnemonic,
    showMnemonic,
    setShowMnemonic,
    addresses,
    isConnected,
    hasPassword,
    privateKeys,
    setPassword,
    unlockSecrets,
    lockSecrets,
    clearPendingWallet,
    exportWallet,
    disconnect,
    wallet
  } = useWallet();

  React.useEffect(() => {
    if (!isOpen) {
      if (tempWalletData) {
        setTempWalletData(null);
      }
      if (showSetPassword) {
        clearPendingWallet();
        setShowSetPassword(false);
      }
      if (showMnemonic) {
        clearPendingWallet();
        setShowMnemonic(false);
      }
      if (showSecrets) {
        lockSecrets();
        setShowSecrets(false);
        setUnlockPassword('');
      }
      setPasswordInput('');
      setConfirmPassword('');
      setUnlockError('');
    }
  }, [isOpen, clearPendingWallet, lockSecrets, setShowMnemonic, showMnemonic, showSecrets, showSetPassword, tempWalletData]);

  React.useEffect(() => {
    return () => {
      if (autoLockTimer) clearInterval(autoLockTimer);
    };
  }, [autoLockTimer]);

  if (!isOpen) return null;

  const handleCreate = async () => {
    try {
      const newWallet = await createNewWallet();
      setTempWalletData({
        mnemonic: newWallet.getMnemonic(),
        evmPrivateKey: newWallet.evmPrivateKey,
        neoPrivateKey: newWallet.neoPrivateKey
      });
    } catch (e) {
      console.error(e);
    }
  };

  const handleImport = async () => {
    try {
      const restoredWallet = await importWallet(importMnemonic.trim());
      setTempWalletData({
        mnemonic: importMnemonic.trim(),
        evmPrivateKey: restoredWallet.evmPrivateKey,
        neoPrivateKey: restoredWallet.neoPrivateKey
      });
      setShowSetPassword(true);
    } catch (e) {
      console.error(e);
    }
  };

  const handleConfirmMnemonic = () => {
    setShowSetPassword(true);
    setShowMnemonic(false);
  };

  const checkPasswordStrength = (pwd) => {
    let strength = 0;
    if (pwd.length >= 8) strength++;
    if (pwd.length >= 12) strength++;
    if (/[a-z]/.test(pwd)) strength++;
    if (/[A-Z]/.test(pwd)) strength++;
    if (/[0-9]/.test(pwd)) strength++;
    if (/[^a-zA-Z0-9]/.test(pwd)) strength++;
    return strength;
  };

  const getPasswordStrengthLabel = (strength) => {
    if (strength <= 2) return { label: 'Weak', color: '#ff4444' };
    if (strength <= 4) return { label: 'Medium', color: '#ffaa00' };
    return { label: 'Strong', color: '#00ff88' };
  };

  const handleSetPassword = async () => {
    if (password !== confirmPassword) {
      setUnlockError('Passwords do not match');
      return;
    }
    if (password.length < 8) {
      setUnlockError('Password must be at least 8 characters');
      return;
    }
    if (!/[a-z]/.test(password)) {
      setUnlockError('Password must contain at least one lowercase letter');
      return;
    }
    if (!/[A-Z]/.test(password)) {
      setUnlockError('Password must contain at least one uppercase letter');
      return;
    }
    if (!/[0-9]/.test(password)) {
      setUnlockError('Password must contain at least one number');
      return;
    }
    if (!/[^a-zA-Z0-9]/.test(password)) {
      setUnlockError('Password must contain at least one special character (!@#$%^&*)');
      return;
    }
    try {
      const walletAddresses = {
        evmAddress: addresses.evmAddress,
        neoAddress: addresses.neoAddress,
        evmPublicKey: wallet?.evmPublicKey,
        neoPublicKey: wallet?.neoPublicKey
      };
      await setPassword(password, tempWalletData, walletAddresses);
      setPasswordInput('');
      setConfirmPassword('');
      setShowSetPassword(false);
      setTempWalletData(null);
      handleCloseModal();
    } catch (e) {
      setUnlockError(e.message);
    }
  };

  const handleCancelPasswordSetup = () => {
    setTempWalletData(null);
    setShowSetPassword(false);
    setPasswordInput('');
    setConfirmPassword('');
    setUnlockError('');
    clearPendingWallet();
  };

  const handleCloseModal = () => {
    if (tempWalletData) {
      setTempWalletData(null);
    }
    if (showSetPassword) {
      clearPendingWallet();
      setShowSetPassword(false);
    }
    if (showMnemonic) {
      clearPendingWallet();
      setShowMnemonic(false);
    }
    if (showSecrets) {
      lockSecrets();
      setShowSecrets(false);
      setUnlockPassword('');
    }
    setPasswordInput('');
    setConfirmPassword('');
    setUnlockError('');
    onClose();
  };

  const handleUnlock = async () => {
    setUnlockError('');
    try {
      await unlockSecrets(unlockPassword);
      setShowSecrets(true);
      setTimeRemaining(120);
      
      if (autoLockTimer) clearInterval(autoLockTimer);
      
      const timer = setInterval(() => {
        setTimeRemaining(prev => {
          if (prev <= 1) {
            clearInterval(timer);
            handleLock();
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
      setAutoLockTimer(timer);
    } catch (e) {
      setUnlockError('Invalid password');
    }
  };

  const handleExport = async () => {
    if (!privateKeys) return;
    const exportData = {
      mnemonic: privateKeys.mnemonic,
      evmPrivateKey: privateKeys.evmPrivateKey,
      neoPrivateKey: privateKeys.neoPrivateKey,
      evmAddress: addresses.evmAddress,
      neoAddress: addresses.neoAddress
    };
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'neonet-wallet-backup.json';
    a.click();
    URL.revokeObjectURL(url);
    handleLock();
  };

  const handleLock = () => {
    lockSecrets();
    setShowSecrets(false);
    setUnlockPassword('');
  };

  const handleDisconnect = () => {
    if (showSecrets) {
      lockSecrets();
      setShowSecrets(false);
    }
    setUnlockPassword('');
    setUnlockError('');
    secureStorage.clear();
    disconnect();
    onClose();
  };

  const mnemonicWords = mnemonic ? mnemonic.split(' ') : [];

  return (
    <div style={styles.overlay} onClick={handleCloseModal}>
      <div style={{ ...styles.modal, position: 'relative' }} onClick={e => e.stopPropagation()}>
        <button style={styles.close} onClick={handleCloseModal}>×</button>
        
        <h2 style={styles.title}>NeoNet Web4 Wallet</h2>
        
        {!showMnemonic && !isConnected && (
          <>
            <div style={styles.tabs}>
              <button
                style={{ ...styles.tab, ...(tab === 'create' ? styles.tabActive : {}) }}
                onClick={() => setTab('create')}
              >
                Create New
              </button>
              <button
                style={{ ...styles.tab, ...(tab === 'import' ? styles.tabActive : {}) }}
                onClick={() => setTab('import')}
              >
                Import Wallet
              </button>
            </div>

            {error && <div style={styles.error}>{error}</div>}

            {tab === 'create' && (
              <>
                <p style={{ color: '#8892b0', marginBottom: '20px', fontSize: '14px' }}>
                  Create a new NeoNet wallet with dual keys: EVM (0x...) for Ethereum compatibility 
                  and Quantum-safe (neo1...) for post-quantum security.
                </p>
                <button
                  style={styles.button}
                  onClick={handleCreate}
                  disabled={isLoading}
                >
                  {isLoading ? 'Creating...' : 'Generate New Wallet'}
                </button>
              </>
            )}

            {tab === 'import' && (
              <>
                <p style={{ color: '#8892b0', marginBottom: '16px', fontSize: '14px' }}>
                  Enter your 24-word recovery phrase to restore your wallet.
                </p>
                <textarea
                  style={styles.input}
                  placeholder="Enter your 24-word mnemonic phrase..."
                  value={importMnemonic}
                  onChange={(e) => setImportMnemonic(e.target.value)}
                />
                <button
                  style={styles.button}
                  onClick={handleImport}
                  disabled={isLoading || !importMnemonic.trim()}
                >
                  {isLoading ? 'Restoring...' : 'Import Wallet'}
                </button>
              </>
            )}
          </>
        )}

        {showMnemonic && mnemonic && (
          <>
            <div style={styles.warning}>
              <span>⚠️</span>
              <span>Write down these words in order and store them securely. Never share them!</span>
            </div>
            
            <div style={styles.mnemonicBox}>
              <div style={{ fontSize: '14px', fontWeight: 'bold', marginBottom: '8px' }}>
                Recovery Phrase (24 words)
              </div>
              <div style={isMobile ? styles.mnemonicWordsMobile : styles.mnemonicWords}>
                {mnemonicWords.map((word, i) => (
                  <div key={i} style={styles.word}>
                    <span style={{ color: '#8892b0', marginRight: '2px', fontSize: '9px' }}>{i + 1}.</span>
                    {word}
                  </div>
                ))}
              </div>
            </div>

            <div style={styles.addressBox}>
              <div style={styles.addressLabel}>EVM Address (Ethereum Compatible)</div>
              <div style={styles.addressValue}>{addresses.evmAddress}</div>
            </div>

            <div style={styles.addressBox}>
              <div style={styles.addressLabel}>Quantum Address (Post-Quantum Safe)</div>
              <div style={styles.addressValue}>{addresses.neoAddress}</div>
            </div>

            <button style={styles.button} onClick={handleConfirmMnemonic}>
              I've Saved My Recovery Phrase
            </button>
          </>
        )}

        {showSetPassword && (
          <>
            <h3 style={{ marginBottom: '16px', color: '#00ffc8' }}>Set Wallet Password (Required)</h3>
            <div style={styles.warning}>
              <span>⚠️</span>
              <span>You must set a password to secure your wallet. Your private keys will be encrypted.</span>
            </div>
            <p style={{ color: '#8892b0', marginBottom: '16px', fontSize: '14px' }}>
              Create a strong password. You'll need this password to view or export your private keys.
            </p>
            
            {unlockError && <div style={styles.error}>{unlockError}</div>}
            
            <input
              type="password"
              placeholder="Enter password (min 8 characters)"
              value={password}
              onChange={(e) => setPasswordInput(e.target.value)}
              style={{ ...styles.input, minHeight: 'auto', resize: 'none' }}
            />
            {password && (
              <div style={{ marginBottom: '10px' }}>
                <div style={{ display: 'flex', gap: '4px', marginBottom: '4px' }}>
                  {[1,2,3,4,5,6].map(i => (
                    <div key={i} style={{
                      flex: 1,
                      height: '4px',
                      borderRadius: '2px',
                      background: i <= checkPasswordStrength(password) 
                        ? getPasswordStrengthLabel(checkPasswordStrength(password)).color 
                        : '#333'
                    }} />
                  ))}
                </div>
                <div style={{ fontSize: '12px', color: getPasswordStrengthLabel(checkPasswordStrength(password)).color }}>
                  Password strength: {getPasswordStrengthLabel(checkPasswordStrength(password)).label}
                </div>
              </div>
            )}
            <input
              type="password"
              placeholder="Confirm password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              style={{ ...styles.input, minHeight: 'auto', resize: 'none' }}
            />
            
            <button style={styles.button} onClick={handleSetPassword}>
              Encrypt & Save Wallet
            </button>
            <button style={{ ...styles.button, ...styles.buttonSecondary, marginTop: '10px' }} onClick={handleCancelPasswordSetup}>
              Cancel (Wallet will not be saved)
            </button>
          </>
        )}

        {isConnected && !showMnemonic && !showSetPassword && (
          <>
            <div style={{ textAlign: 'center', marginBottom: '20px' }}>
              <div style={{ fontSize: '48px', marginBottom: '12px' }}>✓</div>
              <div style={{ color: '#00ffc8', fontSize: '18px' }}>Wallet Connected!</div>
            </div>

            <div style={styles.addressBox}>
              <div style={styles.addressLabel}>EVM Address</div>
              <div style={styles.addressValue}>{addresses.evmAddress}</div>
            </div>

            <div style={styles.addressBox}>
              <div style={styles.addressLabel}>Quantum Address</div>
              <div style={styles.addressValue}>{addresses.neoAddress}</div>
            </div>

            {hasPassword && !showSecrets && (
              <div style={{ marginTop: '20px', borderTop: '1px solid rgba(0,255,200,0.2)', paddingTop: '20px' }}>
                <h4 style={{ marginBottom: '12px', color: '#ffc800' }}>View Private Keys</h4>
                {unlockError && <div style={styles.error}>{unlockError}</div>}
                <input
                  type="password"
                  placeholder="Enter password to unlock"
                  value={unlockPassword}
                  onChange={(e) => setUnlockPassword(e.target.value)}
                  style={{ ...styles.input, minHeight: 'auto', resize: 'none' }}
                />
                <button style={{ ...styles.button, ...styles.buttonSecondary }} onClick={handleUnlock}>
                  Unlock Secrets
                </button>
              </div>
            )}

            {showSecrets && privateKeys && (
              <div style={{ marginTop: '20px', borderTop: '1px solid rgba(255,200,0,0.3)', paddingTop: '20px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                  <div style={styles.warning}>
                    <span>⚠️</span>
                    <span>Never share these keys!</span>
                  </div>
                  <div style={{ 
                    background: timeRemaining < 30 ? '#ff4444' : '#333', 
                    padding: '4px 10px', 
                    borderRadius: '4px',
                    fontSize: '12px',
                    color: '#fff'
                  }}>
                    Auto-lock: {Math.floor(timeRemaining / 60)}:{(timeRemaining % 60).toString().padStart(2, '0')}
                  </div>
                </div>
                
                <div style={styles.mnemonicBox}>
                  <div style={{ fontSize: '14px', fontWeight: 'bold', marginBottom: '8px' }}>Recovery Phrase</div>
                  <div style={isMobile ? styles.mnemonicWordsMobile : styles.mnemonicWords}>
                    {privateKeys.mnemonic?.split(' ').map((word, i) => (
                      <div key={i} style={styles.word}>
                        <span style={{ color: '#8892b0', marginRight: '2px', fontSize: '9px' }}>{i + 1}.</span>
                        {word}
                      </div>
                    ))}
                  </div>
                </div>

                <div style={styles.addressBox}>
                  <div style={styles.addressLabel}>EVM Private Key (Ethereum Compatible)</div>
                  <div style={{ ...styles.addressValue, color: '#ffc800', fontSize: '11px', wordBreak: 'break-all' }}>
                    0x{privateKeys.evmPrivateKey}
                  </div>
                </div>

                <div style={styles.addressBox}>
                  <div style={styles.addressLabel}>Quantum Private Key (Ed25519)</div>
                  <div style={{ ...styles.addressValue, color: '#ffc800', fontSize: '11px', wordBreak: 'break-all' }}>
                    {privateKeys.neoPrivateKey}
                  </div>
                </div>

                <div style={{ display: 'flex', gap: '10px' }}>
                  <button style={{ ...styles.button, background: '#4CAF50' }} onClick={handleExport}>
                    Export Backup
                  </button>
                  <button style={{ ...styles.button, ...styles.buttonSecondary }} onClick={handleLock}>
                    Lock
                  </button>
                </div>
              </div>
            )}

            <div style={{ display: 'flex', gap: '10px', marginTop: '16px' }}>
              <button style={{ ...styles.button, flex: 1 }} onClick={handleCloseModal}>
                Continue to Dashboard
              </button>
              <button 
                style={{ ...styles.button, ...styles.buttonSecondary, flex: 1, background: '#ff4444', borderColor: '#ff4444' }} 
                onClick={handleDisconnect}
              >
                Disconnect Wallet
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default WalletModal;
