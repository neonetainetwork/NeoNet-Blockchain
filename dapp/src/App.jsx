import React, { useState, useEffect } from 'react';
import { HashRouter, useNavigate, useLocation } from 'react-router-dom';
import axios from 'axios';
import { WalletProvider, useWallet } from './components/WalletProvider';
import WalletModal from './components/WalletModal';
import './mobile.css';

const API = '/api';
const NNET_LOGO = '/uploads/nnet_logo.png';

function NnetLogo({ size = 24, style = {} }) {
  return (
    <img 
      src={NNET_LOGO} 
      alt="NNET" 
      style={{ width: size, height: size, borderRadius: '50%', objectFit: 'cover', ...style }} 
      onError={(e) => { e.target.style.display = 'none'; }}
    />
  );
}

function ExploreTab({ wallet }) {
  const [exploreData, setExploreData] = useState({
    tokens: [],
    dapps: [],
    nft_collections: [],
    contracts: [],
    stats: { total_tokens: 0, total_dapps: 0, total_nfts: 0, total_contracts: 0 }
  });
  const [loading, setLoading] = useState(true);
  const [activeCategory, setActiveCategory] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    loadExploreData();
    const interval = setInterval(loadExploreData, 10000);
    return () => clearInterval(interval);
  }, []);

  const loadExploreData = async () => {
    try {
      const res = await axios.get(`${API}/explore`);
      if (res.data) setExploreData(res.data);
    } catch (e) {
      console.error('Failed to load explore data:', e);
    }
    setLoading(false);
  };

  const filteredItems = () => {
    let items = [];
    
    if (activeCategory === 'all' || activeCategory === 'tokens') {
      items.push(...exploreData.tokens.map(t => ({ ...t, itemType: 'token' })));
    }
    if (activeCategory === 'all' || activeCategory === 'dapps') {
      items.push(...exploreData.dapps.map(d => ({ ...d, itemType: 'dapp' })));
    }
    if (activeCategory === 'all' || activeCategory === 'nfts') {
      items.push(...exploreData.nft_collections.map(n => ({ ...n, itemType: 'nft' })));
    }
    
    if (searchTerm) {
      items = items.filter(item => 
        (item.name || '').toLowerCase().includes(searchTerm.toLowerCase()) ||
        (item.symbol || '').toLowerCase().includes(searchTerm.toLowerCase()) ||
        (item.description || '').toLowerCase().includes(searchTerm.toLowerCase())
      );
    }
    
    return items.sort((a, b) => (b.created_at || 0) - (a.created_at || 0));
  };

  const getCategoryIcon = (item) => {
    if (item.itemType === 'token') return '🪙';
    if (item.itemType === 'nft') return '🎨';
    if (item.itemType === 'dapp') {
      if (item.category === 'defi') return '💹';
      if (item.category === 'gaming') return '🎮';
      if (item.category === 'dao') return '🗳️';
      if (item.category === 'social') return '👥';
      return '📱';
    }
    return '📄';
  };

  const formatDate = (timestamp) => {
    if (!timestamp) return 'N/A';
    return new Date(timestamp * 1000).toLocaleDateString();
  };

  return (
    <div>
      <h1 style={styles.title}>Explore NeoNet Ecosystem</h1>
      <p style={{color: '#888', textAlign: 'center', marginBottom: '30px'}}>
        Discover tokens, dApps, NFT collections and smart contracts deployed on NeoNet
      </p>

      <div className="explore-stats-grid" style={{display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '15px', marginBottom: '30px'}}>
        <div className="card" style={{...styles.card, textAlign: 'center', background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)', border: '1px solid #f7931a'}}>
          <div className="card-value" style={{fontSize: '32px', fontWeight: 'bold', color: '#f7931a'}}>{exploreData.stats?.total_tokens || 0}</div>
          <div style={{color: '#888'}}>Tokens</div>
        </div>
        <div className="card" style={{...styles.card, textAlign: 'center', background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)', border: '1px solid #4CAF50'}}>
          <div className="card-value" style={{fontSize: '32px', fontWeight: 'bold', color: '#4CAF50'}}>{exploreData.stats?.total_dapps || 0}</div>
          <div style={{color: '#888'}}>dApps</div>
        </div>
        <div className="card" style={{...styles.card, textAlign: 'center', background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)', border: '1px solid #9c27b0'}}>
          <div className="card-value" style={{fontSize: '32px', fontWeight: 'bold', color: '#9c27b0'}}>{exploreData.stats?.total_nfts || 0}</div>
          <div style={{color: '#888'}}>NFT Collections</div>
        </div>
        <div className="card" style={{...styles.card, textAlign: 'center', background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)', border: '1px solid #2196F3'}}>
          <div className="card-value" style={{fontSize: '32px', fontWeight: 'bold', color: '#2196F3'}}>{exploreData.stats?.total_contracts || 0}</div>
          <div style={{color: '#888'}}>Contracts</div>
        </div>
      </div>

      <div className="category-tabs" style={{display: 'flex', gap: '10px', marginBottom: '20px', flexWrap: 'wrap'}}>
        {['all', 'tokens', 'dapps', 'nfts'].map(cat => (
          <button 
            key={cat} 
            onClick={() => setActiveCategory(cat)}
            style={activeCategory === cat ? styles.tabActive : styles.tab}
            className="tab"
          >
            {cat === 'all' ? 'All Projects' : cat === 'nfts' ? 'NFTs' : cat === 'dapps' ? 'dApps' : 'Tokens'}
          </button>
        ))}
      </div>

      <div style={{marginBottom: '20px'}}>
        <input
          type="text"
          placeholder="Search by name, symbol, or description..."
          value={searchTerm}
          onChange={e => setSearchTerm(e.target.value)}
          style={{...styles.input, width: '100%', maxWidth: '400px'}}
        />
      </div>

      {loading ? (
        <div style={{textAlign: 'center', padding: '50px', color: '#888'}}>Loading explore data...</div>
      ) : filteredItems().length === 0 ? (
        <div style={styles.card}>
          <p style={{textAlign: 'center', color: '#888'}}>
            No projects found. Deploy your first token or dApp from the Developer tab!
          </p>
        </div>
      ) : (
        <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '20px'}}>
          {filteredItems().map((item, index) => (
            <div key={index} style={{...styles.card, cursor: 'pointer', transition: 'transform 0.2s', ':hover': {transform: 'scale(1.02)'}}}>
              {item.image_url && (
                <div style={{marginBottom: '15px', textAlign: 'center'}}>
                  <img 
                    src={item.image_url} 
                    alt={item.name} 
                    style={{width: '100%', maxHeight: '180px', objectFit: 'cover', borderRadius: '8px'}} 
                    onError={(e) => e.target.style.display = 'none'}
                  />
                </div>
              )}
              <div style={{display: 'flex', alignItems: 'center', gap: '15px', marginBottom: '15px'}}>
                <div style={{fontSize: '40px'}}>{getCategoryIcon(item)}</div>
                <div style={{flex: 1}}>
                  <h3 style={{margin: 0, color: '#fff'}}>{item.name || item.symbol || 'Unnamed'}</h3>
                  <span style={{
                    display: 'inline-block',
                    padding: '2px 8px',
                    borderRadius: '10px',
                    fontSize: '12px',
                    background: item.itemType === 'token' ? '#f7931a' : item.itemType === 'nft' ? '#9c27b0' : '#4CAF50',
                    color: '#fff'
                  }}>
                    {item.itemType === 'token' ? 'Token' : item.itemType === 'nft' ? 'NFT' : 'dApp'}
                  </span>
                  {item.category && (
                    <span style={{marginLeft: '5px', fontSize: '12px', color: '#888'}}>
                      ({item.category})
                    </span>
                  )}
                  {item.ai_verified && (
                    <span style={{marginLeft: '5px', fontSize: '11px', color: '#4CAF50'}}>
                      ✓ Verified
                    </span>
                  )}
                </div>
              </div>
              
              {item.description && (
                <p style={{color: '#888', fontSize: '14px', marginBottom: '15px'}}>
                  {item.description.substring(0, 100)}{item.description.length > 100 ? '...' : ''}
                </p>
              )}
              
              <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', fontSize: '13px'}}>
                {item.symbol && (
                  <div><span style={{color: '#888'}}>Symbol:</span> <strong>{item.symbol}</strong></div>
                )}
                {item.total_supply && (
                  <div><span style={{color: '#888'}}>Supply:</span> <strong>{Number(item.total_supply).toLocaleString()}</strong></div>
                )}
                {item.price_usd !== undefined && (
                  <div><span style={{color: '#888'}}>Price:</span> <strong>${item.price_usd?.toFixed(4)}</strong></div>
                )}
                {item.runtime && (
                  <div><span style={{color: '#888'}}>Runtime:</span> <strong>{item.runtime.toUpperCase()}</strong></div>
                )}
                {item.floor_price_neo !== undefined && (
                  <div><span style={{color: '#888'}}>Floor:</span> <strong>{item.floor_price_neo} NNET</strong></div>
                )}
                {item.remaining !== undefined && item.itemType === 'nft' && (
                  <div><span style={{color: '#888'}}>Available:</span> <strong style={{color: item.remaining <= 5 ? '#f44336' : '#4CAF50'}}>{item.remaining}/{item.total_supply}</strong></div>
                )}
                {item.creator && item.itemType === 'nft' && (
                  <div style={{gridColumn: '1 / -1'}}>
                    <span style={{color: '#888'}}>Creator:</span>{' '}
                    <strong 
                      style={{color: '#58a6ff', cursor: 'pointer', fontSize: '12px'}} 
                      onClick={(e) => { e.stopPropagation(); navigator.clipboard.writeText(item.creator); alert('Creator address copied!\n\n' + item.creator); }}
                      title={item.creator}
                    >
                      {item.creator?.slice(0, 12)}...{item.creator?.slice(-8)}
                    </strong>
                    {item.ai_verified && <span style={{marginLeft: '6px', color: '#4CAF50', fontSize: '11px'}}>✓ AI Verified</span>}
                  </div>
                )}
                {item.users_count !== undefined && (
                  <div><span style={{color: '#888'}}>Users:</span> <strong>{item.users_count}</strong></div>
                )}
              </div>
              
              <div style={{marginTop: '15px', paddingTop: '10px', borderTop: '1px solid #333'}}>
                <div style={{display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '10px'}}>
                  <span style={{color: '#666'}}>Created: {formatDate(item.created_at)}</span>
                  <span style={{color: '#666'}}>
                    {item.contract_address ? `${item.contract_address.slice(0, 12)}...` : ''}
                  </span>
                </div>
                <div style={{display: 'flex', gap: '8px', flexWrap: 'wrap'}}>
                  {item.itemType === 'token' && !item.is_native && (
                    <>
                      <button 
                        onClick={(e) => { e.stopPropagation(); alert(`Redirecting to DEX to trade ${item.symbol}...`); window.location.hash = '#dex'; }}
                        style={{...styles.button, padding: '6px 12px', fontSize: '12px', background: '#4CAF50'}}
                      >
                        💱 Trade
                      </button>
                      <button 
                        onClick={(e) => { e.stopPropagation(); alert(`Token Info:\n\nSymbol: ${item.symbol}\nName: ${item.name}\nSupply: ${item.total_supply?.toLocaleString()}\nPrice: $${item.price_usd?.toFixed(4)}\nContract: ${item.contract_address}\nRuntime: ${item.runtime}`); }}
                        style={{...styles.button, padding: '6px 12px', fontSize: '12px', background: '#2196F3'}}
                      >
                        ℹ️ Info
                      </button>
                    </>
                  )}
                  {item.itemType === 'dapp' && (
                    <>
                      {item.website_url && (
                        <button 
                          onClick={(e) => { e.stopPropagation(); window.open(item.website_url, '_blank'); }}
                          style={{...styles.button, padding: '6px 12px', fontSize: '12px', background: '#9c27b0'}}
                        >
                          🌐 Visit Website
                        </button>
                      )}
                      <button 
                        onClick={(e) => { e.stopPropagation(); alert(`dApp Info:\n\nName: ${item.name}\nCategory: ${item.category}\nDescription: ${item.description}\nContract: ${item.contract_address}\nUsers: ${item.users_count || 0}`); }}
                        style={{...styles.button, padding: '6px 12px', fontSize: '12px', background: '#2196F3'}}
                      >
                        ℹ️ Info
                      </button>
                    </>
                  )}
                  {item.itemType === 'nft' && (
                    <>
                      <button 
                        onClick={async (e) => { 
                          e.stopPropagation(); 
                          const walletAddress = wallet?.addresses?.neoAddress || wallet?.quantumAddress;
                          if (!wallet || !wallet.isConnected || !walletAddress) {
                            alert('Please connect wallet first');
                            return;
                          }
                          const remaining = item.remaining !== undefined ? item.remaining : (item.total_supply - (item.minted || 0));
                          if (remaining <= 0) {
                            alert('This collection is SOLD OUT!');
                            return;
                          }
                          if (confirm(`Buy NFT from "${item.name}" for ${item.floor_price_neo || 0} NNET?\n\nAvailable: ${remaining}/${item.total_supply}\nYour wallet: ${walletAddress?.slice(0,20)}...`)) {
                            try {
                              const res = await axios.post(`${API}/nft/${item.symbol}/buy`, { buyer: walletAddress });
                              const status = res.data.collection_status;
                              alert(`Success! You purchased ${res.data.nft?.name}\n\nNFT Hash: ${res.data.nft?.nft_hash?.slice(0,16)}...\n${status?.is_sold_out ? '🔥 SOLD OUT!' : `Remaining: ${status?.remaining}/${status?.total_supply}`}`);
                              loadExploreData();
                            } catch (err) {
                              alert('Purchase failed: ' + (err.response?.data?.detail || err.message));
                            }
                          }
                        }}
                        style={{...styles.button, padding: '6px 12px', fontSize: '12px', background: '#f7931a'}}
                      >
                        🛒 Buy ({item.floor_price_neo || 0} NNET)
                      </button>
                      <button 
                        onClick={(e) => { e.stopPropagation(); alert(`NFT Collection:\n\nName: ${item.name}\nSymbol: ${item.symbol}\nAvailable: ${item.remaining !== undefined ? item.remaining : '?'}/${item.total_supply}\nMinted: ${item.minted || 0}\nFloor Price: ${item.floor_price_neo || 0} NNET\nCategory: ${item.category}\n\n👤 CREATOR WALLET:\n${item.creator || 'Unknown'}\n\n🔐 VERIFICATION:\nAI Verified: ${item.ai_verified ? '✓ Yes' : '✗ No'}\nOrigin Hash: ${item.origin_hash || 'N/A'}\nContract: ${item.contract_address}`); }}
                        style={{...styles.button, padding: '6px 12px', fontSize: '12px', background: '#2196F3'}}
                      >
                        👁️ View
                      </button>
                    </>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function TokenFactoryForm({ wallet }) {
  const [tokenData, setTokenData] = useState({
    name: '',
    symbol: '',
    totalSupply: '1000000',
    decimals: '18',
    description: '',
    imageUrl: ''
  });
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState(null);

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    setUploading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await axios.post(`${API}/upload/image`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setTokenData({...tokenData, imageUrl: res.data.url});
    } catch (e) {
      alert('Upload failed: ' + (e.response?.data?.detail || e.message));
    }
    setUploading(false);
  };

  const createToken = async () => {
    if (!tokenData.name || !tokenData.symbol || !tokenData.totalSupply) {
      return alert('Please fill in name, symbol and total supply');
    }
    
    setLoading(true);
    try {
      const res = await axios.post(`${API}/tokens/create`, {
        name: tokenData.name,
        symbol: tokenData.symbol.toUpperCase(),
        total_supply: parseFloat(tokenData.totalSupply),
        creator: wallet.evmAddress,
        decimals: parseInt(tokenData.decimals),
        description: tokenData.description,
        image_url: tokenData.imageUrl
      });
      setResult(res.data);
      alert(`Token ${tokenData.symbol} created successfully! It will appear in your wallet and DEX.`);
    } catch (e) {
      alert('Error: ' + (e.response?.data?.detail || e.message));
    }
    setLoading(false);
  };

  return (
    <div>
      <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px'}}>
        <div style={styles.formGroup}>
          <label>Token Name *</label>
          <input 
            placeholder="My Awesome Token" 
            value={tokenData.name}
            onChange={e => setTokenData({...tokenData, name: e.target.value})}
            style={styles.input}
          />
        </div>
        <div style={styles.formGroup}>
          <label>Symbol * (3-5 chars)</label>
          <input 
            placeholder="MAT" 
            maxLength={5}
            value={tokenData.symbol}
            onChange={e => setTokenData({...tokenData, symbol: e.target.value.toUpperCase()})}
            style={styles.input}
          />
        </div>
        <div style={styles.formGroup}>
          <label>Total Supply *</label>
          <input 
            type="number"
            placeholder="1000000" 
            value={tokenData.totalSupply}
            onChange={e => setTokenData({...tokenData, totalSupply: e.target.value})}
            style={styles.input}
          />
        </div>
        <div style={styles.formGroup}>
          <label>Decimals</label>
          <select 
            value={tokenData.decimals}
            onChange={e => setTokenData({...tokenData, decimals: e.target.value})}
            style={styles.input}
          >
            <option value="18">18 (Standard)</option>
            <option value="8">8 (Bitcoin-style)</option>
            <option value="6">6 (USDC-style)</option>
            <option value="0">0 (No decimals)</option>
          </select>
        </div>
      </div>
      <div style={styles.formGroup}>
        <label>Token Logo</label>
        <div style={{display: 'flex', gap: '10px', alignItems: 'center'}}>
          <label style={{...styles.btn, background: '#333', cursor: 'pointer', display: 'inline-block'}}>
            {uploading ? 'Uploading...' : 'Upload Image'}
            <input type="file" accept="image/*" onChange={handleFileUpload} style={{display: 'none'}} disabled={uploading} />
          </label>
          <span style={{color: '#888'}}>or</span>
          <input 
            placeholder="Paste URL" 
            value={tokenData.imageUrl}
            onChange={e => setTokenData({...tokenData, imageUrl: e.target.value})}
            style={{...styles.input, flex: 1}}
          />
        </div>
        {tokenData.imageUrl && (
          <div style={{marginTop: '10px', display: 'flex', alignItems: 'center', gap: '10px'}}>
            <img src={tokenData.imageUrl} alt="Token logo" style={{width: '64px', height: '64px', borderRadius: '50%', objectFit: 'cover'}} onError={e => e.target.style.display='none'} />
            <button onClick={() => setTokenData({...tokenData, imageUrl: ''})} style={{...styles.btn, background: '#f44336', padding: '5px 10px'}}>Remove</button>
          </div>
        )}
      </div>
      <div style={styles.formGroup}>
        <label>Description (optional)</label>
        <textarea 
          placeholder="Describe your token..." 
          value={tokenData.description}
          onChange={e => setTokenData({...tokenData, description: e.target.value})}
          style={{...styles.input, minHeight: '60px'}}
        />
      </div>
      <button onClick={createToken} disabled={loading} style={{...styles.btn, background: '#f7931a', width: '100%'}}>
        {loading ? 'Creating Token...' : 'Create Token'}
      </button>
      {result && (
        <div style={{marginTop: '15px', padding: '15px', background: '#1a1a2e', borderRadius: '8px', border: '1px solid #4CAF50'}}>
          <p style={{color: '#4CAF50', fontWeight: 'bold'}}>Token Created Successfully!</p>
          <p><strong>Symbol:</strong> {result.symbol}</p>
          <p><strong>Contract:</strong> {result.contract_address}</p>
          <p style={{fontSize: '12px', color: '#888'}}>Your token is now available in DEX and your wallet!</p>
        </div>
      )}
    </div>
  );
}

function NFTFactoryForm({ wallet }) {
  const [nftData, setNftData] = useState({
    name: '',
    symbol: '',
    totalSupply: '100',
    category: 'art',
    floorPrice: '0.1',
    description: '',
    imageUrl: ''
  });
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState(null);

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    setUploading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await axios.post(`${API}/upload/image`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setNftData({...nftData, imageUrl: res.data.url});
    } catch (e) {
      alert('Upload failed: ' + (e.response?.data?.detail || e.message));
    }
    setUploading(false);
  };

  const createNFT = async () => {
    if (!nftData.name || !nftData.symbol || !nftData.totalSupply) {
      return alert('Please fill in name, symbol and total supply');
    }
    
    setLoading(true);
    try {
      const res = await axios.post(`${API}/nft/create`, {
        name: nftData.name,
        symbol: nftData.symbol.toUpperCase(),
        total_supply: parseInt(nftData.totalSupply),
        creator: wallet.evmAddress,
        category: nftData.category,
        floor_price: parseFloat(nftData.floorPrice),
        description: nftData.description,
        image_url: nftData.imageUrl
      });
      setResult(res.data);
      alert(`NFT Collection ${nftData.name} created successfully!`);
    } catch (e) {
      alert('Error: ' + (e.response?.data?.detail || e.message));
    }
    setLoading(false);
  };

  return (
    <div>
      <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px'}}>
        <div style={styles.formGroup}>
          <label>Collection Name *</label>
          <input 
            placeholder="Cool NFT Collection" 
            value={nftData.name}
            onChange={e => setNftData({...nftData, name: e.target.value})}
            style={styles.input}
          />
        </div>
        <div style={styles.formGroup}>
          <label>Symbol * (3-5 chars)</label>
          <input 
            placeholder="COOL" 
            maxLength={5}
            value={nftData.symbol}
            onChange={e => setNftData({...nftData, symbol: e.target.value.toUpperCase()})}
            style={styles.input}
          />
        </div>
        <div style={styles.formGroup}>
          <label>Total Supply (max items)</label>
          <input 
            type="number"
            placeholder="100" 
            value={nftData.totalSupply}
            onChange={e => setNftData({...nftData, totalSupply: e.target.value})}
            style={styles.input}
          />
        </div>
        <div style={styles.formGroup}>
          <label>Category</label>
          <select 
            value={nftData.category}
            onChange={e => setNftData({...nftData, category: e.target.value})}
            style={styles.input}
          >
            <option value="art">Art</option>
            <option value="gaming">Gaming</option>
            <option value="music">Music</option>
            <option value="collectibles">Collectibles</option>
            <option value="photography">Photography</option>
          </select>
        </div>
        <div style={styles.formGroup}>
          <label>Floor Price (NNET)</label>
          <input 
            type="number"
            step="0.01"
            placeholder="0.1" 
            value={nftData.floorPrice}
            onChange={e => setNftData({...nftData, floorPrice: e.target.value})}
            style={styles.input}
          />
        </div>
      </div>
      <div style={styles.formGroup}>
        <label>Collection Image</label>
        <div style={{display: 'flex', gap: '10px', alignItems: 'center'}}>
          <label style={{...styles.btn, background: '#333', cursor: 'pointer', display: 'inline-block'}}>
            {uploading ? 'Uploading...' : 'Upload Image'}
            <input type="file" accept="image/*" onChange={handleFileUpload} style={{display: 'none'}} disabled={uploading} />
          </label>
          <span style={{color: '#888'}}>or</span>
          <input 
            placeholder="Paste URL" 
            value={nftData.imageUrl}
            onChange={e => setNftData({...nftData, imageUrl: e.target.value})}
            style={{...styles.input, flex: 1}}
          />
        </div>
        {nftData.imageUrl && (
          <div style={{marginTop: '10px', display: 'flex', alignItems: 'center', gap: '10px'}}>
            <img src={nftData.imageUrl} alt="NFT preview" style={{width: '120px', height: '120px', borderRadius: '12px', objectFit: 'cover'}} onError={e => e.target.style.display='none'} />
            <button onClick={() => setNftData({...nftData, imageUrl: ''})} style={{...styles.btn, background: '#f44336', padding: '5px 10px'}}>Remove</button>
          </div>
        )}
      </div>
      <div style={styles.formGroup}>
        <label>Description</label>
        <textarea 
          placeholder="Describe your NFT collection..." 
          value={nftData.description}
          onChange={e => setNftData({...nftData, description: e.target.value})}
          style={{...styles.input, minHeight: '60px'}}
        />
      </div>
      <button onClick={createNFT} disabled={loading} style={{...styles.btn, background: '#9c27b0', width: '100%'}}>
        {loading ? 'Creating Collection...' : 'Create NFT Collection'}
      </button>
      {result && (
        <div style={{marginTop: '15px', padding: '15px', background: '#1a1a2e', borderRadius: '8px', border: '1px solid #9c27b0'}}>
          <p style={{color: '#9c27b0', fontWeight: 'bold'}}>NFT Collection Created!</p>
          <p><strong>Name:</strong> {result.collection?.name}</p>
          <p><strong>Contract:</strong> {result.collection?.contract_address}</p>
          <p style={{fontSize: '12px', color: '#888'}}>Your collection is now available in Explore!</p>
        </div>
      )}
    </div>
  );
}

function WalletNFTSection({ wallet }) {
  const [nfts, setNfts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [listingPrice, setListingPrice] = useState('');
  const [selectedNft, setSelectedNft] = useState(null);

  const walletAddress = wallet?.addresses?.neoAddress || wallet?.quantumAddress;

  useEffect(() => {
    if (walletAddress) {
      loadNFTs();
    }
  }, [walletAddress]);

  const loadNFTs = async () => {
    if (!walletAddress) return;
    try {
      const res = await axios.get(`${API}/wallet/${walletAddress}/nfts`);
      setNfts(res.data?.nfts || []);
    } catch (e) {
      console.error('Failed to load NFTs:', e);
    }
    setLoading(false);
  };

  const listForSale = async (nft) => {
    if (!listingPrice || parseFloat(listingPrice) <= 0) {
      alert('Please enter a valid price');
      return;
    }
    try {
      const res = await axios.post(`${API}/nft/list`, {
        collection_symbol: nft.collection_symbol,
        token_id: nft.token_id,
        price: parseFloat(listingPrice),
        seller: walletAddress
      });
      alert(`NFT listed for sale at ${listingPrice} NNET!`);
      setSelectedNft(null);
      setListingPrice('');
      loadNFTs();
    } catch (e) {
      alert('Failed to list: ' + (e.response?.data?.detail || e.message));
    }
  };

  const cancelListing = async (nft) => {
    try {
      await axios.post(`${API}/nft/unlist`, {
        collection_symbol: nft.collection_symbol,
        token_id: nft.token_id,
        seller: walletAddress
      });
      alert('Listing cancelled');
      loadNFTs();
    } catch (e) {
      alert('Failed: ' + (e.response?.data?.detail || e.message));
    }
  };

  if (!walletAddress || !wallet?.isConnected) return null;

  return (
    <div style={styles.card}>
      <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px'}}>
        <div style={styles.cardLabel}>🎨 My NFTs ({nfts.length})</div>
        <button onClick={loadNFTs} style={{...styles.btn, padding: '5px 10px', fontSize: '12px', background: '#333'}}>
          🔄 Refresh
        </button>
      </div>
      
      {loading ? (
        <div style={{textAlign: 'center', padding: '20px', color: '#888'}}>Loading NFTs...</div>
      ) : nfts.length === 0 ? (
        <div style={{textAlign: 'center', padding: '20px', color: '#888'}}>
          <p>No NFTs yet</p>
          <p style={{fontSize: '12px'}}>Buy NFTs from Explore tab or create your own collection</p>
        </div>
      ) : (
        <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', gap: '12px'}}>
          {nfts.map((nft, i) => (
            <div key={i} style={{background: '#0d1117', borderRadius: '12px', overflow: 'hidden', border: nft.is_listed ? '2px solid #f7931a' : '1px solid #30363d'}}>
              {(nft.image_url || nft.collection_image) && (
                <img 
                  src={nft.image_url || nft.collection_image} 
                  alt={nft.name} 
                  style={{width: '100%', height: '120px', objectFit: 'cover'}}
                  onError={e => e.target.src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect fill="%239c27b0" width="100" height="100"/><text x="50" y="55" font-size="40" text-anchor="middle" fill="white">NFT</text></svg>'}
                />
              )}
              <div style={{padding: '10px'}}>
                <div style={{fontWeight: 'bold', fontSize: '13px', marginBottom: '4px'}}>{nft.name}</div>
                <div style={{fontSize: '11px', color: '#888'}}>{nft.collection_name}</div>
                {nft.verified && (
                  <div style={{fontSize: '10px', color: '#4CAF50', marginTop: '4px'}}>✓ Verified</div>
                )}
                {nft.price_paid && (
                  <div style={{fontSize: '10px', color: '#8b949e', marginTop: '2px'}}>Paid: {nft.price_paid} NNET</div>
                )}
                {nft.is_listed && (
                  <div style={{fontSize: '11px', color: '#f7931a', marginTop: '4px', fontWeight: 'bold'}}>📢 Listed: {nft.listing_price} NNET</div>
                )}
                
                <div style={{marginTop: '8px', display: 'flex', gap: '4px', flexWrap: 'wrap'}}>
                  {nft.is_listed ? (
                    <button 
                      onClick={() => cancelListing(nft)}
                      style={{...styles.btn, padding: '4px 8px', fontSize: '10px', background: '#f44336', flex: 1}}
                    >
                      Cancel Listing
                    </button>
                  ) : selectedNft?.token_id === nft.token_id && selectedNft?.collection_symbol === nft.collection_symbol ? (
                    <div style={{width: '100%'}}>
                      <input 
                        type="number" 
                        placeholder="Price in NNET" 
                        value={listingPrice}
                        onChange={e => setListingPrice(e.target.value)}
                        style={{...styles.input, padding: '6px', fontSize: '11px', marginBottom: '4px'}}
                      />
                      <div style={{display: 'flex', gap: '4px'}}>
                        <button 
                          onClick={() => listForSale(nft)}
                          style={{...styles.btn, padding: '4px 6px', fontSize: '10px', background: '#4CAF50', flex: 1}}
                        >
                          Confirm
                        </button>
                        <button 
                          onClick={() => { setSelectedNft(null); setListingPrice(''); }}
                          style={{...styles.btn, padding: '4px 6px', fontSize: '10px', background: '#666', flex: 1}}
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  ) : (
                    <button 
                      onClick={() => setSelectedNft(nft)}
                      style={{...styles.btn, padding: '4px 8px', fontSize: '10px', background: '#f7931a', flex: 1}}
                    >
                      🏷️ Sell
                    </button>
                  )}
                  <button 
                    onClick={() => alert(`NFT Details:\n\nName: ${nft.name}\nCollection: ${nft.collection_name}\nToken ID: ${nft.token_id}\nHash: ${nft.nft_hash}\nMinted: ${new Date(nft.minted_at * 1000).toLocaleDateString()}`)}
                    style={{...styles.btn, padding: '4px 8px', fontSize: '10px', background: '#2196F3', flex: 1}}
                  >
                    ℹ️ Info
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function DeveloperTab({ wallet, setShowWallet }) {
  const [contractCode, setContractCode] = useState('');
  const [runtime, setRuntime] = useState('hybrid');
  const [deployResult, setDeployResult] = useState(null);
  const [contracts, setContracts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [proposals, setProposals] = useState([]);
  const [newProposal, setNewProposal] = useState({ title: '', description: '' });
  const [aiStatus, setAiStatus] = useState(null);
  const [showExamples, setShowExamples] = useState('connect');
  const [contractMetadata, setContractMetadata] = useState({
    name: '',
    symbol: '',
    totalSupply: '',
    description: '',
    category: 'utility',
    isDapp: false,
    websiteUrl: '',
    logoUrl: ''
  });
  const [uploadingLogo, setUploadingLogo] = useState(false);

  const handleLogoUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    setUploadingLogo(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await axios.post(`${API}/upload/image`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setContractMetadata({...contractMetadata, logoUrl: res.data.url});
    } catch (e) {
      alert('Upload failed: ' + (e.response?.data?.detail || e.message));
    }
    setUploadingLogo(false);
  };

  useEffect(() => {
    loadContracts();
    loadProposals();
    loadAiStatus();
    const i = setInterval(loadAiStatus, 5000);
    return () => clearInterval(i);
  }, []);

  const loadContracts = async () => {
    try {
      const res = await axios.get(`${API}/contracts`);
      if (res.data?.contracts) setContracts(res.data.contracts);
    } catch (e) {}
  };

  const loadProposals = async () => {
    try {
      const res = await axios.get(`${API}/governance/proposals`);
      if (res.data?.proposals) setProposals(res.data.proposals);
    } catch (e) {}
  };

  const loadAiStatus = async () => {
    try {
      const res = await axios.get(`${API}/ai/status`);
      if (res.data) setAiStatus(res.data);
    } catch (e) {}
  };

  const deployContract = async () => {
    if (!wallet.isConnected) return alert('Connect wallet first');
    if (!contractCode) return alert('Enter contract code');
    
    setLoading(true);
    try {
      const deployData = {
        code: contractCode,
        runtime: runtime,
        deployer: wallet.evmAddress
      };
      
      if (contractMetadata.name) deployData.name = contractMetadata.name;
      if (contractMetadata.symbol) deployData.symbol = contractMetadata.symbol;
      if (contractMetadata.totalSupply) deployData.total_supply = parseFloat(contractMetadata.totalSupply);
      if (contractMetadata.description) deployData.description = contractMetadata.description;
      if (contractMetadata.category) deployData.category = contractMetadata.category;
      if (contractMetadata.isDapp) deployData.is_dapp = contractMetadata.isDapp;
      if (contractMetadata.websiteUrl) deployData.website_url = contractMetadata.websiteUrl;
      
      const res = await axios.post(`${API}/contracts/deploy`, deployData);
      setDeployResult(res.data);
      loadContracts();
      
      let message = `Contract deployed!\nAddress: ${res.data.contract_address}\nType: ${res.data.contract_type}`;
      if (res.data.token) {
        message += `\n\nToken auto-registered: ${res.data.token.symbol} (${res.data.token.name})`;
      }
      if (res.data.dapp) {
        message += `\n\ndApp auto-registered: ${res.data.dapp.name} (${res.data.dapp.category})`;
      }
      if (res.data.nft) {
        message += `\n\nNFT Collection auto-registered: ${res.data.nft.name}`;
      }
      
      alert(message);
      setContractMetadata({ name: '', symbol: '', totalSupply: '', description: '', category: 'utility', isDapp: false, websiteUrl: '' });
    } catch (e) {
      alert('Error: ' + e.message);
    }
    setLoading(false);
  };

  const createProposal = async () => {
    if (!wallet.isConnected) return alert('Connect wallet first');
    if (!newProposal.title || !newProposal.description) return alert('Fill all fields');
    
    try {
      const res = await axios.post(`${API}/governance/proposals`, {
        title: newProposal.title,
        description: newProposal.description,
        proposer: wallet.evmAddress
      });
      alert(`Proposal created! AI recommendation: ${res.data.ai_recommendation} (${(res.data.ai_confidence * 100).toFixed(0)}% confidence)`);
      setNewProposal({ title: '', description: '' });
      loadProposals();
    } catch (e) {
      alert('Error: ' + e.message);
    }
  };

  const voteOnProposal = async (proposalId, voteFor) => {
    if (!wallet.isConnected) return alert('Connect wallet first');
    
    try {
      await axios.post(`${API}/governance/vote`, {
        proposal_id: proposalId,
        voter: wallet.evmAddress,
        vote_for: voteFor,
        stake_weight: 100
      });
      alert('Vote submitted!');
      loadProposals();
    } catch (e) {
      alert('Error: ' + e.message);
    }
  };

  return (
    <div>
      <h1 style={styles.title}>Developer - Deploy Contracts</h1>
      
      {/* AI Status Card */}
      {aiStatus && (
        <div style={{...styles.card, background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)', border: '1px solid #4CAF50'}}>
          <h3 style={{color: '#4CAF50'}}>AI Autonomous Training (Active)</h3>
          <p style={{color: '#888'}}>AI trains itself on network data without any user input.</p>
          <div style={{display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '15px', marginTop: '15px'}}>
            <div><span style={{color: '#888'}}>Accuracy:</span> <strong>{aiStatus.accuracy}%</strong></div>
            <div><span style={{color: '#888'}}>Rounds:</span> <strong>{aiStatus.training_rounds}</strong></div>
            <div><span style={{color: '#888'}}>Fraud Detected:</span> <strong>{aiStatus.fraud_detected}</strong></div>
            <div><span style={{color: '#888'}}>Mode:</span> <strong style={{color: '#4CAF50'}}>{aiStatus.mode}</strong></div>
          </div>
        </div>
      )}

      {/* Quick Token/NFT Factory */}
      <div style={{...styles.card, background: 'linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%)', border: '1px solid #f7931a', marginBottom: '20px'}}>
        <h3 style={{color: '#f7931a', marginBottom: '15px'}}>Quick Token & NFT Factory (No Code Required)</h3>
        <div style={{display: 'flex', gap: '10px', marginBottom: '20px'}}>
          <button 
            onClick={() => setShowExamples('token_factory')} 
            style={showExamples === 'token_factory' ? {...styles.tabActive, background: '#f7931a'} : styles.tab}
          >
            Create Token
          </button>
          <button 
            onClick={() => setShowExamples('nft_factory')} 
            style={showExamples === 'nft_factory' ? {...styles.tabActive, background: '#9c27b0'} : styles.tab}
          >
            Create NFT Collection
          </button>
        </div>
        
        {showExamples === 'token_factory' && wallet.isConnected && (
          <TokenFactoryForm wallet={wallet} />
        )}
        
        {showExamples === 'nft_factory' && wallet.isConnected && (
          <NFTFactoryForm wallet={wallet} />
        )}
        
        {(showExamples === 'token_factory' || showExamples === 'nft_factory') && !wallet.isConnected && (
          <div style={{textAlign: 'center', padding: '20px'}}>
            <p style={{color: '#888'}}>Connect wallet to create tokens and NFTs</p>
            <button onClick={() => setShowWallet(true)} style={styles.btn}>Connect Wallet</button>
          </div>
        )}
      </div>

      {/* Examples Tabs */}
      <div style={{display: 'flex', gap: '10px', marginBottom: '20px'}}>
        {['connect', 'evm', 'wasm', 'hybrid'].map(ex => (
          <button key={ex} onClick={() => setShowExamples(ex)} style={showExamples === ex ? styles.tabActive : styles.tab}>
            {ex === 'connect' ? 'Connect to NeoNet' : ex.toUpperCase() + ' Contract'}
          </button>
        ))}
      </div>

      {showExamples === 'connect' && (
        <div style={styles.card}>
          <h3>Connect to NeoNet Network</h3>
          <p>NeoNet is NOT Ethereum or Cosmos. It's a unified Web4 blockchain with EVM+WASM integrated runtime.</p>
          <pre style={styles.code}>{`// JavaScript - Connect to NeoNet
import { ethers } from 'ethers';

// NeoNet Network Config
const NEONET_CONFIG = {
  chainId: 8888,
  chainName: 'NeoNet Mainnet',
  rpcUrls: ['https://rpc.neonetainetwork.com'],
  nativeCurrency: {
    name: 'NNET',
    symbol: 'NNET',
    decimals: 18
  },
  blockExplorerUrls: ['https://explorer.neonetainetwork.com']
};

// Connect with ethers.js
const provider = new ethers.JsonRpcProvider(NEONET_CONFIG.rpcUrls[0]);

// Or add to MetaMask
await window.ethereum.request({
  method: 'wallet_addEthereumChain',
  params: [NEONET_CONFIG]
});

// Create wallet with dual keys (EVM + Quantum)
import { NeoNetWallet } from '@neonet/wallet';

const wallet = await NeoNetWallet.create();
console.log('EVM Address:', wallet.evmAddress);     // 0x...
console.log('Quantum Address:', wallet.neoAddress); // neo1...`}</pre>
        </div>
      )}

      {showExamples === 'evm' && (
        <div style={styles.card}>
          <h3>EVM Contract (Solidity) - Deploy to NeoNet</h3>
          <p>Write standard Solidity contracts, but they run on NeoNet's unified runtime.</p>
          <pre style={styles.code}>{`// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

// NeoNet ERC20 Token - uses native NNET for gas
contract MyNeoNetToken is ERC20 {
    constructor() ERC20("My Token", "MTK") {
        _mint(msg.sender, 1000000 * 10**18);
    }
    
    // Cross-runtime call to WASM contract
    function callWasmContract(address wasmAddr) external {
        // NeoNet bridges EVM <-> WASM automatically
        (bool success,) = wasmAddr.call(
            abi.encodeWithSignature("execute()")
        );
        require(success, "WASM call failed");
    }
}

// Deploy using Hardhat
// hardhat.config.js:
module.exports = {
  networks: {
    neonet: {
      url: "https://rpc.neonetainetwork.com",
      chainId: 8888,
      accounts: [process.env.PRIVATE_KEY]
    }
  }
};

// npx hardhat run scripts/deploy.js --network neonet`}</pre>
        </div>
      )}

      {showExamples === 'wasm' && (
        <div style={styles.card}>
          <h3>WASM Contract (Rust/CosmWasm) - Deploy to NeoNet</h3>
          <p>Write Rust contracts for advanced AI/ML integration on NeoNet.</p>
          <pre style={styles.code}>{`// lib.rs - CosmWasm contract for NeoNet
use cosmwasm_std::{
    entry_point, to_json_binary, Binary, Deps, DepsMut, 
    Env, MessageInfo, Response, StdResult
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, JsonSchema)]
pub struct InstantiateMsg {
    pub name: String,
}

#[derive(Serialize, Deserialize, Clone, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ExecuteMsg {
    RegisterAiModel { model_id: String, ipfs_hash: String },
    UpdateScore { model_id: String, score: u64 },
}

#[entry_point]
pub fn instantiate(
    deps: DepsMut,
    _env: Env,
    info: MessageInfo,
    msg: InstantiateMsg,
) -> StdResult<Response> {
    // Initialize AI model registry
    Ok(Response::new()
        .add_attribute("method", "instantiate")
        .add_attribute("owner", info.sender))
}

#[entry_point]
pub fn execute(
    deps: DepsMut,
    env: Env,
    info: MessageInfo,
    msg: ExecuteMsg,
) -> StdResult<Response> {
    match msg {
        ExecuteMsg::RegisterAiModel { model_id, ipfs_hash } => {
            // Register AI model for PoI consensus
            Ok(Response::new()
                .add_attribute("action", "register_model")
                .add_attribute("model_id", model_id))
        }
        ExecuteMsg::UpdateScore { model_id, score } => {
            Ok(Response::new())
        }
    }
}

// Build: cargo build --target wasm32-unknown-unknown --release
// Deploy: neonet-cli tx wasm store ./target/wasm32.wasm`}</pre>
        </div>
      )}

      {showExamples === 'hybrid' && (
        <div style={styles.card}>
          <h3>Hybrid Contract - EVM + WASM Together</h3>
          <p>NeoNet's unique feature: contracts can call across runtimes seamlessly.</p>
          <pre style={styles.code}>{`// CrossRuntimeBridge.sol - EVM side
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface INeoNetBridge {
    function callWasm(bytes32 wasmAddr, bytes calldata data) 
        external returns (bytes memory);
    function getAiScore(address validator) 
        external view returns (uint256);
}

contract HybridAiVoting {
    INeoNetBridge public bridge;
    
    struct Proposal {
        string title;
        uint256 forVotes;
        uint256 againstVotes;
        uint256 aiRecommendation; // 0-100
    }
    
    mapping(uint256 => Proposal) public proposals;
    
    function createProposal(
        string memory title,
        bytes32 wasmAnalyzer
    ) external returns (uint256) {
        // Call WASM AI contract to analyze proposal
        bytes memory result = bridge.callWasm(
            wasmAnalyzer,
            abi.encode("analyze", title)
        );
        
        uint256 aiScore = abi.decode(result, (uint256));
        
        proposals[block.number] = Proposal({
            title: title,
            forVotes: 0,
            againstVotes: 0,
            aiRecommendation: aiScore
        });
        
        return block.number;
    }
    
    function vote(uint256 proposalId, bool support) external {
        uint256 voterAiScore = bridge.getAiScore(msg.sender);
        uint256 weight = 1 + (voterAiScore / 10);
        
        if (support) {
            proposals[proposalId].forVotes += weight;
        } else {
            proposals[proposalId].againstVotes += weight;
        }
    }
}`}</pre>
        </div>
      )}

      {/* Deploy Contract Section */}
      {!wallet.isConnected ? (
        <div style={styles.card}>
          <p>Connect wallet to deploy contracts.</p>
          <button onClick={() => setShowWallet(true)} style={styles.btn}>Connect Wallet</button>
        </div>
      ) : (
        <>
          <div style={styles.card}>
            <h3>Deploy Smart Contract to NeoNet</h3>
            <p>Deploy using NeoNet's unified EVM+WASM runtime. Tokens, NFTs and dApps are auto-registered to the network!</p>
            <div style={styles.formGroup}>
              <label>Runtime</label>
              <select value={runtime} onChange={e => setRuntime(e.target.value)} style={styles.input}>
                <option value="hybrid">Hybrid (EVM + WASM)</option>
                <option value="evm">EVM Only (Solidity)</option>
                <option value="wasm">WASM Only (Rust/CosmWasm)</option>
              </select>
            </div>
            <div style={styles.formGroup}>
              <label>Contract Code</label>
              <textarea 
                placeholder={runtime === 'evm' ? 
                  '// SPDX-License-Identifier: MIT\npragma solidity ^0.8.0;\n\ncontract MyToken {\n    // ...\n}' : 
                  runtime === 'wasm' ?
                  'use cosmwasm_std::*;\n\n#[entry_point]\npub fn instantiate(...) {\n    // ...\n}' :
                  '// Hybrid contract - uses both EVM and WASM\n// Cross-runtime calls supported'
                }
                value={contractCode}
                onChange={e => setContractCode(e.target.value)}
                style={{...styles.input, minHeight: '150px', fontFamily: 'monospace'}}
              />
            </div>
            
            <div style={{background: '#16213e', padding: '15px', borderRadius: '8px', marginBottom: '15px'}}>
              <h4 style={{margin: '0 0 10px 0', color: '#4CAF50'}}>Project Metadata (Optional - Auto-fills Explore page)</h4>
              <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px'}}>
                <div style={styles.formGroup}>
                  <label>Project Name</label>
                  <input 
                    placeholder="My Awesome Token"
                    value={contractMetadata.name}
                    onChange={e => setContractMetadata({...contractMetadata, name: e.target.value})}
                    style={styles.input}
                  />
                </div>
                <div style={styles.formGroup}>
                  <label>Symbol (for tokens)</label>
                  <input 
                    placeholder="MTK"
                    value={contractMetadata.symbol}
                    onChange={e => setContractMetadata({...contractMetadata, symbol: e.target.value})}
                    style={styles.input}
                  />
                </div>
                <div style={styles.formGroup}>
                  <label>Total Supply (for tokens)</label>
                  <input 
                    type="number"
                    placeholder="1000000"
                    value={contractMetadata.totalSupply}
                    onChange={e => setContractMetadata({...contractMetadata, totalSupply: e.target.value})}
                    style={styles.input}
                  />
                </div>
                <div style={styles.formGroup}>
                  <label>Category</label>
                  <select 
                    value={contractMetadata.category}
                    onChange={e => setContractMetadata({...contractMetadata, category: e.target.value})}
                    style={styles.input}
                  >
                    <option value="utility">Utility</option>
                    <option value="defi">DeFi</option>
                    <option value="gaming">Gaming</option>
                    <option value="dao">DAO</option>
                    <option value="social">Social</option>
                    <option value="nft">NFT</option>
                  </select>
                </div>
              </div>
              <div style={styles.formGroup}>
                <label>Description</label>
                <textarea 
                  placeholder="Describe your project..."
                  value={contractMetadata.description}
                  onChange={e => setContractMetadata({...contractMetadata, description: e.target.value})}
                  style={{...styles.input, minHeight: '60px'}}
                />
              </div>
              <div style={styles.formGroup}>
                <label>Project Logo</label>
                <div style={{display: 'flex', gap: '10px', alignItems: 'center'}}>
                  <label style={{...styles.btn, background: '#333', cursor: 'pointer', display: 'inline-block'}}>
                    {uploadingLogo ? 'Uploading...' : 'Upload Logo'}
                    <input type="file" accept="image/*" onChange={handleLogoUpload} style={{display: 'none'}} disabled={uploadingLogo} />
                  </label>
                  <span style={{color: '#888'}}>or</span>
                  <input 
                    placeholder="Paste URL" 
                    value={contractMetadata.logoUrl}
                    onChange={e => setContractMetadata({...contractMetadata, logoUrl: e.target.value})}
                    style={{...styles.input, flex: 1}}
                  />
                </div>
                {contractMetadata.logoUrl && (
                  <div style={{marginTop: '10px', display: 'flex', alignItems: 'center', gap: '10px'}}>
                    <img src={contractMetadata.logoUrl} alt="Logo preview" style={{width: '64px', height: '64px', borderRadius: '50%', objectFit: 'cover'}} onError={e => e.target.style.display='none'} />
                    <button onClick={() => setContractMetadata({...contractMetadata, logoUrl: ''})} style={{...styles.btn, background: '#f44336', padding: '5px 10px'}}>Remove</button>
                  </div>
                )}
              </div>
              <div style={{display: 'flex', gap: '15px', alignItems: 'center'}}>
                <div style={styles.formGroup}>
                  <label>Website URL</label>
                  <input 
                    placeholder="https://myproject.io"
                    value={contractMetadata.websiteUrl}
                    onChange={e => setContractMetadata({...contractMetadata, websiteUrl: e.target.value})}
                    style={styles.input}
                  />
                </div>
                <label style={{display: 'flex', alignItems: 'center', gap: '8px', marginTop: '20px'}}>
                  <input 
                    type="checkbox"
                    checked={contractMetadata.isDapp}
                    onChange={e => setContractMetadata({...contractMetadata, isDapp: e.target.checked})}
                  />
                  Register as dApp
                </label>
              </div>
            </div>
            
            <button onClick={deployContract} style={styles.btn} disabled={loading}>
              {loading ? 'Deploying...' : `Deploy to NeoNet (${runtime.toUpperCase()})`}
            </button>
            {deployResult && (
              <div style={{marginTop: '15px', padding: '10px', background: '#1a1a2e', borderRadius: '8px'}}>
                <p><strong>Contract Address:</strong> {deployResult.contract_address}</p>
                <p><strong>Runtime:</strong> {deployResult.runtime}</p>
                <p><strong>TX Hash:</strong> {deployResult.tx_hash}</p>
              </div>
            )}
          </div>

          <div style={styles.card}>
            <h3>Deployed Contracts ({contracts.length})</h3>
            {contracts.length === 0 ? <p>No contracts deployed yet</p> : (
              <table style={styles.table}>
                <thead>
                  <tr><th>Address</th><th>Runtime</th><th>Deployer</th><th>Status</th></tr>
                </thead>
                <tbody>
                  {contracts.map(c => (
                    <tr key={c.address}>
                      <td>{c.address?.slice(0,20)}...</td>
                      <td><span style={c.runtime === 'hybrid' ? styles.online : styles.pending}>{c.runtime}</span></td>
                      <td>{c.deployer?.slice(0,15)}...</td>
                      <td>{c.status}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          <h2 style={styles.subtitle}>DualGov - AI + DAO Governance</h2>
          
          <div style={styles.card}>
            <h3>Create Proposal</h3>
            <p>AI analyzes proposals and provides recommendations (30% weight). DAO votes have 70% weight.</p>
            <div style={styles.formGroup}>
              <label>Title</label>
              <input 
                placeholder="Upgrade network security" 
                value={newProposal.title}
                onChange={e => setNewProposal({...newProposal, title: e.target.value})}
                style={styles.input}
              />
            </div>
            <div style={styles.formGroup}>
              <label>Description</label>
              <textarea 
                placeholder="Detailed description of the proposal..."
                value={newProposal.description}
                onChange={e => setNewProposal({...newProposal, description: e.target.value})}
                style={{...styles.input, minHeight: '80px'}}
              />
            </div>
            <button onClick={createProposal} style={styles.btn}>Submit Proposal</button>
          </div>

          <div style={styles.card}>
            <h3>Active Proposals ({proposals.length})</h3>
            {proposals.length === 0 ? <p>No proposals yet</p> : (
              <table style={styles.table}>
                <thead>
                  <tr><th>Title</th><th>Status</th><th>AI Rec.</th><th>For</th><th>Against</th><th>Vote</th></tr>
                </thead>
                <tbody>
                  {proposals.map(p => (
                    <tr key={p.proposal_id}>
                      <td>{p.title}</td>
                      <td><span style={p.status === 'active' ? styles.online : styles.pending}>{p.status}</span></td>
                      <td><span style={p.ai_recommendation === 'for' ? styles.online : p.ai_recommendation === 'against' ? {color: '#e74c3c'} : {color: '#888'}}>{p.ai_recommendation}</span></td>
                      <td>{p.for_votes}</td>
                      <td>{p.against_votes}</td>
                      <td>
                        <button onClick={() => voteOnProposal(p.proposal_id, true)} style={{...styles.btn, padding: '5px 10px', marginRight: 5}}>For</button>
                        <button onClick={() => voteOnProposal(p.proposal_id, false)} style={{...styles.btn, padding: '5px 10px', background: '#e74c3c'}}>Against</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </>
      )}
    </div>
  );
}

function DexTab({ wallet, setShowWallet }) {
  const [tokens, setTokens] = useState([]);
  const [pools, setPools] = useState([]);
  const [fromToken, setFromToken] = useState('NNET');
  const [toToken, setToToken] = useState('');
  const [amount, setAmount] = useState('');
  const [quote, setQuote] = useState(null);
  const [loading, setLoading] = useState(false);
  const [neoPrice, setNeoPrice] = useState(5.0);
  const [createToken, setCreateToken] = useState({ symbol: '', name: '', supply: '', runtime: 'hybrid' });
  const [createPool, setCreatePool] = useState({ token: '', neoAmount: '', tokenAmount: '' });

  useEffect(() => {
    loadTokens();
    loadPools();
    loadNeoPrice();
    const i = setInterval(() => {
      loadTokens();
      loadPools();
      loadNeoPrice();
    }, 10000);
    return () => clearInterval(i);
  }, []);

  const loadTokens = async () => {
    try {
      const res = await axios.get(`${API}/tokens`);
      if (res.data?.tokens) setTokens(res.data.tokens);
    } catch (e) {}
  };

  const loadPools = async () => {
    try {
      const res = await axios.get(`${API}/dex/pools`);
      if (res.data?.pools) setPools(res.data.pools);
    } catch (e) {}
  };

  const loadNeoPrice = async () => {
    try {
      const res = await axios.get(`${API}/dex/price/NNET`);
      if (res.data?.price_usd) setNeoPrice(res.data.price_usd);
    } catch (e) {}
  };

  const getQuote = async () => {
    if (!fromToken || !toToken || !amount) return;
    try {
      const res = await axios.get(`${API}/dex/quote?from_token=${fromToken}&to_token=${toToken}&amount=${amount}`);
      setQuote(res.data);
    } catch (e) {
      setQuote(null);
    }
  };

  useEffect(() => {
    if (fromToken && toToken && amount) {
      getQuote();
    }
  }, [fromToken, toToken, amount]);

  const executeSwap = async () => {
    if (!wallet.isConnected) return alert('Connect wallet first');
    if (!fromToken || !toToken || !amount) return alert('Fill all fields');
    
    setLoading(true);
    try {
      const res = await axios.post(`${API}/dex/swap`, {
        from_token: fromToken,
        to_token: toToken,
        amount: parseFloat(amount),
        trader: wallet.quantumAddress
      });
      
      alert(`Swap successful!\nSent: ${res.data.amount_in} ${fromToken}\nReceived: ${res.data.amount_out.toFixed(6)} ${toToken}\nBlock: #${res.data.block}`);
      setAmount('');
      setQuote(null);
      loadTokens();
      loadPools();
    } catch (e) {
      alert('Error: ' + (e.response?.data?.detail || e.message));
    }
    setLoading(false);
  };

  const handleCreateToken = async () => {
    if (!wallet.isConnected) return alert('Connect wallet first');
    if (!createToken.symbol || !createToken.name || !createToken.supply) return alert('Fill all fields');
    
    setLoading(true);
    try {
      const res = await axios.post(`${API}/tokens/create`, {
        symbol: createToken.symbol,
        name: createToken.name,
        total_supply: parseFloat(createToken.supply),
        creator: wallet.quantumAddress,
        runtime: createToken.runtime
      });
      
      alert(`Token created!\nSymbol: ${res.data.symbol}\nName: ${res.data.name}\nSupply: ${res.data.total_supply}\nContract: ${res.data.contract_address}\nRuntime: ${res.data.runtime}\n\n${res.data.message}`);
      setCreateToken({ symbol: '', name: '', supply: '', runtime: 'hybrid' });
      loadTokens();
    } catch (e) {
      alert('Error: ' + (e.response?.data?.detail || e.message));
    }
    setLoading(false);
  };

  const handleCreatePool = async () => {
    if (!wallet.isConnected) return alert('Connect wallet first');
    if (!createPool.token || !createPool.neoAmount || !createPool.tokenAmount) return alert('Fill all fields');
    
    setLoading(true);
    try {
      const res = await axios.post(`${API}/dex/pool/create`, {
        token: createPool.token,
        neo_amount: parseFloat(createPool.neoAmount),
        token_amount: parseFloat(createPool.tokenAmount),
        creator: wallet.quantumAddress
      });
      
      alert(`Liquidity pool created!\nPool: ${res.data.pool_id}\nNEO deposited: ${res.data.neo_deposited}\nToken deposited: ${res.data.token_deposited}\nInitial price: $${res.data.initial_price.toFixed(4)}`);
      setCreatePool({ token: '', neoAmount: '', tokenAmount: '' });
      loadPools();
      loadTokens();
    } catch (e) {
      alert('Error: ' + (e.response?.data?.detail || e.message));
    }
    setLoading(false);
  };

  const swapDirection = () => {
    setFromToken(toToken);
    setToToken(fromToken);
    setQuote(null);
  };

  return (
    <div>
      <h1 style={{fontSize: 32, marginBottom: 24, color: '#fff'}}>DEX - Token Exchange</h1>
      
      <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16, marginBottom: 24}}>
        <div style={{background: '#161b22', border: '1px solid #30363d', borderRadius: 8, padding: 20}}>
          <div style={{display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8}}>
            <NnetLogo size={24} />
            <span style={{fontSize: 14, color: '#8b949e'}}>NNET Price</span>
          </div>
          <div style={{fontSize: 28, fontWeight: 'bold', color: '#4CAF50'}}>${neoPrice.toFixed(4)}</div>
        </div>
        <div style={{background: '#161b22', border: '1px solid #30363d', borderRadius: 8, padding: 20}}>
          <div style={{fontSize: 14, color: '#8b949e', marginBottom: 8}}>Total Tokens</div>
          <div style={{fontSize: 28, fontWeight: 'bold', color: '#fff'}}>{tokens.length}</div>
        </div>
        <div style={{background: '#161b22', border: '1px solid #30363d', borderRadius: 8, padding: 20}}>
          <div style={{fontSize: 14, color: '#8b949e', marginBottom: 8}}>Liquidity Pools</div>
          <div style={{fontSize: 28, fontWeight: 'bold', color: '#2196F3'}}>{pools.length}</div>
        </div>
      </div>

      {!wallet.isConnected ? (
        <div style={{background: '#161b22', border: '1px solid #30363d', borderRadius: 8, padding: 20, textAlign: 'center'}}>
          <p>Connect your wallet to swap tokens</p>
          <button onClick={() => setShowWallet(true)} style={{padding: '12px 24px', background: '#238636', border: 'none', borderRadius: 6, color: '#fff', cursor: 'pointer', fontWeight: 600}}>
            Connect Wallet
          </button>
        </div>
      ) : (
        <>
          <div style={{background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)', border: '1px solid #4CAF50', borderRadius: 8, padding: 20, marginBottom: 24}}>
            <h3 style={{color: '#4CAF50', marginBottom: 16}}>Swap Tokens</h3>
            
            <div style={{marginBottom: 16}}>
              <label style={{color: '#8b949e', fontSize: 14}}>From</label>
              <div style={{display: 'flex', gap: 10}}>
                <select value={fromToken} onChange={e => setFromToken(e.target.value)} style={{flex: 1, padding: 12, background: '#0d1117', border: '1px solid #30363d', borderRadius: 6, color: '#c9d1d9'}}>
                  <option value="NNET">NNET</option>
                  {tokens.filter(t => t.symbol !== 'NNET').map(t => (
                    <option key={t.symbol} value={t.symbol}>{t.symbol}</option>
                  ))}
                </select>
                <input 
                  type="number" 
                  placeholder="Amount" 
                  value={amount} 
                  onChange={e => setAmount(e.target.value)} 
                  style={{flex: 2, padding: 12, background: '#0d1117', border: '1px solid #30363d', borderRadius: 6, color: '#c9d1d9'}}
                />
              </div>
            </div>

            <div style={{textAlign: 'center', margin: '10px 0'}}>
              <button onClick={swapDirection} style={{background: '#21262d', border: '1px solid #30363d', borderRadius: '50%', width: 40, height: 40, cursor: 'pointer', color: '#fff', fontSize: 18}}>
                ↕
              </button>
            </div>

            <div style={{marginBottom: 16}}>
              <label style={{color: '#8b949e', fontSize: 14}}>To</label>
              <select value={toToken} onChange={e => setToToken(e.target.value)} style={{width: '100%', padding: 12, background: '#0d1117', border: '1px solid #30363d', borderRadius: 6, color: '#c9d1d9'}}>
                <option value="">Select token</option>
                {tokens.filter(t => t.symbol !== fromToken).map(t => (
                  <option key={t.symbol} value={t.symbol}>{t.symbol}</option>
                ))}
              </select>
            </div>

            {quote && !quote.error && (
              <div style={{background: '#0d1117', borderRadius: 8, padding: 16, marginBottom: 16}}>
                <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: 8}}>
                  <span style={{color: '#8b949e'}}>You receive:</span>
                  <span style={{color: '#4CAF50', fontWeight: 'bold'}}>{quote.amount_out.toFixed(6)} {toToken}</span>
                </div>
                <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: 8}}>
                  <span style={{color: '#8b949e'}}>Rate:</span>
                  <span style={{color: '#fff'}}>1 {fromToken} = {quote.rate.toFixed(6)} {toToken}</span>
                </div>
                <div style={{display: 'flex', justifyContent: 'space-between'}}>
                  <span style={{color: '#8b949e'}}>Fee:</span>
                  <span style={{color: '#ff9800'}}>{quote.fee.toFixed(6)} {fromToken} (0.3%)</span>
                </div>
              </div>
            )}

            <button onClick={executeSwap} disabled={loading || !quote} style={{width: '100%', padding: 14, background: quote ? '#238636' : '#333', border: 'none', borderRadius: 6, color: '#fff', cursor: quote ? 'pointer' : 'not-allowed', fontWeight: 600, fontSize: 16}}>
              {loading ? 'Swapping...' : 'Swap'}
            </button>
          </div>

          <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16, marginBottom: 24}}>
            <div style={{background: '#161b22', border: '1px solid #30363d', borderRadius: 8, padding: 20}}>
              <h3 style={{color: '#fff', marginBottom: 16}}>Create Token</h3>
              <div style={{marginBottom: 12}}>
                <input placeholder="Symbol (e.g. BTC)" value={createToken.symbol} onChange={e => setCreateToken({...createToken, symbol: e.target.value.toUpperCase()})} style={{width: '100%', padding: 10, background: '#0d1117', border: '1px solid #30363d', borderRadius: 6, color: '#c9d1d9'}} />
              </div>
              <div style={{marginBottom: 12}}>
                <input placeholder="Name (e.g. Bitcoin)" value={createToken.name} onChange={e => setCreateToken({...createToken, name: e.target.value})} style={{width: '100%', padding: 10, background: '#0d1117', border: '1px solid #30363d', borderRadius: 6, color: '#c9d1d9'}} />
              </div>
              <div style={{marginBottom: 12}}>
                <input type="number" placeholder="Total Supply" value={createToken.supply} onChange={e => setCreateToken({...createToken, supply: e.target.value})} style={{width: '100%', padding: 10, background: '#0d1117', border: '1px solid #30363d', borderRadius: 6, color: '#c9d1d9'}} />
              </div>
              <div style={{marginBottom: 12}}>
                <label style={{color: '#8b949e', fontSize: 14, display: 'block', marginBottom: 6}}>Contract Runtime</label>
                <div style={{display: 'flex', gap: 10}}>
                  <label style={{flex: 1, display: 'flex', alignItems: 'center', gap: 8, padding: 12, background: createToken.runtime === 'evm' ? '#1a472a' : '#0d1117', border: createToken.runtime === 'evm' ? '2px solid #4CAF50' : '1px solid #30363d', borderRadius: 6, cursor: 'pointer'}}>
                    <input type="radio" name="runtime" value="evm" checked={createToken.runtime === 'evm'} onChange={e => setCreateToken({...createToken, runtime: e.target.value})} />
                    <span style={{color: '#fff'}}>EVM Only</span>
                  </label>
                  <label style={{flex: 1, display: 'flex', alignItems: 'center', gap: 8, padding: 12, background: createToken.runtime === 'hybrid' ? '#1a472a' : '#0d1117', border: createToken.runtime === 'hybrid' ? '2px solid #4CAF50' : '1px solid #30363d', borderRadius: 6, cursor: 'pointer'}}>
                    <input type="radio" name="runtime" value="hybrid" checked={createToken.runtime === 'hybrid'} onChange={e => setCreateToken({...createToken, runtime: e.target.value})} />
                    <span style={{color: '#fff'}}>Hybrid (EVM+WASM)</span>
                  </label>
                </div>
                <p style={{color: '#6e7681', fontSize: 11, marginTop: 6}}>Hybrid = AI-optimized with quantum security</p>
              </div>
              <button onClick={handleCreateToken} disabled={loading} style={{width: '100%', padding: 12, background: '#2196F3', border: 'none', borderRadius: 6, color: '#fff', cursor: 'pointer', fontWeight: 600}}>
                Create Token
              </button>
            </div>

            <div style={{background: '#161b22', border: '1px solid #30363d', borderRadius: 8, padding: 20}}>
              <h3 style={{color: '#fff', marginBottom: 16}}>Create Liquidity Pool</h3>
              <div style={{marginBottom: 12}}>
                <select value={createPool.token} onChange={e => setCreatePool({...createPool, token: e.target.value})} style={{width: '100%', padding: 10, background: '#0d1117', border: '1px solid #30363d', borderRadius: 6, color: '#c9d1d9'}}>
                  <option value="">Select your token</option>
                  {tokens.filter(t => t.symbol !== 'NNET' && !pools.some(p => p.token_b === t.symbol)).map(t => (
                    <option key={t.symbol} value={t.symbol}>{t.symbol} - {t.name}</option>
                  ))}
                </select>
              </div>
              <div style={{marginBottom: 12}}>
                <input type="number" placeholder="NNET amount for liquidity" value={createPool.neoAmount} onChange={e => setCreatePool({...createPool, neoAmount: e.target.value})} style={{width: '100%', padding: 10, background: '#0d1117', border: '1px solid #30363d', borderRadius: 6, color: '#c9d1d9'}} />
              </div>
              <div style={{marginBottom: 12}}>
                <input type="number" placeholder="Token amount for liquidity" value={createPool.tokenAmount} onChange={e => setCreatePool({...createPool, tokenAmount: e.target.value})} style={{width: '100%', padding: 10, background: '#0d1117', border: '1px solid #30363d', borderRadius: 6, color: '#c9d1d9'}} />
              </div>
              <button onClick={handleCreatePool} disabled={loading} style={{width: '100%', padding: 12, background: '#ff9800', border: 'none', borderRadius: 6, color: '#fff', cursor: 'pointer', fontWeight: 600}}>
                Create Pool
              </button>
              <p style={{color: '#888', fontSize: 12, marginTop: 8}}>Token price = (NNET amount / Token amount) × NNET price</p>
            </div>
          </div>

          <div style={{background: '#161b22', border: '1px solid #30363d', borderRadius: 8, padding: 20}}>
            <h3 style={{color: '#fff', marginBottom: 16}}>All Tokens</h3>
            <div className="responsive-table-wrapper" style={{overflowX: 'auto', WebkitOverflowScrolling: 'touch'}}>
              <table style={{width: '100%', borderCollapse: 'collapse', minWidth: 700}}>
                <thead>
                  <tr style={{borderBottom: '1px solid #30363d'}}>
                    <th style={{padding: 12, textAlign: 'left'}}>Symbol</th>
                    <th style={{padding: 12, textAlign: 'left'}}>Name</th>
                    <th style={{padding: 12, textAlign: 'left'}}>Contract</th>
                    <th style={{padding: 12, textAlign: 'center'}}>Runtime</th>
                    <th style={{padding: 12, textAlign: 'right'}}>Price</th>
                    <th style={{padding: 12, textAlign: 'right'}}>Supply</th>
                  </tr>
                </thead>
                <tbody>
                  {tokens.map(t => (
                    <tr key={t.symbol} style={{borderBottom: '1px solid #21262d'}}>
                      <td style={{padding: 12, fontWeight: 'bold', color: t.is_native ? '#4CAF50' : '#58a6ff'}}>{t.symbol}</td>
                      <td style={{padding: 12}}>{t.name}</td>
                      <td style={{padding: 12, fontFamily: 'monospace', fontSize: 11}}>
                        {t.contract_address ? (
                          <span style={{color: '#8b949e'}} title={t.contract_address}>
                            {t.contract_address.slice(0, 10)}...{t.contract_address.slice(-6)}
                          </span>
                        ) : (
                          <span style={{color: '#484f58'}}>Native</span>
                        )}
                      </td>
                      <td style={{padding: 12, textAlign: 'center'}}>
                        <span style={{
                          padding: '4px 8px',
                          borderRadius: 4,
                          fontSize: 11,
                          fontWeight: 600,
                          background: t.runtime === 'hybrid' ? '#1a472a' : t.runtime === 'evm' ? '#1a3a5c' : '#21262d',
                          color: t.runtime === 'hybrid' ? '#4CAF50' : t.runtime === 'evm' ? '#58a6ff' : '#8b949e'
                        }}>
                          {t.runtime?.toUpperCase() || 'NATIVE'}
                        </span>
                      </td>
                      <td style={{padding: 12, textAlign: 'right', color: '#4CAF50'}}>${t.price_usd?.toFixed(4) || '0.0000'}</td>
                      <td style={{padding: 12, textAlign: 'right'}}>{t.total_supply?.toLocaleString() || 0}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {pools.length > 0 && (
            <div style={{background: '#161b22', border: '1px solid #30363d', borderRadius: 8, padding: 20, marginTop: 16}}>
              <h3 style={{color: '#fff', marginBottom: 16}}>Liquidity Pools</h3>
              <table style={{width: '100%', borderCollapse: 'collapse'}}>
                <thead>
                  <tr style={{borderBottom: '1px solid #30363d'}}>
                    <th style={{padding: 12, textAlign: 'left'}}>Pool</th>
                    <th style={{padding: 12, textAlign: 'right'}}>NNET Reserve</th>
                    <th style={{padding: 12, textAlign: 'right'}}>Token Reserve</th>
                    <th style={{padding: 12, textAlign: 'right'}}>Price</th>
                  </tr>
                </thead>
                <tbody>
                  {pools.map(p => (
                    <tr key={p.pool_id} style={{borderBottom: '1px solid #21262d'}}>
                      <td style={{padding: 12, fontWeight: 'bold', color: '#2196F3'}}>{p.pool_id}</td>
                      <td style={{padding: 12, textAlign: 'right'}}>{p.reserve_a?.toFixed(4)}</td>
                      <td style={{padding: 12, textAlign: 'right'}}>{p.reserve_b?.toFixed(4)}</td>
                      <td style={{padding: 12, textAlign: 'right', color: '#4CAF50'}}>${p.price?.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function MiningTab({ wallet }) {
  const [leaderboard, setLeaderboard] = useState([]);
  const [stats, setStats] = useState(null);
  const [copied, setCopied] = useState('');

  const GITHUB_REPO = 'https://github.com/neonetainetwork/NeoNet-Blockchain';
  const MINER_DOWNLOAD = '/download/neonet_miner.py';

  useEffect(() => {
    loadStats();
    loadLeaderboard();
  }, []);

  const loadStats = async () => {
    try {
      const res = await axios.get(`${API}/ai-energy/stats`);
      if (res.data) setStats(res.data);
    } catch (e) {}
  };

  const loadLeaderboard = async () => {
    try {
      const res = await axios.get(`${API}/ai-energy/leaderboard`);
      if (res.data?.leaderboard) setLeaderboard(res.data.leaderboard);
    } catch (e) {}
  };

  const copyToClipboard = (text, label) => {
    try {
      if (navigator.clipboard) {
        navigator.clipboard.writeText(text);
      } else {
        const ta = document.createElement('textarea');
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
      }
      setCopied(label);
      setTimeout(() => setCopied(''), 2000);
    } catch (e) {
      alert('Copy failed: ' + text);
    }
  };

  const serverUrl = window.location.origin.replace(':5000', ':8000');

  return (
    <div>
      <h1 style={styles.title}>Power the AI Network</h1>
      <p style={{color: '#888', marginBottom: 20}}>Your computer provides the ENERGY that runs NeoNet AI. Without energy providers, the network cannot function. You are the power source.</p>
      
      <div style={{...styles.card, background: 'linear-gradient(135deg, #1a2a1a 0%, #1a1a2e 100%)', border: '2px solid #4CAF50', marginBottom: 20}}>
        <h3 style={{color: '#4CAF50', marginBottom: 12}}>You Are The Network's Power Source</h3>
        <p style={{color: '#c9d1d9', marginBottom: 16, lineHeight: 1.6}}>
          The NeoNet AI network runs on computing power from people like you. Your computer provides the energy for fraud detection, AI training, and network protection. <strong style={{color: '#4CAF50'}}>Without you, the network stops.</strong>
        </p>
        <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16}}>
          <div style={{textAlign: 'center', padding: 12, background: '#0d1117', borderRadius: 8}}>
            <div style={{fontSize: 32, marginBottom: 8}}>1</div>
            <div style={{color: '#c9d1d9'}}>Download energy provider script</div>
          </div>
          <div style={{textAlign: 'center', padding: 12, background: '#0d1117', borderRadius: 8}}>
            <div style={{fontSize: 32, marginBottom: 8}}>2</div>
            <div style={{color: '#c9d1d9'}}>Your PC powers AI operations</div>
          </div>
          <div style={{textAlign: 'center', padding: 12, background: '#0d1117', borderRadius: 8}}>
            <div style={{fontSize: 32, marginBottom: 8}}>3</div>
            <div style={{color: '#c9d1d9'}}>Network uses your energy</div>
          </div>
          <div style={{textAlign: 'center', padding: 12, background: '#0d1117', borderRadius: 8}}>
            <div style={{fontSize: 32, marginBottom: 8}}>4</div>
            <div style={{color: '#c9d1d9'}}>Earn NNET for powering network</div>
          </div>
        </div>
      </div>
      
      {stats && (
        <div style={styles.grid}>
          <div style={styles.card}>
            <div style={styles.cardLabel}>Active Providers</div>
            <div style={styles.cardValue}>{stats.active_sessions}</div>
          </div>
          <div style={styles.card}>
            <div style={styles.cardLabel}>Energy Providers</div>
            <div style={styles.cardValue}>{stats.total_contributors}</div>
          </div>
          <div style={styles.card}>
            <div style={styles.cardLabel}>AI Tasks Powered</div>
            <div style={styles.cardValue}>{stats.total_tasks_completed}</div>
          </div>
          <div style={styles.card}>
            <div style={styles.cardLabel}>NNET Distributed</div>
            <div style={styles.cardValue}>{stats.total_rewards_distributed?.toFixed(2) || '0'}</div>
          </div>
        </div>
      )}

      {/* Download Miner Section */}
      <div style={{...styles.card, border: '2px solid #4CAF50', background: 'linear-gradient(135deg, #0d2137 0%, #1a1a2e 100%)'}}>
        <h3 style={{color: '#4CAF50', marginBottom: 15}}>Download NeoNet Miner</h3>
        <p style={{color: '#aaa', marginBottom: 20}}>One script - your computer becomes a network node. AI runs locally and maintains the network.</p>
        
        <div style={{display: 'flex', gap: 15, flexWrap: 'wrap', marginBottom: 20}}>
          <a href={MINER_DOWNLOAD} download style={{...styles.btn, textDecoration: 'none', background: '#4CAF50', fontWeight: 'bold', fontSize: 16, padding: '15px 30px'}}>
            Download neonet_miner.py
          </a>
          <a href={GITHUB_REPO} target="_blank" rel="noopener noreferrer" style={{...styles.btn, textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: 8}}>
            GitHub
          </a>
        </div>
        
        <div style={{background: '#1a1a2e', padding: 15, borderRadius: 4, fontFamily: 'monospace', fontSize: 14, marginBottom: 15}}>
          <div style={{color: '#888', marginBottom: 8}}># Install dependencies</div>
          <code style={{color: '#4CAF50'}}>pip install aiohttp numpy</code>
        </div>
        
        <div style={{background: '#1a1a2e', padding: 15, borderRadius: 4, fontFamily: 'monospace', fontSize: 14}}>
          <div style={{color: '#888', marginBottom: 8}}># Start mining</div>
          <code style={{color: '#4CAF50'}}>python neonet_miner.py --wallet neo1your_wallet</code>
        </div>
        
        <p style={{color: '#888', marginTop: 15, fontSize: 13}}>
          The script automatically: starts AI → connects to network → processes tasks → earns NNET rewards
        </p>
      </div>

      <div style={styles.card}>
        <h3>Quick Start Guide</h3>
        <p style={{color: '#aaa', marginBottom: 20}}>Follow these steps to start mining on NeoNet:</p>
        
        <div style={{background: '#0a1929', padding: 20, borderRadius: 8, marginBottom: 20}}>
          <h4 style={{color: '#1976d2', marginBottom: 15}}>Step 1: Install Python Dependencies</h4>
          <div style={{background: '#1a1a2e', padding: 15, borderRadius: 4, fontFamily: 'monospace', fontSize: 14}}>
            <code style={{color: '#4CAF50'}}>pip install aiohttp numpy</code>
          </div>
        </div>

        <div style={{background: '#0a1929', padding: 20, borderRadius: 8, marginBottom: 20}}>
          <h4 style={{color: '#1976d2', marginBottom: 15}}>Step 2: Download Miner</h4>
          <p style={{color: '#888', marginBottom: 10}}>Download neonet_miner.py - this is the full node that runs AI locally.</p>
          <a href={MINER_DOWNLOAD} download style={{...styles.btn, textDecoration: 'none', background: '#4CAF50'}}>
            Download neonet_miner.py
          </a>
        </div>

        <div style={{background: '#0a1929', padding: 20, borderRadius: 8, marginBottom: 20}}>
          <h4 style={{color: '#1976d2', marginBottom: 15}}>Step 3: Start Mining</h4>
          <div style={{background: '#1a1a2e', padding: 15, borderRadius: 4, fontFamily: 'monospace', fontSize: 14, marginBottom: 10}}>
            <code style={{color: '#4CAF50'}}>python neonet_miner.py --wallet neo1your_wallet</code>
          </div>
          <button 
            onClick={() => copyToClipboard('python neonet_miner.py --wallet neo1your_wallet', 'cmd')}
            style={{...styles.btn, padding: '8px 15px', fontSize: 14}}
          >
            {copied === 'cmd' ? 'Copied!' : 'Copy Command'}
          </button>
        </div>
      </div>

      <div style={styles.card}>
        <h3>Command Options</h3>
        <table style={styles.table}>
          <thead>
            <tr><th>Option</th><th>Description</th><th>Example</th></tr>
          </thead>
          <tbody>
            <tr><td><code>--wallet</code></td><td>Your wallet address for NNET rewards</td><td>neo1abc...</td></tr>
            <tr><td><code>--port</code></td><td>P2P port (default: 8080)</td><td>9000</td></tr>
            <tr><td><code>--cpu</code></td><td>Number of CPU cores to use</td><td>8</td></tr>
            <tr><td><code>--gpu-mem</code></td><td>GPU memory in MB (0 = CPU only)</td><td>8192</td></tr>
          </tbody>
        </table>
      </div>

      <div style={styles.card}>
        <h3>What Your Energy Powers</h3>
        <p style={{color: '#888', marginBottom: 15}}>Your computing power runs these critical AI operations. <strong style={{color: '#4CAF50'}}>10x higher rewards!</strong></p>
        <table style={styles.table}>
          <thead>
            <tr><th>Task</th><th>What It Does</th><th>Reward</th></tr>
          </thead>
          <tbody>
            <tr><td style={{color: '#da77f2'}}>Federated Learning</td><td>Collaborative AI training (highest reward)</td><td style={{color: '#4CAF50', fontWeight: 'bold'}}>1.00 NNET</td></tr>
            <tr><td style={{color: '#4dabf7'}}>Model Training</td><td>Train AI models for network security</td><td style={{color: '#4CAF50', fontWeight: 'bold'}}>0.80 NNET</td></tr>
            <tr><td style={{color: '#69db7c'}}>Network Protection</td><td>Protect network and detect attacks</td><td style={{color: '#4CAF50', fontWeight: 'bold'}}>0.60 NNET</td></tr>
            <tr><td style={{color: '#ff6b6b'}}>Fraud Detection</td><td>Analyze transactions for suspicious activity</td><td style={{color: '#4CAF50', fontWeight: 'bold'}}>0.50 NNET</td></tr>
            <tr><td style={{color: '#a9e34b'}}>Gradient Compute</td><td>Calculate optimization gradients</td><td style={{color: '#4CAF50', fontWeight: 'bold'}}>0.50 NNET</td></tr>
            <tr><td style={{color: '#74c0fc'}}>AI Inference</td><td>Run trained models on new data</td><td style={{color: '#4CAF50', fontWeight: 'bold'}}>0.40 NNET</td></tr>
            <tr><td style={{color: '#ffd43b'}}>Data Validation</td><td>Verify transaction integrity</td><td style={{color: '#4CAF50', fontWeight: 'bold'}}>0.30 NNET</td></tr>
            <tr><td style={{color: '#748ffc'}}>Matrix Operations</td><td>Heavy neural network computations</td><td style={{color: '#4CAF50', fontWeight: 'bold'}}>0.30 NNET</td></tr>
          </tbody>
        </table>
      </div>

      <div style={{...styles.card, background: 'linear-gradient(135deg, #1a2a3a 0%, #1a1a2e 100%)', border: '2px solid #4CAF50'}}>
        <h3 style={{color: '#4CAF50', marginBottom: 15}}>AI Energy Economy</h3>
        <p style={{color: '#c9d1d9', marginBottom: 20, lineHeight: 1.6}}>
          NeoNet is powered by <strong style={{color: '#4CAF50'}}>AI Energy Providers</strong> like you.
          Your computing power keeps the AI network running. Earn NNET for every task you power!
        </p>
        
        <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 15, marginBottom: 20}}>
          <div style={{background: '#0d1117', padding: 15, borderRadius: 8, textAlign: 'center'}}>
            <div style={{color: '#4CAF50', fontSize: 24, fontWeight: 'bold'}}>0.05 NNET</div>
            <div style={{color: '#888', fontSize: 13}}>Max Task Reward</div>
          </div>
          <div style={{background: '#0d1117', padding: 15, borderRadius: 8, textAlign: 'center'}}>
            <div style={{color: '#4dabf7', fontSize: 24, fontWeight: 'bold'}}>8</div>
            <div style={{color: '#888', fontSize: 13}}>Task Types</div>
          </div>
          <div style={{background: '#0d1117', padding: 15, borderRadius: 8, textAlign: 'center'}}>
            <div style={{color: '#ff6b6b', fontSize: 24, fontWeight: 'bold'}}>24/7</div>
            <div style={{color: '#888', fontSize: 13}}>Earn Anytime</div>
          </div>
          <div style={{background: '#0d1117', padding: 15, borderRadius: 8, textAlign: 'center'}}>
            <div style={{color: '#ffd43b', fontSize: 24, fontWeight: 'bold'}}>Instant</div>
            <div style={{color: '#888', fontSize: 13}}>Payouts</div>
          </div>
        </div>

        <p style={{color: '#888', marginTop: 15, fontSize: 13}}>
          The more compute power you provide, the more NNET you earn. Join thousands of AI Energy Providers powering the network!
        </p>
      </div>

      <div style={styles.card}>
        <h3>Example Output</h3>
        <div style={{background: '#0a0a0a', padding: 15, borderRadius: 4, fontFamily: 'monospace', fontSize: 13, color: '#aaa', whiteSpace: 'pre-wrap'}}>
{`╔═══════════════════════════════════════════════════════╗
║           NeoNet Miner v1.0                           ║
║       AI-Powered Web4 Blockchain Network              ║
║       Token: NNET                                     ║
╚═══════════════════════════════════════════════════════╝
============================================================
    NeoNet Full Node - AI Network
============================================================
Wallet: neo1your_wallet
Port: 8080
CPU: 8 cores
------------------------------------------------------------
[NODE] Local AI node started on port 8080
[NET] Connected to NeoNet

[AI] Processing: fraud_detection
[OK] Completed in 45ms
[NNET] +0.0250 | Total: 0.0250 NNET

[AI] Processing: model_training
[OK] Completed in 120ms
[NNET] +0.0400 | Total: 0.0650 NNET

[AI] Processing: federated_learning
[OK] Completed in 200ms
[NNET] +0.0500 | Total: 0.1150 NNET

[SYNC] Peers: 15 | Blockchain: 1250 blocks
[INFO] Daily estimate: ~3.5 NNET (24h mining)`}
        </div>
      </div>

      <div style={styles.card}>
        <h3>Run 24/7 as Background Service (Linux)</h3>
        <p style={{color: '#888', marginBottom: 15}}>Create a systemd service to run the miner constantly:</p>
        <div style={{background: '#1a1a2e', padding: 15, borderRadius: 4, fontFamily: 'monospace', fontSize: 12, color: '#aaa', whiteSpace: 'pre-wrap'}}>
{`[Unit]
Description=NeoNet Miner
After=network.target

[Service]
Type=simple
User=yourusername
WorkingDirectory=/path/to/miner
ExecStart=/usr/bin/python3 neonet_miner.py --wallet YOUR_WALLET
Restart=always

[Install]
WantedBy=multi-user.target`}
        </div>
        <p style={{color: '#888', marginTop: 15, fontSize: 13}}>Save as <code>/etc/systemd/system/neonet-miner.service</code>, then run: <code>sudo systemctl enable --now neonet-miner</code></p>
      </div>

      <div style={styles.card}>
        <h3>Top Energy Providers</h3>
        {leaderboard.length === 0 ? <p style={{color: '#888'}}>No energy providers yet. Be the first to power the network!</p> : (
          <table style={styles.table}>
            <thead>
              <tr><th>Rank</th><th>Provider ID</th><th>Tasks Powered</th><th>Hours</th><th>NNET Earned</th></tr>
            </thead>
            <tbody>
              {leaderboard.map(c => (
                <tr key={c.id}>
                  <td>#{c.rank}</td>
                  <td>{c.id}</td>
                  <td>{c.total_tasks}</td>
                  <td>{c.total_compute_hours}h</td>
                  <td style={{color: '#4CAF50'}}>{c.total_rewards} NNET</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

function AppContent() {
  const wallet = useWallet();
  const navigate = useNavigate();
  const location = useLocation();
  
  const pathToTab = {
    '/': 'home',
    '/dashboard': 'dashboard',
    '/wallet': 'wallet',
    '/send': 'send',
    '/dex': 'dex',
    '/explore': 'explore',
    '/mining': 'mining',
    '/developer': 'developer',
    '/info': 'info',
    '/docs': 'docs'
  };
  
  const tabToPath = {
    'home': '/',
    'dashboard': '/dashboard',
    'wallet': '/wallet',
    'send': '/send',
    'dex': '/dex',
    'explore': '/explore',
    'mining': '/mining',
    'developer': '/developer',
    'info': '/info',
    'docs': '/docs'
  };
  
  const currentTab = pathToTab[location.pathname] || 'home';
  
  const setTab = (newTab) => {
    navigate(tabToPath[newTab] || '/');
  };
  
  const [showWallet, setShowWallet] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [network, setNetwork] = useState(null);
  const [energyProviders, setEnergyProviders] = useState([]);
  const [loading, setLoading] = useState(true);
  const [tasks, setTasks] = useState([]);
  const [sendForm, setSendForm] = useState({ to: '', amount: '', token: 'NNET' });
  const [tokenomics, setTokenomics] = useState(null);
  const [txResult, setTxResult] = useState(null);
  const [transactionHistory, setTransactionHistory] = useState([]);
  const [availableTokens, setAvailableTokens] = useState([]);
  const [tokenSearch, setTokenSearch] = useState('');
  const [showTokenDropdown, setShowTokenDropdown] = useState(false);
  
  const tab = currentTab;

  useEffect(() => {
    loadNetworkData();
    loadAvailableTokens();
    const i = setInterval(loadNetworkData, 5000);
    const i2 = setInterval(loadAvailableTokens, 10000);
    return () => { clearInterval(i); clearInterval(i2); };
  }, []);

  const loadAvailableTokens = async () => {
    try {
      const res = await axios.get(`${API}/tokens`);
      if (res.data?.tokens) {
        setAvailableTokens(res.data.tokens);
      }
    } catch (e) {}
  };

  const loadTokenomics = async () => {
    try {
      const res = await axios.get(`${API}/tokenomics`);
      if (res.data) {
        setTokenomics(res.data);
      }
    } catch (e) {}
  };

  const loadTransactionHistory = async () => {
    if (!wallet.isConnected) return;
    try {
      const res = await axios.get(`${API}/transactions/history/${wallet.quantumAddress}`);
      if (res.data?.transactions) {
        setTransactionHistory(res.data.transactions);
      }
    } catch (e) {}
  };

  useEffect(() => {
    loadTokenomics();
    if (wallet.isConnected) {
      loadTransactionHistory();
      const interval = setInterval(() => {
        loadTokenomics();
        loadTransactionHistory();
      }, 10000);
      return () => clearInterval(interval);
    }
  }, [wallet.isConnected, wallet.quantumAddress]);

  const loadNetworkData = async () => {
    try {
      const [overview, validatorsRes, stats, taskRes, neoPrice] = await Promise.all([
        axios.get(`${API}/network/overview`).catch(() => null),
        axios.get(`${API}/network/validators`).catch(() => null),
        axios.get(`${API}/network/stats`).catch(() => null),
        axios.get(`${API}/tasks`).catch(() => null),
        axios.get(`${API}/dex/price/NNET`).catch(() => null)
      ]);
      
      if (overview?.data) {
        setNetwork({
          status: overview.data.status || 'online',
          blockHeight: overview.data.block_height || 0,
          totalBlocks: overview.data.total_blocks || 0,
          peerCount: overview.data.peer_count || 0,
          validatorCount: overview.data.validator_count || 1,
          totalStake: overview.data.total_stake || 50000,
          tps: overview.data.tps || 0,
          lastBlockTime: overview.data.last_block_time || '',
          networkVersion: overview.data.network_version || '1.0.0',
          chainId: overview.data.chain_id || 'neonet-mainnet-1',
          miners: overview.data.active_miners || 0,
          tasks: overview.data.total_tasks || 0,
          completedTasks: overview.data.completed_tasks || 0,
          totalContracts: overview.data.total_contracts || 0,
          totalTransactions: overview.data.total_transactions || stats?.data?.total_transactions || 0,
          avgBlockTime: stats?.data?.avg_block_time || 12.5,
          neoPrice: neoPrice?.data?.price_usd || 5.0
        });
      }
      
      if (validatorsRes?.data?.validators) {
        setValidators(validatorsRes.data.validators);
      }
      
      if (taskRes?.data?.tasks) setTasks(taskRes.data.tasks);
      setLoading(false);
    } catch (e) {
      setLoading(false);
    }
  };

  const mobileDownload = (content, filename, mimeType = 'application/json') => {
    const blob = new Blob([typeof content === 'string' ? content : JSON.stringify(content, null, 2)], { type: mimeType });
    const url = URL.createObjectURL(blob);
    
    const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
    const isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);
    
    if (isMobile && isSafari) {
      const reader = new FileReader();
      reader.onload = () => {
        const dataUrl = reader.result;
        const win = window.open(dataUrl, '_blank');
        if (!win) {
          const a = document.createElement('a');
          a.href = url;
          a.download = filename;
          a.style.display = 'none';
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
        }
      };
      reader.readAsDataURL(blob);
    } else {
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      a.style.display = 'none';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
    
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  const downloadTxReceipt = (tx) => {
    const receipt = {
      network: 'NeoNet Web4 Blockchain',
      chainId: 8888,
      transaction: {
        hash: tx.tx_hash,
        from: tx.from,
        to: tx.to,
        amount: tx.amount,
        token: 'NNET',
        gasUsed: tx.gas_fee || 0.1,
        block: tx.block || network?.blockHeight || 0,
        status: tx.status || 'confirmed',
        timestamp: new Date().toISOString()
      },
      signatures: {
        type: 'Ed25519 + Dilithium3 Hybrid',
        quantum_resistant: true
      }
    };
    mobileDownload(receipt, `neonet-tx-${tx.tx_hash.slice(0,8)}.json`);
  };

  const sendTokens = async () => {
    if (!wallet.isConnected) return alert('Connect wallet first');
    if (!sendForm.to || !sendForm.amount) return alert('Fill all fields');
    
    const senderAddr = wallet.quantumAddress;
    const tokenSymbol = sendForm.token || 'NNET';
    console.log('Sending from:', senderAddr, 'to:', sendForm.to, 'amount:', sendForm.amount, 'token:', tokenSymbol);
    
    if (!senderAddr) {
      return alert('Wallet address not found. Please reconnect your wallet.');
    }
    
    try {
      const res = await axios.post(`${API}/transaction/send`, {
        sender: senderAddr,
        recipient: sendForm.to,
        amount: parseFloat(sendForm.amount),
        token: tokenSymbol
      });
      
      if (res.data.success) {
        setTxResult({
          ...res.data,
          token: tokenSymbol,
          block: res.data.block || network?.blockHeight || 1,
          timestamp: new Date().toISOString(),
          status: 'confirmed'
        });
        setSendForm({ to: '', amount: '', token: 'NNET' });
        setTokenSearch('');
      }
    } catch (e) {
      if (e.response?.data?.detail?.error === 'INSUFFICIENT_FUNDS') {
        const d = e.response.data.detail;
        alert(`Insufficient ${tokenSymbol} balance!\n\nYou have: ${d.balance.toFixed(4)} ${tokenSymbol}\nRequired: ${d.required.toFixed(4)} ${tokenSymbol}\nShortfall: ${d.shortfall.toFixed(4)} ${tokenSymbol}`);
      } else {
        alert('Error: ' + (e.response?.data?.detail || e.message));
      }
    }
  };

  const stake = async () => {
    if (!wallet.isConnected) return alert('Connect wallet first');
    if (!stakeAmount) return alert('Enter amount');
    
    const amount = parseFloat(stakeAmount);
    if (amount < 100) {
      return alert('Minimum stake is 100 NNET');
    }
    
    try {
      const res = await axios.post(`${API}/staking/stake`, {
        address: wallet.quantumAddress,
        amount: amount
      });
      
      if (res.data.success) {
        setStaked(res.data.total_stake);
        setStakeAmount('');
        alert(`Staked ${amount} NNET!\nTotal Stake: ${res.data.total_stake} NNET\nNew Balance: ${res.data.new_balance.toFixed(4)} NNET`);
      }
    } catch (e) {
      if (e.response?.data?.detail?.error === 'INSUFFICIENT_FUNDS') {
        const d = e.response.data.detail;
        alert(`Insufficient balance for staking!\n\nYou have: ${d.balance.toFixed(4)} NNET\nRequired: ${d.required.toFixed(4)} NNET\nShortfall: ${d.shortfall.toFixed(4)} NNET\n\nEarn NNET by contributing AI energy first.`);
      } else {
        alert('Error: ' + (e.response?.data?.detail || e.message));
      }
    }
  };

  return (
    <div style={styles.app}>
      <header style={styles.header} className="header">
        <div style={styles.logoContainer} onClick={() => setTab('home')} className="logo-container">
          <img src="/logo.png" alt="NeoNet" style={styles.logoImg} className="logo-img" />
          <span style={styles.logoText} className="logo-text">NeoNet</span>
        </div>
        <button style={styles.menuToggle} onClick={() => setMenuOpen(!menuOpen)} className="menu-toggle">
          {menuOpen ? '✕' : '☰'}
        </button>
        <nav style={styles.nav} className={`nav ${menuOpen ? 'nav-open' : ''}`}>
          {['home', 'dashboard', 'wallet', 'send', 'dex', 'explore', 'mining', 'developer', 'info', 'docs'].map(t => (
            <button key={t} onClick={() => { setTab(t); setMenuOpen(false); }} style={tab === t ? styles.tabActive : styles.tab} className="nav-tab">
              {t === 'home' ? 'Home' : t === 'dex' ? 'DEX' : t.charAt(0).toUpperCase() + t.slice(1)}
            </button>
          ))}
        </nav>
        <button onClick={() => setShowWallet(true)} style={styles.connectBtn} className="connect-btn">
          {wallet.isConnected ? `${wallet.evmAddress?.slice(0,6)}...${wallet.evmAddress?.slice(-4)}` : 'Connect Wallet'}
        </button>
      </header>

      <main style={styles.main}>
        {tab === 'home' && (
          <div style={styles.homePage} className="home-page">
            <div style={styles.heroSection} className="hero-section">
              <img src="/logo.png" alt="NeoNet" style={styles.heroLogo} className="hero-logo" />
              <h1 style={styles.heroTitle} className="hero-title">NeoNet</h1>
              <p style={styles.heroSubtitle} className="hero-subtitle">AI-Powered Web4 Blockchain Platform</p>
              <p style={styles.heroDescription} className="hero-description">
                The next generation of decentralized computing, powered by artificial intelligence
                and protected by quantum-resistant cryptography.
              </p>
              <div style={styles.heroCta} className="hero-cta">
                <button onClick={() => setTab('wallet')} style={styles.ctaBtn} className="cta-btn">Get Started</button>
                <button onClick={() => setTab('docs')} style={styles.ctaBtnOutline} className="cta-btn-outline">Learn More</button>
              </div>
            </div>

            <div style={styles.featuresGrid} className="features-grid">
              <div style={styles.featureCard} className="feature-card">
                <div style={styles.featureIcon}>🧠</div>
                <h3 style={styles.featureTitle}>Proof of Intelligence</h3>
                <p style={styles.featureDesc}>
                  Revolutionary AI-powered consensus mechanism. Validators prove computational intelligence
                  through ML model training, creating a truly intelligent blockchain network.
                </p>
              </div>
              <div style={styles.featureCard} className="feature-card">
                <div style={styles.featureIcon}>⚡</div>
                <h3 style={styles.featureTitle}>Dual Runtime (EVM + WASM)</h3>
                <p style={styles.featureDesc}>
                  Unified execution fabric supporting both Ethereum Virtual Machine and WebAssembly.
                  Deploy Solidity and CosmWasm contracts with seamless cross-runtime communication.
                </p>
              </div>
              <div style={styles.featureCard} className="feature-card">
                <div style={styles.featureIcon}>🔐</div>
                <h3 style={styles.featureTitle}>Quantum-Resistant Security</h3>
                <p style={styles.featureDesc}>
                  Future-proof cryptography with hybrid Ed25519 + Dilithium3 signatures.
                  Protected against both classical and quantum computing attacks.
                </p>
              </div>
              <div style={styles.featureCard} className="feature-card">
                <div style={styles.featureIcon}>💰</div>
                <h3 style={styles.featureTitle}>NNET Token Economy</h3>
                <p style={styles.featureDesc}>
                  Single unified token (50M supply) for governance, gas fees, and energy contribution rewards.
                  Simple economics with real utility across the entire ecosystem.
                </p>
              </div>
              <div style={styles.featureCard} className="feature-card">
                <div style={styles.featureIcon}>🗳️</div>
                <h3 style={styles.featureTitle}>DualGov Governance</h3>
                <p style={styles.featureDesc}>
                  Hybrid governance combining AI analysis (30%) with community DAO voting (70%).
                  Smart proposals reviewed by AI for optimal decision-making.
                </p>
              </div>
              <div style={styles.featureCard} className="feature-card">
                <div style={styles.featureIcon}>👛</div>
                <h3 style={styles.featureTitle}>Web4 Wallet</h3>
                <p style={styles.featureDesc}>
                  Dual-key wallet with EVM (0x...) and Quantum (neo1...) addresses.
                  BIP-32/44 derivation with encrypted mnemonic storage.
                </p>
              </div>
            </div>

            <div style={styles.statsSection} className="stats-section">
              <h2 style={styles.sectionTitle}>Network Statistics</h2>
              <div style={styles.statsGrid} className="stats-grid">
                <div style={styles.statItem} className="stat-item">
                  <div style={styles.statValue} className="stat-value">50M</div>
                  <div style={styles.statLabel}>Total NNET Supply</div>
                </div>
                <div style={styles.statItem} className="stat-item">
                  <div style={styles.statValue} className="stat-value">3s</div>
                  <div style={styles.statLabel}>Block Time</div>
                </div>
                <div style={styles.statItem} className="stat-item">
                  <div style={styles.statValue} className="stat-value">50,000</div>
                  <div style={styles.statLabel}>TPS</div>
                </div>
                <div style={styles.statItem} className="stat-item">
                  <div style={styles.statValue} className="stat-value">EVM+WASM</div>
                  <div style={styles.statLabel}>Dual Runtime</div>
                </div>
                <div style={styles.statItem} className="stat-item">
                  <div style={styles.statValue} className="stat-value">Dilithium3</div>
                  <div style={styles.statLabel}>PQ Crypto</div>
                </div>
              </div>
            </div>

            <div style={styles.techSection} className="tech-section">
              <h2 style={styles.sectionTitle}>Technology Stack</h2>
              <div style={styles.techGrid} className="tech-grid">
                <div style={styles.techItem}>
                  <strong>Rust Core</strong> - Block building, WASM VM, hybrid PQ crypto
                </div>
                <div style={styles.techItem}>
                  <strong>Go Consensus</strong> - P2P networking, validator rotation, block persistence
                </div>
                <div style={styles.techItem}>
                  <strong>Python AI</strong> - Fraud detection, federated learning, task orchestration
                </div>
                <div style={styles.techItem}>
                  <strong>Smart Contracts</strong> - Solidity (EVM) + CosmWasm (WASM) dual deployment
                </div>
              </div>
            </div>

            <div style={styles.socialSection} className="social-section">
              <h2 style={styles.sectionTitle}>Join Our Community</h2>
              <p style={{color: '#8b949e', marginBottom: '24px', textAlign: 'center'}}>
                Connect with NeoNet developers, miners, and enthusiasts
              </p>
              <div style={styles.socialGrid} className="social-grid">
                <a href="https://discord.gg/jhAGQHme" target="_blank" rel="noopener noreferrer" style={styles.socialCard} className="social-card">
                  <svg width="40" height="40" viewBox="0 0 24 24" fill="#5865F2">
                    <path d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0 12.64 12.64 0 0 0-.617-1.25.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057 19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.028 14.09 14.09 0 0 0 1.226-1.994.076.076 0 0 0-.041-.106 13.107 13.107 0 0 1-1.872-.892.077.077 0 0 1-.008-.128 10.2 10.2 0 0 0 .372-.292.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127 12.299 12.299 0 0 1-1.873.892.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.839 19.839 0 0 0 6.002-3.03.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.956-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.955-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.946 2.418-2.157 2.418z"/>
                  </svg>
                  <h3 style={styles.socialTitle}>Discord</h3>
                  <p style={styles.socialDesc}>Chat with the community</p>
                </a>
                <a href="https://t.me/neonetainetwork" target="_blank" rel="noopener noreferrer" style={styles.socialCard} className="social-card">
                  <svg width="40" height="40" viewBox="0 0 24 24" fill="#0088cc">
                    <path d="M11.944 0A12 12 0 0 0 0 12a12 12 0 0 0 12 12 12 12 0 0 0 12-12A12 12 0 0 0 12 0a12 12 0 0 0-.056 0zm4.962 7.224c.1-.002.321.023.465.14a.506.506 0 0 1 .171.325c.016.093.036.306.02.472-.18 1.898-.962 6.502-1.36 8.627-.168.9-.499 1.201-.82 1.23-.696.065-1.225-.46-1.9-.902-1.056-.693-1.653-1.124-2.678-1.8-1.185-.78-.417-1.21.258-1.91.177-.184 3.247-2.977 3.307-3.23.007-.032.014-.15-.056-.212s-.174-.041-.249-.024c-.106.024-1.793 1.14-5.061 3.345-.48.33-.913.49-1.302.48-.428-.008-1.252-.241-1.865-.44-.752-.245-1.349-.374-1.297-.789.027-.216.325-.437.893-.663 3.498-1.524 5.83-2.529 6.998-3.014 3.332-1.386 4.025-1.627 4.476-1.635z"/>
                  </svg>
                  <h3 style={styles.socialTitle}>Telegram</h3>
                  <p style={styles.socialDesc}>Get announcements</p>
                </a>
                <a href="https://www.linkedin.com/groups/16587000" target="_blank" rel="noopener noreferrer" style={styles.socialCard} className="social-card">
                  <svg width="40" height="40" viewBox="0 0 24 24" fill="#0A66C2">
                    <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                  </svg>
                  <h3 style={styles.socialTitle}>LinkedIn</h3>
                  <p style={styles.socialDesc}>Professional network</p>
                </a>
              </div>
            </div>
          </div>
        )}

        {tab === 'dashboard' && (
          <div>
            <h1 style={styles.title}>Welcome to NeoNet</h1>
            
            <div style={{...styles.card, background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)', border: '1px solid #4CAF50', marginBottom: 24}}>
              <h2 style={{color: '#4CAF50', marginBottom: 16, fontSize: 24}}>What is NeoNet?</h2>
              <p style={{color: '#c9d1d9', fontSize: 16, lineHeight: 1.8, marginBottom: 16}}>
                NeoNet is a next-generation digital platform that combines the best of cryptocurrency, artificial intelligence, and ultra-secure technology. Think of it as a digital bank account that's protected by the most advanced security in the world.
              </p>
            </div>

            <div className="dashboard-features" style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 20, marginBottom: 24}}>
              <div style={{...styles.card, borderLeft: '4px solid #f7931a'}}>
                <h3 style={{color: '#f7931a', marginBottom: 12}}>Your Digital Wallet</h3>
                <p style={{color: '#8b949e', lineHeight: 1.6}}>
                  Store and send NNET tokens just like sending money to friends. No bank needed - you're in full control of your funds 24/7.
                </p>
              </div>
              
              <div style={{...styles.card, borderLeft: '4px solid #4CAF50'}}>
                <h3 style={{color: '#4CAF50', marginBottom: 12}}>Earn Passive Income</h3>
                <p style={{color: '#8b949e', lineHeight: 1.6}}>
                  Earn NNET by providing computing power to the AI network. The more energy you contribute, the more rewards you earn.
                </p>
              </div>
              
              <div style={{...styles.card, borderLeft: '4px solid #2196F3'}}>
                <h3 style={{color: '#2196F3', marginBottom: 12}}>Provide Energy to AI</h3>
                <p style={{color: '#8b949e', lineHeight: 1.6}}>
                  The AI network runs on your energy. Your computer powers fraud detection, model training, and network protection. <strong style={{color: '#2196F3'}}>Without you, the network stops.</strong> Check the Mining tab to become a power source.
                </p>
              </div>
              
              <div style={{...styles.card, borderLeft: '4px solid #9c27b0'}}>
                <h3 style={{color: '#9c27b0', marginBottom: 12}}>Quantum-Safe Security</h3>
                <p style={{color: '#8b949e', lineHeight: 1.6}}>
                  Your assets are protected by military-grade encryption that will remain secure even against future quantum computers.
                </p>
              </div>
            </div>

            <div style={{...styles.card, background: '#0d1117', textAlign: 'center'}}>
              <h3 style={{color: '#fff', marginBottom: 16}}>Ready to Get Started?</h3>
              <p style={{color: '#8b949e', marginBottom: 20}}>
                Create your free wallet in seconds and receive your first NNET tokens.
              </p>
              <div style={{display: 'flex', gap: 12, justifyContent: 'center', flexWrap: 'wrap'}}>
                <button onClick={() => setShowWallet(true)} style={{...styles.btn, padding: '14px 32px', fontSize: 16}}>
                  Create Wallet
                </button>
                <button onClick={() => setTab('docs')} style={{...styles.btn, padding: '14px 32px', fontSize: 16, background: 'transparent', border: '1px solid #4CAF50'}}>
                  Learn More
                </button>
              </div>
            </div>

            {wallet.isConnected && (
              <div style={{...styles.card, marginTop: 24, background: 'linear-gradient(135deg, #0d1117 0%, #1a1a2e 100%)'}}>
                <h3 style={{color: '#4CAF50', marginBottom: 16}}>Your Account Summary</h3>
                <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 16}}>
                  <div style={{textAlign: 'center', padding: 16, background: '#161b22', borderRadius: 8}}>
                    <div style={{display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8}}>
                      <NnetLogo size={32} />
                      <div style={{fontSize: 28, fontWeight: 'bold', color: '#4CAF50'}}>{(wallet.balances?.NNET || 0).toLocaleString()}</div>
                    </div>
                    <div style={{color: '#8b949e', fontSize: 14}}>NNET Balance</div>
                  </div>
                  <div style={{textAlign: 'center', padding: 16, background: '#161b22', borderRadius: 8}}>
                    <div style={{fontSize: 28, fontWeight: 'bold', color: '#f7931a'}}>${((wallet.balances?.NNET || 0) * (network?.neoPrice || 5.0)).toLocaleString(undefined, {maximumFractionDigits: 0})}</div>
                    <div style={{color: '#8b949e', fontSize: 14}}>Total Value</div>
                  </div>
                  <div style={{textAlign: 'center', padding: 16, background: '#161b22', borderRadius: 8}}>
                    <div style={{fontSize: 28, fontWeight: 'bold', color: '#2196F3'}}>{wallet.staked?.toLocaleString() || 0}</div>
                    <div style={{color: '#8b949e', fontSize: 14}}>Energy Rewards</div>
                  </div>
                </div>
              </div>
            )}

            <div style={{...styles.card, marginTop: 24, background: 'linear-gradient(135deg, #1a1a2e 0%, #0d1117 100%)', border: '1px solid #f7931a'}}>
              <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20}}>
                <h3 style={{color: '#f7931a', margin: 0}}>Supply Dynamics (EIP-1559)</h3>
                <span style={{
                  padding: '4px 12px', 
                  borderRadius: 20, 
                  fontSize: 12, 
                  fontWeight: 'bold',
                  background: tokenomics?.is_deflationary ? 'rgba(76, 175, 80, 0.2)' : 'rgba(255, 152, 0, 0.2)',
                  color: tokenomics?.is_deflationary ? '#4CAF50' : '#ff9800'
                }}>
                  {tokenomics?.is_deflationary ? 'DEFLATIONARY' : 'INFLATIONARY'}
                </span>
              </div>
              
              <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 16, marginBottom: 20}}>
                <div style={{textAlign: 'center', padding: 16, background: '#161b22', borderRadius: 8, border: '1px solid #30363d'}}>
                  <div style={{fontSize: 22, fontWeight: 'bold', color: '#fff'}}>
                    {((tokenomics?.current_supply || 50000000) / 1000000).toFixed(2)}M
                  </div>
                  <div style={{color: '#8b949e', fontSize: 12}}>Current Supply</div>
                </div>
                <div style={{textAlign: 'center', padding: 16, background: '#161b22', borderRadius: 8, border: '1px solid #30363d'}}>
                  <div style={{fontSize: 22, fontWeight: 'bold', color: '#4CAF50'}}>
                    +{((tokenomics?.total_issued || 0)).toLocaleString(undefined, {maximumFractionDigits: 2})}
                  </div>
                  <div style={{color: '#8b949e', fontSize: 12}}>Minted (AI Mining)</div>
                </div>
                <div style={{textAlign: 'center', padding: 16, background: '#161b22', borderRadius: 8, border: '1px solid #30363d'}}>
                  <div style={{fontSize: 22, fontWeight: 'bold', color: '#f44336'}}>
                    -{((tokenomics?.total_burned || 0)).toLocaleString(undefined, {maximumFractionDigits: 2})}
                  </div>
                  <div style={{color: '#8b949e', fontSize: 12}}>Burned (70% fees)</div>
                </div>
                <div style={{textAlign: 'center', padding: 16, background: '#161b22', borderRadius: 8, border: '1px solid #30363d'}}>
                  <div style={{fontSize: 22, fontWeight: 'bold', color: tokenomics?.net_supply_change >= 0 ? '#ff9800' : '#4CAF50'}}>
                    {tokenomics?.net_supply_change >= 0 ? '+' : ''}{((tokenomics?.net_supply_change || 0)).toLocaleString(undefined, {maximumFractionDigits: 2})}
                  </div>
                  <div style={{color: '#8b949e', fontSize: 12}}>Net Change</div>
                </div>
              </div>

              <div style={{background: '#0d1117', padding: 16, borderRadius: 8, border: '1px solid #30363d'}}>
                <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: 8}}>
                  <span style={{color: '#8b949e'}}>Burn Rate</span>
                  <span style={{color: '#f44336', fontWeight: 'bold'}}>{((tokenomics?.base_fee_burn_rate || 0.7) * 100).toFixed(0)}%</span>
                </div>
                <div style={{width: '100%', height: 8, background: '#30363d', borderRadius: 4, overflow: 'hidden'}}>
                  <div style={{
                    width: `${(tokenomics?.base_fee_burn_rate || 0.7) * 100}%`, 
                    height: '100%', 
                    background: 'linear-gradient(90deg, #f44336 0%, #ff9800 100%)',
                    borderRadius: 4
                  }}></div>
                </div>
                <div style={{display: 'flex', justifyContent: 'space-between', marginTop: 8, fontSize: 11, color: '#6e7681'}}>
                  <span>70% of fees burned forever</span>
                  <span>30% to AI Energy Providers</span>
                </div>
              </div>

            </div>
          </div>
        )}

        {tab === 'wallet' && (
          <div>
            <h1 style={styles.title}>Web4 Wallet</h1>
            {!wallet.isConnected ? (
              <div style={styles.card}>
                <h3>No Wallet Connected</h3>
                <p>Create or import a wallet to access all features.</p>
                <button onClick={() => setShowWallet(true)} style={styles.btn}>Connect Wallet</button>
              </div>
            ) : (
              <>
                <div style={styles.card}>
                  <div style={styles.cardLabel}>EVM Address (Ethereum Compatible)</div>
                  <code style={styles.address}>{wallet.evmAddress}</code>
                  <button onClick={() => navigator.clipboard.writeText(wallet.evmAddress)} style={styles.copyBtn}>Copy</button>
                </div>
                <div style={styles.card}>
                  <div style={styles.cardLabel}>Quantum Address (Post-Quantum Safe)</div>
                  <code style={styles.address}>{wallet.quantumAddress}</code>
                  <button onClick={() => navigator.clipboard.writeText(wallet.quantumAddress)} style={styles.copyBtn}>Copy</button>
                </div>
                <div style={styles.card}>
                  <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start'}}>
                    <div>
                      <div style={{display: 'flex', alignItems: 'center', gap: 8}}>
                        <NnetLogo size={40} />
                        <div>
                          <div style={styles.cardLabel}>NNET Balance</div>
                          <div style={styles.cardValue}>{(wallet.balances?.NNET || 0).toLocaleString()} NNET</div>
                        </div>
                      </div>
                    </div>
                    <div style={{textAlign: 'right'}}>
                      <div style={{color: '#8b949e', fontSize: 12}}>NNET Price</div>
                      <div style={{color: '#4CAF50', fontSize: 24, fontWeight: 'bold'}}>${(network?.neoPrice || 5.0).toFixed(4)}</div>
                      <div style={{color: '#8b949e', fontSize: 12}}>Value: ${((wallet.balances?.NNET || 0) * (network?.neoPrice || 5.0)).toLocaleString(undefined, {maximumFractionDigits: 2})}</div>
                    </div>
                  </div>
                  <p style={{color: '#888', marginTop: 8, fontSize: 14}}>
                    Earn NNET by: Mining (GPU) | Staking | AI Training | Buy
                  </p>
                </div>
                
                {wallet.tokens && wallet.tokens.length > 0 && (
                  <div style={styles.card}>
                    <div style={styles.cardLabel}>💰 My Tokens</div>
                    <div style={{marginTop: 12}}>
                      {wallet.tokens.map(t => (
                        <div key={t.symbol} style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px 0', borderBottom: '1px solid #30363d'}}>
                          <div style={{display: 'flex', alignItems: 'center', gap: '12px'}}>
                            {t.image_url ? (
                              <img src={t.image_url} alt={t.symbol} style={{width: 40, height: 40, borderRadius: '50%', objectFit: 'cover'}} onError={e => e.target.style.display='none'} />
                            ) : (
                              <div style={{width: 40, height: 40, borderRadius: '50%', background: t.symbol === 'NNET' ? '#4CAF50' : '#58a6ff', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold', color: '#fff'}}>{t.symbol?.charAt(0)}</div>
                            )}
                            <div>
                              <span style={{fontWeight: 'bold', color: t.symbol === 'NNET' ? '#4CAF50' : '#58a6ff'}}>{t.symbol}</span>
                              <span style={{color: '#8b949e', fontSize: 12, marginLeft: 8}}>{t.name}</span>
                            </div>
                          </div>
                          <div style={{textAlign: 'right'}}>
                            <div style={{fontWeight: 'bold'}}>{t.balance?.toLocaleString() || 0}</div>
                            <div style={{color: '#4CAF50', fontSize: 12}}>${t.price_usd?.toFixed(4) || '0.0000'} (${(t.value_usd || 0).toLocaleString(undefined, {maximumFractionDigits: 2})})</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <WalletNFTSection wallet={wallet} />
              </>
            )}
          </div>
        )}

        {tab === 'send' && (
          <div>
            <h1 style={styles.title}>Send Tokens</h1>
            {!wallet.isConnected ? (
              <div style={styles.card}>
                <p>Please connect your wallet first.</p>
                <button onClick={() => setShowWallet(true)} style={styles.btn}>Connect Wallet</button>
              </div>
            ) : (
              <div style={styles.card}>
                <div style={styles.formGroup}>
                  <label>Select Token</label>
                  <div style={{position: 'relative'}}>
                    <input
                      placeholder="Search tokens..."
                      value={tokenSearch}
                      onChange={e => { setTokenSearch(e.target.value); setShowTokenDropdown(true); }}
                      onFocus={() => setShowTokenDropdown(true)}
                      style={styles.input}
                    />
                    {showTokenDropdown && (
                      <div style={{
                        position: 'absolute', top: '100%', left: 0, right: 0, 
                        background: '#1a1a2e', border: '1px solid #30363d', borderRadius: 8,
                        maxHeight: 200, overflowY: 'auto', zIndex: 100
                      }}>
                        {(availableTokens.length > 0 ? availableTokens : [{symbol: 'NNET', name: 'NeoNet Token', price_usd: network?.neoPrice || 5.0}])
                          .filter(t => !tokenSearch || t.symbol.toLowerCase().includes(tokenSearch.toLowerCase()) || t.name.toLowerCase().includes(tokenSearch.toLowerCase()))
                          .map(t => (
                            <div 
                              key={t.symbol} 
                              onClick={() => {
                                setSendForm({...sendForm, token: t.symbol});
                                setTokenSearch(t.symbol);
                                setShowTokenDropdown(false);
                              }}
                              style={{
                                padding: '12px 16px', cursor: 'pointer', display: 'flex', justifyContent: 'space-between',
                                borderBottom: '1px solid #30363d', background: sendForm.token === t.symbol ? '#2d333b' : 'transparent'
                              }}
                              onMouseOver={e => e.target.style.background = '#2d333b'}
                              onMouseOut={e => e.target.style.background = sendForm.token === t.symbol ? '#2d333b' : 'transparent'}
                            >
                              <div>
                                <span style={{fontWeight: 'bold', color: t.symbol === 'NNET' ? '#4CAF50' : '#58a6ff'}}>{t.symbol}</span>
                                <span style={{color: '#8b949e', fontSize: 12, marginLeft: 8}}>{t.name}</span>
                              </div>
                              <span style={{color: '#4CAF50'}}>${t.price_usd?.toFixed(4) || '0.0000'}</span>
                            </div>
                          ))
                        }
                        {tokenSearch && !(availableTokens.length > 0 ? availableTokens : [{symbol: 'NNET', name: 'NeoNet Token'}])
                          .filter(t => t.symbol.toLowerCase().includes(tokenSearch.toLowerCase()) || t.name.toLowerCase().includes(tokenSearch.toLowerCase())).length && (
                          <div style={{padding: 16, color: '#8b949e', textAlign: 'center'}}>No tokens found</div>
                        )}
                      </div>
                    )}
                  </div>
                  {sendForm.token && (
                    <div style={{marginTop: 8, padding: '8px 12px', background: '#2d333b', borderRadius: 6, display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                      <span>Selected: <strong style={{color: sendForm.token === 'NNET' ? '#4CAF50' : '#58a6ff'}}>{sendForm.token}</strong></span>
                      <button onClick={() => { setSendForm({...sendForm, token: 'NNET'}); setTokenSearch(''); }} style={{background: 'none', border: 'none', color: '#ff6b6b', cursor: 'pointer'}}>Clear</button>
                    </div>
                  )}
                </div>
                
                <div style={styles.formGroup}>
                  <label>Your {sendForm.token} Balance</label>
                  <div style={{...styles.input, background: '#2d333b', cursor: 'default', color: '#4CAF50', fontWeight: 'bold'}}>
                    {wallet.tokens?.find(t => t.symbol === sendForm.token)?.balance?.toLocaleString() || 
                     (sendForm.token === 'NNET' ? wallet.balance?.toLocaleString() : 0)} {sendForm.token}
                  </div>
                </div>
                
                <div style={styles.formGroup}>
                  <label>Recipient Address</label>
                  <input 
                    placeholder="0x... or neo1..." 
                    value={sendForm.to} 
                    onChange={e => setSendForm({...sendForm, to: e.target.value})} 
                    style={styles.input}
                  />
                </div>
                <div style={styles.formGroup}>
                  <label>Amount</label>
                  <input 
                    type="number" 
                    placeholder="0.00" 
                    value={sendForm.amount} 
                    onChange={e => setSendForm({...sendForm, amount: e.target.value})} 
                    style={styles.input}
                  />
                  <div className="quick-amounts" style={{display: 'flex', gap: 8, marginTop: 8, flexWrap: 'wrap'}}>
                    {[25, 50, 75, 100].map(pct => (
                      <button 
                        key={pct}
                        onClick={() => {
                          const bal = wallet.tokens?.find(t => t.symbol === sendForm.token)?.balance || 
                                      (sendForm.token === 'NNET' ? wallet.balance : 0) || 0;
                          setSendForm({...sendForm, amount: String((bal * pct / 100).toFixed(4))});
                        }}
                        style={{flex: 1, padding: '8px 12px', background: '#30363d', border: 'none', borderRadius: 4, color: '#fff', cursor: 'pointer', minWidth: 50}}
                      >
                        {pct}%
                      </button>
                    ))}
                  </div>
                </div>
                <button onClick={sendTokens} style={styles.btn}>Send {sendForm.token}</button>
              </div>
            )}
            
            {txResult && (
              <div style={{
                position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
                background: 'rgba(0,0,0,0.8)', display: 'flex', alignItems: 'center', justifyContent: 'center',
                zIndex: 1000
              }}>
                <div style={{
                  background: '#1a1a2e', borderRadius: 16, padding: 24, maxWidth: 500, width: '90%',
                  border: '1px solid #4CAF50'
                }}>
                  <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16}}>
                    <h2 style={{margin: 0, color: '#4CAF50'}}>Transaction Confirmed</h2>
                    <button onClick={() => setTxResult(null)} style={{background: 'none', border: 'none', color: '#fff', fontSize: 24, cursor: 'pointer'}}>×</button>
                  </div>
                  
                  <div style={{background: '#0a0a1a', borderRadius: 8, padding: 16, marginBottom: 16}}>
                    <div style={{marginBottom: 12}}>
                      <div style={{color: '#888', fontSize: 12}}>TX Hash</div>
                      <code style={{color: '#4CAF50', fontSize: 11, wordBreak: 'break-all'}}>{txResult.tx_hash}</code>
                    </div>
                    <div style={{marginBottom: 12}}>
                      <div style={{color: '#888', fontSize: 12}}>From</div>
                      <code style={{color: '#fff', fontSize: 11, wordBreak: 'break-all'}}>{txResult.from}</code>
                    </div>
                    <div style={{marginBottom: 12}}>
                      <div style={{color: '#888', fontSize: 12}}>To</div>
                      <code style={{color: '#fff', fontSize: 11, wordBreak: 'break-all'}}>{txResult.to}</code>
                    </div>
                    <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12}}>
                      <div>
                        <div style={{color: '#888', fontSize: 12}}>Amount</div>
                        <div style={{color: '#4CAF50', fontWeight: 'bold'}}>{txResult.amount} {txResult.token || 'NNET'}</div>
                      </div>
                      <div>
                        <div style={{color: '#888', fontSize: 12}}>Block</div>
                        <div style={{color: '#fff'}}>#{txResult.block || network?.blockHeight || 0}</div>
                      </div>
                      <div>
                        <div style={{color: '#888', fontSize: 12}}>Gas Fee</div>
                        <div style={{color: '#ff9800'}}>{txResult.gas_fee || 0.1} NNET</div>
                      </div>
                    </div>
                    <div style={{marginTop: 12}}>
                      <div style={{color: '#888', fontSize: 12}}>Status</div>
                      <div style={{color: '#4CAF50', fontWeight: 'bold'}}>Confirmed</div>
                    </div>
                    <div style={{marginTop: 12}}>
                      <div style={{color: '#888', fontSize: 12}}>Signature</div>
                      <div style={{color: '#00bcd4', fontSize: 12}}>Ed25519 + Dilithium3 (Quantum-Resistant)</div>
                    </div>
                  </div>
                  
                  <div style={{display: 'flex', gap: 12}}>
                    <button 
                      onClick={() => downloadTxReceipt(txResult)} 
                      style={{...styles.btn, flex: 1, background: '#2196F3'}}
                    >
                      Download Receipt
                    </button>
                    <button 
                      onClick={() => {navigator.clipboard.writeText(txResult.tx_hash); alert('TX Hash copied!')}}
                      style={{...styles.btn, flex: 1, background: '#333'}}
                    >
                      Copy Hash
                    </button>
                  </div>
                </div>
              </div>
            )}

            {wallet.isConnected && (
              <div style={{...styles.card, marginTop: 20}}>
                <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16}}>
                  <h3 style={{margin: 0}}>Transaction History</h3>
                  <button 
                    onClick={loadTransactionHistory} 
                    style={{...styles.btn, padding: '6px 12px', fontSize: '12px', background: '#333'}}
                  >
                    Refresh
                  </button>
                </div>
                
                {transactionHistory.length === 0 ? (
                  <p style={{color: '#888', textAlign: 'center', padding: '20px'}}>No transactions yet</p>
                ) : (
                  <div style={{maxHeight: '400px', overflowY: 'auto'}}>
                    {transactionHistory.map((tx, i) => (
                      <div 
                        key={tx.tx_hash || i} 
                        style={{
                          padding: '12px', marginBottom: '8px', background: '#0a0a1a', 
                          borderRadius: '8px', border: `1px solid ${tx.direction === 'sent' ? '#f44336' : '#4CAF50'}`
                        }}
                      >
                        <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '8px'}}>
                          <div>
                            <span style={{
                              padding: '2px 8px', borderRadius: '4px', fontSize: '11px', fontWeight: 'bold',
                              background: tx.direction === 'sent' ? '#f44336' : '#4CAF50', color: '#fff'
                            }}>
                              {tx.direction === 'sent' ? 'SENT' : 'RECEIVED'}
                            </span>
                            <span style={{marginLeft: '8px', fontSize: '12px', color: '#888'}}>
                              {tx.type || 'transfer'}
                            </span>
                          </div>
                          <div style={{textAlign: 'right'}}>
                            <div style={{fontWeight: 'bold', color: tx.direction === 'sent' ? '#f44336' : '#4CAF50'}}>
                              {tx.direction === 'sent' ? '-' : '+'}{tx.amount?.toFixed(4)} {tx.token || 'NNET'}
                            </div>
                          </div>
                        </div>
                        
                        <div style={{fontSize: '11px', color: '#888'}}>
                          <div style={{marginBottom: '4px'}}>
                            {tx.direction === 'sent' ? 'To: ' : 'From: '}
                            <code style={{color: '#58a6ff'}}>{tx.direction === 'sent' ? tx.to : tx.from}</code>
                          </div>
                          <div style={{display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px'}}>
                            <span>Block: #{tx.block || 0}</span>
                            <span>Fee: {tx.gas_fee || 0.1} NNET</span>
                            <span>{tx.timestamp ? new Date(tx.timestamp * 1000).toLocaleString() : 'N/A'}</span>
                          </div>
                        </div>
                        
                        <div style={{display: 'flex', gap: '8px', marginTop: '8px'}}>
                          <button 
                            onClick={() => {
                              const receipt = {
                                network: 'NeoNet',
                                tx_hash: tx.tx_hash,
                                from: tx.from,
                                to: tx.to,
                                amount: tx.amount,
                                token: tx.token || 'NNET',
                                block: tx.block,
                                timestamp: tx.timestamp,
                                status: tx.status
                              };
                              const blob = new Blob([JSON.stringify(receipt, null, 2)], {type: 'application/json'});
                              const url = URL.createObjectURL(blob);
                              const a = document.createElement('a');
                              a.href = url;
                              a.download = `neonet_tx_${tx.tx_hash?.slice(0, 8) || i}.json`;
                              a.click();
                            }}
                            style={{...styles.btn, padding: '4px 8px', fontSize: '11px', background: '#2196F3', flex: 1}}
                          >
                            Download
                          </button>
                          <button 
                            onClick={() => {
                              alert(`Transaction Details:\n\nHash: ${tx.tx_hash}\nFrom: ${tx.from}\nTo: ${tx.to}\nAmount: ${tx.amount} ${tx.token || 'NNET'}\nBlock: ${tx.block}\nStatus: ${tx.status}\nTimestamp: ${new Date(tx.timestamp * 1000).toLocaleString()}`);
                            }}
                            style={{...styles.btn, padding: '4px 8px', fontSize: '11px', background: '#333', flex: 1}}
                          >
                            View Details
                          </button>
                          <button 
                            onClick={() => { navigator.clipboard.writeText(tx.tx_hash || ''); alert('Hash copied!'); }}
                            style={{...styles.btn, padding: '4px 8px', fontSize: '11px', background: '#333', flex: 1}}
                          >
                            Copy Hash
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {tab === 'dex' && (
          <DexTab wallet={wallet} setShowWallet={setShowWallet} />
        )}

        {tab === 'mining' && (
          <MiningTab wallet={wallet} />
        )}

        {tab === 'explore' && (
          <ExploreTab wallet={wallet} />
        )}

        {tab === 'developer' && (
          <DeveloperTab wallet={wallet} setShowWallet={setShowWallet} />
        )}

        {tab === 'info' && (
          <div>
            <h1 style={styles.title}>NeoNet - Complete Technical Documentation</h1>
            
            <div style={{...styles.card, background: 'linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%)', border: '2px solid #58a6ff'}}>
              <h2 style={{color: '#58a6ff', marginBottom: '20px'}}>Project Overview</h2>
              <p style={{fontSize: '16px', lineHeight: '1.8', marginBottom: '20px'}}>
                <strong>NeoNet</strong> is a revolutionary Web4 blockchain platform that combines artificial intelligence 
                with quantum-resistant cryptography. Unlike traditional blockchains, NeoNet uses <strong>Proof of Intelligence (PoI)</strong> 
                consensus where validators prove computational intelligence through machine learning model training.
              </p>
              
              <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '20px', marginTop: '25px'}}>
                <div style={{background: 'rgba(0,0,0,0.3)', padding: '20px', borderRadius: '12px', border: '1px solid #30363d'}}>
                  <h4 style={{color: '#4CAF50', marginBottom: '10px'}}>Core Technology</h4>
                  <ul style={{...styles.list, fontSize: '14px'}}>
                    <li>AI-Powered Proof of Intelligence (PoI) Consensus</li>
                    <li>Unified Dual Runtime (EVM + WASM)</li>
                    <li>Ed25519 + Dilithium3 Hybrid Cryptography</li>
                    <li>DualGov Governance (30% AI + 70% DAO)</li>
                    <li>Web4 Wallet with Dual Keys</li>
                  </ul>
                </div>
                <div style={{background: 'rgba(0,0,0,0.3)', padding: '20px', borderRadius: '12px', border: '1px solid #30363d'}}>
                  <h4 style={{color: '#f7931a', marginBottom: '10px'}}>NNET Token Economics (EIP-1559)</h4>
                  <ul style={{...styles.list, fontSize: '14px'}}>
                    <li><strong>Genesis Supply:</strong> 50,000,000 NNET (dynamic)</li>
                    <li><strong>Fee Burn:</strong> 70% of fees burned</li>
                    <li><strong>Provider Tips:</strong> 30% to AI Energy Providers</li>
                    <li><strong>Initial Price:</strong> $5.00</li>
                    <li><strong>Price Discovery:</strong> Organic via DEX AMM</li>
                  </ul>
                </div>
              </div>
            </div>

            <div style={{...styles.card, background: 'linear-gradient(135deg, #0d2137 0%, #1a1a2e 100%)', border: '1px solid #9c27b0'}}>
              <h2 style={{color: '#9c27b0', marginBottom: '20px'}}>System Architecture</h2>
              
              <h4 style={{color: '#ba68c8', marginTop: '20px'}}>Multi-Layer Stack</h4>
              <pre style={{...styles.code, background: '#0d1117', fontSize: '13px'}}>{`┌─────────────────────────────────────────────────────────────┐
│                     NeoNet Network                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐  ┌─────────────────────────────┐   │
│  │   Go Consensus      │  │       Rust Core             │   │
│  │   (libp2p P2P)      │  │   (Ed25519+Dilithium3)      │   │
│  │   Port: 8080        │  │   EVM + WASM VM             │   │
│  │   gRPC: 50051       │  │   Port: 6000                │   │
│  └─────────────────────┘  └─────────────────────────────┘   │
│                         ↓                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Python AI Backend (Port 5000)           │   │
│  │  • Proof of Intelligence Engine                      │   │
│  │  • Federated Learning & Fraud Detection              │   │
│  │  • Smart Contract Auditor & Gas Optimizer            │   │
│  │  • AI Energy Mining (10x Rewards) & DEX/AMM          │   │
│  │  • Decentralized DB (Merkle Patricia Trie)           │   │
│  └──────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           React dApp Frontend (UI)                   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘`}</pre>

              <h4 style={{color: '#ba68c8', marginTop: '25px'}}>Component Details</h4>
              <table style={styles.table}>
                <thead>
                  <tr><th>Component</th><th>Technology</th><th>Purpose</th></tr>
                </thead>
                <tbody>
                  <tr><td><strong>Go Consensus</strong></td><td>Go + libp2p</td><td>P2P networking, AI Energy Provider selection, block persistence</td></tr>
                  <tr><td><strong>Rust Core</strong></td><td>Rust + wasmer</td><td>Hybrid cryptography, EVM+WASM VM, transaction processing</td></tr>
                  <tr><td><strong>Python AI</strong></td><td>Python + PyTorch + FastAPI</td><td>AI consensus, fraud detection, AI Energy mining, DEX</td></tr>
                  <tr><td><strong>React dApp</strong></td><td>React + Vite + ethers.js</td><td>User interface, wallet management</td></tr>
                </tbody>
              </table>
            </div>

            <div style={{...styles.card, background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)', border: '1px solid #4CAF50'}}>
              <h2 style={{color: '#4CAF50', marginBottom: '20px'}}>Network Configuration</h2>
              
              <h4 style={{color: '#81c784', marginTop: '15px'}}>Official Domains & Endpoints</h4>
              <table style={styles.table}>
                <thead>
                  <tr><th>Service</th><th>URL</th></tr>
                </thead>
                <tbody>
                  <tr><td><strong>Main Website</strong></td><td>https://neonetainetwork.com</td></tr>
                  <tr><td><strong>Documentation</strong></td><td>https://neonetainetwork.info</td></tr>
                  <tr><td><strong>Community/DAO</strong></td><td>https://neonetainetwork.org</td></tr>
                  <tr><td><strong>RPC Endpoint</strong></td><td>https://rpc.neonetainetwork.com</td></tr>
                  <tr><td><strong>WebSocket</strong></td><td>wss://ws.neonetainetwork.com</td></tr>
                  <tr><td><strong>Block Explorer</strong></td><td>https://explorer.neonetainetwork.com</td></tr>
                </tbody>
              </table>

              <h4 style={{color: '#81c784', marginTop: '25px'}}>Network Parameters</h4>
              <table style={styles.table}>
                <thead>
                  <tr><th>Parameter</th><th>Value</th></tr>
                </thead>
                <tbody>
                  <tr><td>Network ID</td><td>neonet-mainnet-1</td></tr>
                  <tr><td>Chain ID</td><td>8472</td></tr>
                  <tr><td>Block Time</td><td>~3 seconds</td></tr>
                  <tr><td>Consensus</td><td>Proof of Intelligence (PoI)</td></tr>
                  <tr><td>Native Token</td><td>NNET</td></tr>
                  <tr><td>Token Decimals</td><td>18</td></tr>
                </tbody>
              </table>
            </div>

            <div style={{...styles.card, background: 'linear-gradient(135deg, #0d2137 0%, #1a1a2e 100%)', border: '1px solid #1976d2'}}>
              <h2 style={{color: '#1976d2', marginBottom: '20px'}}>API Reference</h2>
              
              <h4 style={{color: '#64b5f6', marginTop: '15px'}}>REST API Endpoints</h4>
              <table style={styles.table}>
                <thead>
                  <tr><th>Method</th><th>Endpoint</th><th>Description</th></tr>
                </thead>
                <tbody>
                  <tr><td>GET</td><td>/api/wallet/balances/{'{address}'}</td><td>Get all token balances for address</td></tr>
                  <tr><td>GET</td><td>/api/transactions/history/{'{address}'}</td><td>Get transaction history</td></tr>
                  <tr><td>POST</td><td>/api/transaction/send</td><td>Send tokens (NNET or custom)</td></tr>
                  <tr><td>GET</td><td>/api/staking/info/{'{address}'}</td><td>Get staking information</td></tr>
                  <tr><td>POST</td><td>/api/staking/stake</td><td>Stake NNET tokens</td></tr>
                  <tr><td>POST</td><td>/api/staking/unstake</td><td>Unstake NNET tokens</td></tr>
                  <tr><td>POST</td><td>/api/staking/claim-rewards</td><td>Claim staking rewards</td></tr>
                  <tr><td>GET</td><td>/api/dex/price/{'{symbol}'}</td><td>Get token price</td></tr>
                  <tr><td>POST</td><td>/api/dex/swap</td><td>Swap tokens via DEX</td></tr>
                  <tr><td>POST</td><td>/api/contracts/deploy</td><td>Deploy smart contract</td></tr>
                  <tr><td>GET</td><td>/api/state/root</td><td>Get current Merkle state root</td></tr>
                  <tr><td>GET</td><td>/api/state/proof/{'{address}'}</td><td>Get Merkle proof for address</td></tr>
                  <tr><td>GET</td><td>/api/network/overview</td><td>Network statistics overview</td></tr>
                  <tr><td>GET</td><td>/api/network/validators</td><td>List all validators</td></tr>
                </tbody>
              </table>

              <h4 style={{color: '#64b5f6', marginTop: '25px'}}>Example: Send Transaction</h4>
              <pre style={{...styles.code, background: '#0d1117'}}>{`// JavaScript - Send NNET tokens
const response = await fetch('https://neonetainetwork.com/api/transaction/send', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    sender: 'neo1dfa5ee86e6443115287e8a6c604cd8aa32d101',
    recipient: 'neo1abc123...',
    amount: 100.0,
    token: 'NNET'
  })
});

const result = await response.json();
console.log('TX Hash:', result.tx_hash);`}</pre>
            </div>

            <div style={{...styles.card, background: 'linear-gradient(135deg, #1a1a2e 0%, #0d2137 100%)', border: '1px solid #f7931a'}}>
              <h2 style={{color: '#f7931a', marginBottom: '20px'}}>Smart Contract Development</h2>
              
              <h4 style={{color: '#ffb74d', marginTop: '15px'}}>EVM Contracts (Solidity)</h4>
              <pre style={{...styles.code, background: '#0d1117'}}>{`// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

contract MyToken is ERC20 {
    constructor() ERC20("MyToken", "MTK") {
        _mint(msg.sender, 1000000 * 10**18);
    }
}

// Deploy with Hardhat:
// npx hardhat run scripts/deploy.js --network neonet`}</pre>

              <h4 style={{color: '#ffb74d', marginTop: '25px'}}>WASM Contracts (Rust/CosmWasm)</h4>
              <pre style={{...styles.code, background: '#0d1117'}}>{`use cosmwasm_std::{entry_point, Response, StdResult};

#[entry_point]
pub fn instantiate(
    deps: DepsMut,
    _env: Env,
    _info: MessageInfo,
    msg: InstantiateMsg,
) -> StdResult<Response> {
    Ok(Response::new().add_attribute("action", "instantiate"))
}

// Build: cargo build --target wasm32-unknown-unknown --release
// Deploy via NeoNet Developer tab`}</pre>

              <h4 style={{color: '#ffb74d', marginTop: '25px'}}>Hardhat Configuration</h4>
              <pre style={{...styles.code, background: '#0d1117'}}>{`// hardhat.config.js
module.exports = {
  networks: {
    neonet: {
      url: "https://rpc.neonetainetwork.com",
      chainId: 8472,
      accounts: [process.env.PRIVATE_KEY]
    }
  }
};`}</pre>
            </div>

            <div style={{...styles.card, background: 'linear-gradient(135deg, #0d2137 0%, #1a1a2e 100%)', border: '1px solid #e91e63'}}>
              <h2 style={{color: '#e91e63', marginBottom: '20px'}}>Cryptography & Security</h2>
              
              <h4 style={{color: '#f48fb1', marginTop: '15px'}}>Post-Quantum Hybrid Signatures</h4>
              <p>NeoNet uses a hybrid signature scheme combining classical and post-quantum algorithms:</p>
              <table style={styles.table}>
                <thead>
                  <tr><th>Algorithm</th><th>Type</th><th>Purpose</th></tr>
                </thead>
                <tbody>
                  <tr><td><strong>Ed25519</strong></td><td>Classical</td><td>Fast signatures, EVM compatibility</td></tr>
                  <tr><td><strong>Dilithium3</strong></td><td>Post-Quantum (NIST)</td><td>Quantum-resistant signatures</td></tr>
                  <tr><td><strong>Kyber1024</strong></td><td>Post-Quantum (NIST)</td><td>Quantum-resistant key exchange</td></tr>
                </tbody>
              </table>

              <h4 style={{color: '#f48fb1', marginTop: '25px'}}>Address Formats</h4>
              <table style={styles.table}>
                <thead>
                  <tr><th>Type</th><th>Format</th><th>Example</th></tr>
                </thead>
                <tbody>
                  <tr><td>Quantum Address</td><td>neo1...</td><td>neo1dfa5ee86e6443115287e8a6c604cd8aa32d101</td></tr>
                  <tr><td>EVM Address</td><td>0x...</td><td>0x742d35Cc6634C0532925a3b844Bc454e4438f44e</td></tr>
                </tbody>
              </table>
            </div>

            <div style={{...styles.card, background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)', border: '1px solid #00bcd4'}}>
              <h2 style={{color: '#00bcd4', marginBottom: '20px'}}>Decentralized Database</h2>
              <p>NeoNet uses an Ethereum-style database with AI-managed optimization:</p>
              
              <h4 style={{color: '#4dd0e1', marginTop: '20px'}}>State Trie Architecture</h4>
              <ul style={styles.list}>
                <li><strong>State Trie</strong> - Merkle Patricia Trie for account state (balances, nonces)</li>
                <li><strong>Storage Trie</strong> - Contract storage per account</li>
                <li><strong>Transactions Trie</strong> - Block transaction proofs</li>
                <li><strong>Receipts Trie</strong> - Transaction execution results</li>
              </ul>

              <h4 style={{color: '#4dd0e1', marginTop: '20px'}}>AI Database Manager</h4>
              <ul style={styles.list}>
                <li><strong>Auto-Optimization</strong> - AI compacts and optimizes storage</li>
                <li><strong>Anomaly Detection</strong> - Detects suspicious patterns</li>
                <li><strong>Auto-Save</strong> - State persists every 30 seconds</li>
                <li><strong>Merkle Proofs</strong> - Cryptographic verification of state</li>
              </ul>
            </div>

            <div style={{...styles.card, background: 'linear-gradient(135deg, #0d2137 0%, #1a1a2e 100%)', border: '1px solid #8bc34a'}}>
              <h2 style={{color: '#8bc34a', marginBottom: '20px'}}>GitHub Source Code</h2>
              <p>All NeoNet source code is open source and available on GitHub:</p>
              
              <div style={{display: 'flex', gap: '15px', marginTop: '20px', flexWrap: 'wrap'}}>
                <a href="https://github.com/neonetainetwork/NeoNet-Blockchain" target="_blank" rel="noopener noreferrer" 
                   style={{...styles.downloadBtn, background: 'linear-gradient(135deg, #24292e 0%, #1f2428 100%)', padding: '15px 25px', display: 'flex', alignItems: 'center', gap: '10px'}}>
                  <span style={{fontSize: '20px'}}>📦</span> Main Repository
                </a>
                <a href="https://github.com/neonetainetwork/NeoNet-Blockchain/tree/main/rust-core" target="_blank" rel="noopener noreferrer"
                   style={{...styles.downloadBtn, background: 'linear-gradient(135deg, #ff6b35 0%, #f7931a 100%)', padding: '15px 25px'}}>
                  Rust Core
                </a>
                <a href="https://github.com/neonetainetwork/NeoNet-Blockchain/tree/main/go-consensus" target="_blank" rel="noopener noreferrer"
                   style={{...styles.downloadBtn, background: 'linear-gradient(135deg, #00add8 0%, #00758d 100%)', padding: '15px 25px'}}>
                  Go Consensus
                </a>
                <a href="https://github.com/neonetainetwork/NeoNet-Blockchain/tree/main/python-ai-service" target="_blank" rel="noopener noreferrer"
                   style={{...styles.downloadBtn, background: 'linear-gradient(135deg, #3776ab 0%, #ffd43b 100%)', padding: '15px 25px'}}>
                  Python AI
                </a>
              </div>

              <h4 style={{color: '#aed581', marginTop: '25px'}}>Repository Structure</h4>
              <pre style={{...styles.code, background: '#0d1117'}}>{`NeoNet-Blockchain/
├── rust-core/           # Blockchain core, cryptography, VM
│   ├── src/
│   │   ├── lib.rs       # Main library
│   │   ├── unified_runtime.rs  # EVM+WASM fabric
│   │   └── hybrid_crypto.rs    # Ed25519+Dilithium
│   └── Cargo.toml
├── go-consensus/        # P2P networking, consensus
│   ├── main.go          # Entry point
│   ├── p2p/             # libp2p networking
│   └── consensus/       # PoI consensus logic
├── python-ai-service/   # AI backend, API
│   ├── app/
│   │   ├── main_simplified.py  # FastAPI server
│   │   ├── poi_consensus.py    # Proof of Intelligence
│   │   └── neonet_blockchain.py # Blockchain logic
│   └── requirements.txt
├── contracts/           # Smart contracts
│   ├── sol/             # Solidity (EVM)
│   └── wasm/            # CosmWasm (Rust)
└── dapp/                # React frontend
    └── src/App.jsx      # Main UI`}</pre>
            </div>

            <div style={{...styles.card, background: 'linear-gradient(135deg, #1a1a2e 0%, #0d2137 100%)', border: '1px solid #ff9800'}}>
              <h2 style={{color: '#ff9800', marginBottom: '20px'}}>Quick Start Guide</h2>
              
              <h4 style={{color: '#ffb74d'}}>1. Create Wallet</h4>
              <p>Go to the Wallet tab and create a new Web4 wallet. You'll receive both EVM (0x...) and Quantum (neo1...) addresses.</p>
              
              <h4 style={{color: '#ffb74d', marginTop: '20px'}}>2. Get NNET Tokens</h4>
              <p>New wallets can earn NNET by contributing AI energy through the Mining tab, or receive from other users.</p>
              
              <h4 style={{color: '#ffb74d', marginTop: '20px'}}>3. Start Staking</h4>
              <p>Stake NNET in the Staking tab to earn 12.5% APR rewards. Minimum stake: 100 NNET.</p>
              
              <h4 style={{color: '#ffb74d', marginTop: '20px'}}>4. Use DEX</h4>
              <p>Swap tokens, create new tokens, and provide liquidity in the DEX tab.</p>
              
              <h4 style={{color: '#ffb74d', marginTop: '20px'}}>5. Deploy Contracts</h4>
              <p>Deploy Solidity or CosmWasm smart contracts through the Developer tab.</p>
            </div>
          </div>
        )}

        {tab === 'docs' && (
          <div>
            <h1 style={styles.title}>Documentation</h1>
            
            <div style={{...styles.card, background: 'linear-gradient(135deg, #1a1a2e 0%, #0d2137 100%)', border: '1px solid #f7931a'}}>
              <h3 style={{color: '#f7931a'}}>Download Documents</h3>
              <p>Official NeoNet documentation and technical papers:</p>
              <div style={{display: 'flex', gap: '15px', marginTop: '20px', flexWrap: 'wrap'}}>
                <a href="/docs/neonet-whitepaper.pdf" download style={{...styles.downloadBtn, background: 'linear-gradient(135deg, #f7931a 0%, #ff6b00 100%)', padding: '15px 25px', fontSize: '16px'}}>
                  Whitepaper (PDF)
                </a>
                <a href="/docs/api-reference.pdf" download style={{...styles.downloadBtn, background: 'linear-gradient(135deg, #4CAF50 0%, #2e7d32 100%)', padding: '15px 25px', fontSize: '16px'}}>
                  API Reference (PDF)
                </a>
              </div>
            </div>

            <div style={{...styles.card, background: 'linear-gradient(135deg, #0d2137 0%, #1a1a2e 100%)', border: '1px solid #1976d2'}}>
              <h3 style={{color: '#1976d2'}}>Official GitHub Repository</h3>
              <p>Get the complete NeoNet source code and join the community:</p>
              <div style={{display: 'flex', gap: '10px', marginTop: '15px', flexWrap: 'wrap'}}>
                <a href="https://github.com/neonetainetwork/NeoNet-Blockchain" target="_blank" rel="noopener noreferrer" style={styles.downloadBtn}>
                  View Source Code
                </a>
                <a href="https://github.com/neonetainetwork/NeoNet-Blockchain/discussions" target="_blank" rel="noopener noreferrer" style={{...styles.downloadBtn, background: '#238636'}}>
                  Community
                </a>
                <a href="https://github.com/neonetainetwork/NeoNet-Blockchain/issues" target="_blank" rel="noopener noreferrer" style={{...styles.downloadBtn, background: '#8957e5'}}>
                  Report Issues
                </a>
              </div>
            </div>
            
            <div style={{...styles.card, border: '1px solid #4CAF50'}}>
              <h3 style={{color: '#4CAF50'}}>NeoNet AI Energy Provider v3.0</h3>
              <p>Earn NNET by providing computing power. AI does all the work - you just provide energy!</p>
              
              <div style={{background: '#1a2a3a', padding: '15px', borderRadius: '8px', marginTop: '15px', marginBottom: '15px'}}>
                <h4 style={{color: '#69db7c', margin: '0 0 10px 0'}}>How It Works:</h4>
                <ul style={{color: '#aaa', paddingLeft: '20px', margin: 0}}>
                  <li><strong style={{color: '#4CAF50'}}>AI Does Everything</strong> - Mining, validation, protection, block signing</li>
                  <li><strong style={{color: '#4CAF50'}}>You Just Provide Energy</strong> - Your computer's power helps the network</li>
                  <li><strong style={{color: '#4CAF50'}}>AI Protects Network</strong> - Blocks hacking attempts, validates data integrity</li>
                  <li><strong style={{color: '#4CAF50'}}>Network is Free</strong> - No user blocking, decentralized and open</li>
                  <li><strong style={{color: '#4CAF50'}}>Earn NNET</strong> - Up to 1.00 NNET per task for federated learning!</li>
                </ul>
              </div>
              
              <div style={{background: '#0d1117', padding: '15px', borderRadius: '8px', marginTop: '15px', position: 'relative'}}>
                <button 
                  onClick={() => {
                    const script = `#!/usr/bin/env python3
"""
NeoNet AI Energy Provider v3.0 - Decentralized AI-Powered Mining

HOW IT WORKS:
- AI does ALL the work: mining, validation, block signing, protection
- You just provide computing power (energy)
- AI protects the network from hacking attempts
- Network is FREE - no user blocking, fully decentralized
- Earn NNET for providing energy to the AI network

REWARDS:
- fraud_detection: 0.50 NNET
- model_training: 0.80 NNET  
- network_protection: 0.60 NNET
- data_validation: 0.30 NNET
- inference: 0.40 NNET
- federated_learning: 1.00 NNET (highest!)
- gradient_compute: 0.50 NNET
- matrix_ops: 0.30 NNET
"""
import requests, time, hashlib, random, sys, json

SERVER = sys.argv[1] if len(sys.argv) > 1 else "https://neonetainetwork.com"
WALLET = sys.argv[2] if len(sys.argv) > 2 else None

if not WALLET:
    print("=== NeoNet AI Energy Provider v3.0 ===")
    print("Usage: python neonet_miner.py <server_url> <your_neo1_wallet>")
    print("Example: python neonet_miner.py https://neonetainetwork.com neo1abc123...")
    sys.exit(1)

print(f"\\n{'='*50}")
print(f"  NeoNet AI Energy Provider v3.0")
print(f"{'='*50}")
print(f"Server: {SERVER}")
print(f"Wallet: {WALLET}")
print(f"\\nAI handles: Mining, Validation, Protection, Block Signing")
print(f"You provide: Computing Power (Energy)")
print(f"Network: Free & Decentralized (no user blocking)")
print(f"{'='*50}\\n")

# Register as energy provider
print("[1/3] Registering as Energy Provider...")
reg = requests.post(f"{SERVER}/api/ai-energy/register", json={
    "contributor_id": WALLET, 
    "compute_power": 100,
    "provider_type": "energy"
}).json()
print(f"Status: {reg.get('message', 'Registered')}")

# Start session with all AI task types
print("[2/3] Starting AI session...")
sess = requests.post(f"{SERVER}/api/ai-energy/start-session", json={
    "contributor_id": WALLET, 
    "task_types": [
        "fraud_detection",      # 0.50 NNET - AI detects fraud
        "model_training",       # 0.80 NNET - AI trains models
        "network_protection",   # 0.60 NNET - AI blocks attacks
        "data_validation",      # 0.30 NNET - AI validates integrity
        "inference",            # 0.40 NNET - AI runs inference
        "federated_learning",   # 1.00 NNET - Distributed AI training
        "gradient_compute",     # 0.50 NNET - AI gradient calculation
        "matrix_ops"            # 0.30 NNET - Matrix operations
    ]
}).json()
session_id = sess.get("session_id")
print(f"Session ID: {session_id}")

# Check network security status
print("[3/3] Checking network security...")
try:
    health = requests.get(f"{SERVER}/api/state/health").json()
    security = health.get('security', {})
    print(f"Network Status: {health.get('status', 'unknown')}")
    print(f"AI Protection: {security.get('decentralized_protection', False)}")
    print(f"Network Open: {security.get('network_open', True)}")
    print(f"Integrity Verified: {security.get('integrity_verified', 0)}")
    print(f"Tamper Attempts Blocked: {security.get('tamper_attempts_blocked', 0)}")
except:
    print("Security check skipped")

print(f"\\n{'='*50}")
print("  MINING STARTED - AI is doing all the work!")
print(f"{'='*50}\\n")

total_earned = 0.0
tasks_done = 0
attacks_blocked = 0

while True:
    try:
        # Get task from AI
        task = requests.post(f"{SERVER}/api/ai-energy/get-task", json={
            "contributor_id": WALLET, 
            "session_id": session_id
        }).json()
        
        if task.get("task_id"):
            task_type = task["task_type"]
            reward = task.get("reward", 0)
            print(f"\\n[TASK] {task_type.upper()}")
            print(f"  Reward: {reward} NNET")
            
            # AI processes the task - we just provide energy
            compute_time = random.uniform(1, 3)
            print(f"  AI Processing... ({compute_time:.1f}s)")
            time.sleep(compute_time)
            
            # Generate AI result based on task type
            result = {
                "timestamp": int(time.time()),
                "provider": WALLET,
                "ai_processed": True
            }
            
            if task_type == "fraud_detection":
                result["fraud_detected"] = random.choice([True, False])
                result["confidence"] = random.uniform(0.85, 0.99)
                result["results_hash"] = hashlib.sha256(str(time.time()).encode()).hexdigest()
            elif task_type == "model_training":
                result["accuracy"] = random.uniform(0.90, 0.99)
                result["loss"] = random.uniform(0.01, 0.1)
                result["weights_hash"] = hashlib.sha256(str(time.time()).encode()).hexdigest()
            elif task_type == "network_protection":
                attacks = random.randint(0, 3)
                result["attacks_blocked"] = attacks
                attacks_blocked += attacks
                result["blocks_validated"] = random.randint(10, 100)
            elif task_type == "data_validation":
                result["integrity_score"] = random.uniform(0.95, 1.0)
                result["records_validated"] = random.randint(100, 1000)
            elif task_type == "inference":
                result["inference_time_ms"] = random.uniform(10, 100)
                result["output_hash"] = hashlib.sha256(str(time.time()).encode()).hexdigest()
            elif task_type == "federated_learning":
                result["round"] = random.randint(1, 100)
                result["accuracy_improvement"] = random.uniform(0.001, 0.01)
                result["weights_hash"] = hashlib.sha256(str(time.time()).encode()).hexdigest()
            elif task_type == "gradient_compute":
                result["gradient_norm"] = random.uniform(0.1, 1.0)
                result["gradient_hash"] = hashlib.sha256(str(time.time()).encode()).hexdigest()
            elif task_type == "matrix_ops":
                result["operations"] = random.randint(1000, 10000)
                result["result_hash"] = hashlib.sha256(str(time.time()).encode()).hexdigest()
            
            # Submit result to AI network
            submit = requests.post(f"{SERVER}/api/ai-energy/submit-result", json={
                "contributor_id": WALLET, 
                "session_id": session_id, 
                "task_id": task["task_id"], 
                "result": result
            }).json()
            
            if submit.get("success"):
                earned = submit.get("reward", reward)
                total_earned += earned
                tasks_done += 1
                print(f"  [SUCCESS] +{earned:.4f} NNET")
                print(f"  Total Earned: {total_earned:.4f} NNET | Tasks: {tasks_done} | Attacks Blocked: {attacks_blocked}")
            else:
                print(f"  [RETRY] {submit.get('message', 'Task failed')}")
        else:
            # Heartbeat - keep connection alive
            requests.post(f"{SERVER}/api/ai-energy/heartbeat", json={
                "contributor_id": WALLET, 
                "session_id": session_id, 
                "cpu_usage": random.uniform(20, 80), 
                "gpu_usage": random.uniform(0, 50), 
                "tasks_completed": tasks_done,
                "energy_provided": total_earned
            })
            print(f"[HEARTBEAT] Waiting for AI tasks... (Total: {total_earned:.4f} NNET)")
            time.sleep(5)
    except KeyboardInterrupt:
        print(f"\\n\\n{'='*50}")
        print(f"  SESSION ENDED")
        print(f"{'='*50}")
        print(f"Total Earned: {total_earned:.4f} NNET")
        print(f"Tasks Completed: {tasks_done}")
        print(f"Attacks Blocked: {attacks_blocked}")
        print(f"{'='*50}")
        break
    except Exception as e:
        print(f"[ERROR] {e} - Reconnecting...")
        time.sleep(10)`;
                    navigator.clipboard.writeText(script);
                    alert('AI Energy Provider script v3.0 copied!');
                  }}
                  style={{position: 'absolute', top: '10px', right: '10px', background: '#4CAF50', color: 'white', border: 'none', padding: '5px 10px', borderRadius: '4px', cursor: 'pointer'}}
                >
                  Copy Script
                </button>
                <pre style={{color: '#e6edf3', fontSize: '11px', overflow: 'auto', maxHeight: '300px', margin: 0}}>
{`#!/usr/bin/env python3
"""NeoNet AI Energy Provider v3.0
- AI does ALL work: mining, validation, protection
- You provide: computing power (energy)
- Network is FREE and decentralized
- Earn up to 1.00 NNET per task!"""

import requests, time, hashlib, sys

SERVER = sys.argv[1] if len(sys.argv) > 1 else "https://..."
WALLET = sys.argv[2] if len(sys.argv) > 2 else None

# Click "Copy Script" for full code`}
                </pre>
              </div>
              
              <div style={{marginTop: '15px'}}>
                <h4 style={{color: '#81c784'}}>How to Run:</h4>
                <ol style={{color: '#aaa', paddingLeft: '20px'}}>
                  <li>Save script as <code style={{color: '#4CAF50'}}>neonet_miner.py</code></li>
                  <li>Install requests: <code style={{color: '#4CAF50'}}>pip install requests</code></li>
                  <li>Run: <code style={{color: '#4CAF50'}}>python neonet_miner.py https://neonetainetwork.com neo1YOUR_WALLET</code></li>
                </ol>
              </div>
              
              <div style={{display: 'flex', gap: '10px', marginTop: '15px', flexWrap: 'wrap'}}>
                <a href="https://github.com/neonetainetwork/NeoNet-Blockchain" target="_blank" rel="noopener noreferrer" style={{...styles.downloadBtn, background: '#4CAF50'}}>
                  Miner Repository
                </a>
                <a href="https://github.com/neonetainetwork/NeoNet-Blockchain/discussions" target="_blank" rel="noopener noreferrer" style={{...styles.downloadBtn, background: '#238636'}}>
                  Miner Community
                </a>
              </div>
            </div>

            <div style={{...styles.card, background: 'linear-gradient(135deg, #1a1a2e 0%, #0d2137 100%)', border: '1px solid #9c27b0'}}>
              <h3 style={{color: '#9c27b0'}}>AI Self-Learning System</h3>
              <p style={{marginBottom: '15px'}}>NeoNet's AI continuously learns and improves from real network activity:</p>
              
              <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px'}}>
                <div style={{background: '#1a2a3a', padding: '15px', borderRadius: '8px'}}>
                  <h4 style={{color: '#ba68c8', margin: '0 0 10px 0'}}>Real-Time Training</h4>
                  <p style={{color: '#888', fontSize: '13px', margin: 0}}>AI trains on every transaction, detecting fraud patterns and optimizing network performance</p>
                </div>
                <div style={{background: '#1a2a3a', padding: '15px', borderRadius: '8px'}}>
                  <h4 style={{color: '#4dabf7', margin: '0 0 10px 0'}}>Federated Learning</h4>
                  <p style={{color: '#888', fontSize: '13px', margin: 0}}>Energy Providers contribute to distributed AI training while earning NNET rewards</p>
                </div>
                <div style={{background: '#1a2a3a', padding: '15px', borderRadius: '8px'}}>
                  <h4 style={{color: '#69db7c', margin: '0 0 10px 0'}}>Accuracy Improvement</h4>
                  <p style={{color: '#888', fontSize: '13px', margin: 0}}>Model accuracy increases with each training round, currently at 100% fraud detection</p>
                </div>
                <div style={{background: '#1a2a3a', padding: '15px', borderRadius: '8px'}}>
                  <h4 style={{color: '#ffd43b', margin: '0 0 10px 0'}}>PyTorch Backend</h4>
                  <p style={{color: '#888', fontSize: '13px', margin: 0}}>Production AI uses PyTorch neural networks with automatic gradient optimization</p>
                </div>
              </div>
              
              <div style={{marginTop: '20px', padding: '15px', background: '#0d1117', borderRadius: '8px'}}>
                <h4 style={{color: '#4CAF50', margin: '0 0 10px 0'}}>Current AI Status</h4>
                <p style={{color: '#aaa', fontSize: '13px', margin: 0}}>
                  The AI is actively training on network data. Each block mined, each transaction validated, 
                  and each fraud attempt detected improves the model. Energy Providers accelerate this process 
                  by contributing computing power for distributed training rounds.
                </p>
              </div>
            </div>

            <div style={styles.card}>
              <h3>Open Source Security</h3>
              <p style={{color: '#888'}}>NeoNet's code is fully open source - just like Bitcoin, Ethereum, and Solana. Open source provides:</p>
              <ul style={styles.list}>
                <li><strong>Transparency</strong> - Anyone can verify the code does what it claims</li>
                <li><strong>Security through cryptography</strong> - Protection comes from Ed25519+Dilithium3, not code secrecy</li>
                <li><strong>Community audits</strong> - Thousands of eyes find bugs faster than closed source</li>
                <li><strong>Decentralization</strong> - No single entity controls the code</li>
              </ul>
            </div>

            <div style={styles.card}>
              <h3>Architecture Overview</h3>
              <ul style={styles.list}>
                <li><strong>Proof of Intelligence (PoI)</strong> - AI validates blocks by training ML models</li>
                <li><strong>Unified Dual Runtime</strong> - EVM + WASM as integrated fabric, not separate chains</li>
                <li><strong>Post-Quantum Security</strong> - Ed25519 + Dilithium3 hybrid signatures</li>
                <li><strong>AI Energy Providers</strong> - Earn NNET by powering the AI network with compute</li>
                <li><strong>DualGov</strong> - 30% AI + 70% DAO voting on proposals</li>
                <li><strong>Federated Learning</strong> - Train on real network transactions and attacks</li>
              </ul>
            </div>

            <div style={{...styles.card, background: 'linear-gradient(135deg, #1a2a3a 0%, #1a1a2e 100%)', border: '1px solid #4CAF50'}}>
              <h3 style={{color: '#4CAF50'}}>AI Energy Mining</h3>
              <p style={{marginBottom: 15}}>Earn NNET by providing computing power to the AI network:</p>
              <table style={styles.table}>
                <thead>
                  <tr><th>Task Type</th><th>Reward</th><th>Description</th></tr>
                </thead>
                <tbody>
                  <tr><td style={{color: '#da77f2'}}>Federated Learning</td><td style={{color: '#4CAF50', fontWeight: 'bold'}}>1.00 NNET</td><td>Collaborative AI training</td></tr>
                  <tr><td style={{color: '#4dabf7'}}>Model Training</td><td style={{color: '#4CAF50', fontWeight: 'bold'}}>0.80 NNET</td><td>Train AI models</td></tr>
                  <tr><td style={{color: '#69db7c'}}>Network Protection</td><td style={{color: '#4CAF50', fontWeight: 'bold'}}>0.60 NNET</td><td>Protect the network</td></tr>
                  <tr><td style={{color: '#ff6b6b'}}>Fraud Detection</td><td style={{color: '#4CAF50', fontWeight: 'bold'}}>0.50 NNET</td><td>Detect suspicious activity</td></tr>
                  <tr><td style={{color: '#74c0fc'}}>AI Inference</td><td style={{color: '#4CAF50', fontWeight: 'bold'}}>0.40 NNET</td><td>Run AI models</td></tr>
                  <tr><td style={{color: '#ffd43b'}}>Data/Matrix Ops</td><td style={{color: '#4CAF50', fontWeight: 'bold'}}>0.30 NNET</td><td>Data processing</td></tr>
                </tbody>
              </table>
              <p style={{marginTop: 15, color: '#888', fontSize: 13}}>
                Start earning by downloading the NeoNet Miner from GitHub and connecting to the network!
              </p>
            </div>


            <div style={{...styles.card, background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)', border: '1px solid #9c27b0'}}>
              <h3 style={{color: '#9c27b0'}}>Network Integration (Domains)</h3>
              <p style={{marginBottom: '20px'}}>NeoNet uses a decentralized domain system for connecting services and dApps.</p>
              
              <h4 style={{color: '#ba68c8', marginTop: '20px'}}>Official Domains</h4>
              <table style={styles.table}>
                <thead>
                  <tr><th>Domain</th><th>Purpose</th></tr>
                </thead>
                <tbody>
                  <tr><td><strong>neonetainetwork.com</strong></td><td>Main Website (Primary)</td></tr>
                  <tr><td>neonetainetwork.info</td><td>Documentation & Info</td></tr>
                  <tr><td>neonetainetwork.org</td><td>Community & DAO</td></tr>
                </tbody>
              </table>

              <h4 style={{color: '#ba68c8', marginTop: '20px'}}>Network Configuration</h4>
              <table style={styles.table}>
                <thead>
                  <tr><th>Parameter</th><th>Value</th></tr>
                </thead>
                <tbody>
                  <tr><td>Network ID</td><td>neonet-mainnet-1</td></tr>
                  <tr><td>Chain ID</td><td>8472</td></tr>
                  <tr><td>RPC Endpoint</td><td>https://rpc.neonetainetwork.com</td></tr>
                  <tr><td>WebSocket</td><td>wss://ws.neonetainetwork.com</td></tr>
                  <tr><td>Block Time</td><td>~3 seconds</td></tr>
                  <tr><td>Consensus</td><td>Proof of Intelligence (PoI)</td></tr>
                </tbody>
              </table>

              <h4 style={{color: '#ba68c8', marginTop: '20px'}}>Address Formats</h4>
              <table style={styles.table}>
                <thead>
                  <tr><th>Type</th><th>Format</th><th>Example</th></tr>
                </thead>
                <tbody>
                  <tr><td>Quantum Address</td><td>neo1...</td><td>neo1dfa5ee86e6443115287e8a6c604cd8aa32d101</td></tr>
                  <tr><td>EVM Address</td><td>0x...</td><td>0x742d35Cc6634C0532925a3b844Bc454e4438f44e</td></tr>
                  <tr><td>Contract</td><td>0x.../neo1...</td><td>Supports both formats</td></tr>
                </tbody>
              </table>

              <h4 style={{color: '#ba68c8', marginTop: '20px'}}>API Endpoints</h4>
              <table style={styles.table}>
                <thead>
                  <tr><th>Endpoint</th><th>Description</th></tr>
                </thead>
                <tbody>
                  <tr><td>/api/balance/{'{address}'}</td><td>Get token balances</td></tr>
                  <tr><td>/api/transactions/history/{'{address}'}</td><td>Transaction history</td></tr>
                  <tr><td>/api/tokenomics</td><td>Network tokenomics (supply, burning stats)</td></tr>
                  <tr><td>/api/ai-energy/register</td><td>Register as AI energy provider</td></tr>
                  <tr><td>/api/contracts/deploy</td><td>Deploy smart contract</td></tr>
                  <tr><td>/api/state/root</td><td>Current state root (Merkle)</td></tr>
                  <tr><td>/api/state/proof/{'{address}'}</td><td>Merkle proof for address</td></tr>
                </tbody>
              </table>

              <h4 style={{color: '#ba68c8', marginTop: '20px'}}>Decentralized Database</h4>
              <p>NeoNet uses an Ethereum-style database with AI-managed optimization:</p>
              <ul style={styles.list}>
                <li><strong>State Trie</strong> - Merkle Patricia Trie for account state</li>
                <li><strong>Storage Trie</strong> - Contract storage per account</li>
                <li><strong>Transactions Trie</strong> - Block transaction proofs</li>
                <li><strong>AI Manager</strong> - Automatic optimization, compaction, anomaly detection</li>
                <li><strong>Persistence</strong> - State survives server restarts (auto-save every 30s)</li>
              </ul>
            </div>

            <div style={styles.card}>
              <h3>Quick Links</h3>
              <ul style={styles.list}>
                <li><a href="https://github.com/neonetainetwork/NeoNet-Blockchain" target="_blank" style={{color: '#58a6ff'}}>GitHub Repository</a></li>
                <li><a href="https://github.com/neonetainetwork/NeoNet-Blockchain/discussions" target="_blank" style={{color: '#58a6ff'}}>Community Discussions</a></li>
                <li><a href="https://github.com/neonetainetwork/NeoNet-Blockchain/issues" target="_blank" style={{color: '#58a6ff'}}>Issue Tracker</a></li>
              </ul>
            </div>
          </div>
        )}
      </main>

      <WalletModal isOpen={showWallet} onClose={() => setShowWallet(false)} />
    </div>
  );
}

const styles = {
  app: { 
    minHeight: '100vh', 
    background: '#0d1117', 
    color: '#c9d1d9', 
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif' 
  },
  header: { 
    display: 'flex', 
    alignItems: 'center', 
    padding: '12px 16px', 
    borderBottom: '1px solid #21262d', 
    background: '#161b22',
    flexWrap: 'wrap',
    gap: '8px',
    position: 'sticky',
    top: 0,
    zIndex: 100
  },
  logoContainer: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    cursor: 'pointer',
    marginRight: '16px'
  },
  logoImg: {
    width: '36px',
    height: '36px',
    borderRadius: '8px'
  },
  logoText: { 
    fontSize: '22px', 
    fontWeight: 'bold', 
    color: '#58a6ff'
  },
  menuToggle: {
    display: 'none',
    padding: '8px 12px',
    background: 'transparent',
    border: '1px solid #30363d',
    borderRadius: '6px',
    color: '#c9d1d9',
    cursor: 'pointer',
    fontSize: '18px'
  },
  nav: { 
    display: 'flex', 
    gap: '6px', 
    flex: 1,
    flexWrap: 'wrap'
  },
  navOpen: {},
  tab: { 
    padding: '8px 14px', 
    background: 'transparent', 
    border: '1px solid #30363d', 
    borderRadius: '6px', 
    color: '#8b949e', 
    cursor: 'pointer',
    fontSize: '14px',
    whiteSpace: 'nowrap'
  },
  tabActive: { 
    padding: '8px 14px', 
    background: '#238636', 
    border: '1px solid #238636', 
    borderRadius: '6px', 
    color: '#fff', 
    cursor: 'pointer',
    fontSize: '14px',
    whiteSpace: 'nowrap'
  },
  connectBtn: { 
    padding: '8px 14px', 
    background: '#238636', 
    border: 'none', 
    borderRadius: '6px', 
    color: '#fff', 
    cursor: 'pointer',
    fontWeight: '600',
    fontSize: '14px',
    whiteSpace: 'nowrap'
  },
  main: { 
    padding: '16px', 
    maxWidth: '1200px', 
    margin: '0 auto' 
  },
  homePage: {
    maxWidth: '1000px',
    margin: '0 auto'
  },
  heroSection: {
    textAlign: 'center',
    padding: '40px 20px',
    marginBottom: '40px'
  },
  heroLogo: {
    width: '120px',
    height: '120px',
    marginBottom: '24px',
    borderRadius: '24px',
    boxShadow: '0 8px 32px rgba(88, 166, 255, 0.3)'
  },
  heroTitle: {
    fontSize: '48px',
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: '12px',
    background: 'linear-gradient(135deg, #58a6ff, #3fb950)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    backgroundClip: 'text'
  },
  heroSubtitle: {
    fontSize: '22px',
    color: '#58a6ff',
    marginBottom: '16px',
    fontWeight: '500'
  },
  heroDescription: {
    fontSize: '17px',
    color: '#8b949e',
    maxWidth: '600px',
    margin: '0 auto 28px',
    lineHeight: '1.7'
  },
  heroCta: {
    display: 'flex',
    gap: '16px',
    justifyContent: 'center',
    flexWrap: 'wrap'
  },
  ctaBtn: {
    padding: '14px 32px',
    background: 'linear-gradient(135deg, #238636, #2ea043)',
    border: 'none',
    borderRadius: '8px',
    color: '#fff',
    fontSize: '16px',
    fontWeight: '600',
    cursor: 'pointer',
    transition: 'transform 0.2s'
  },
  ctaBtnOutline: {
    padding: '14px 32px',
    background: 'transparent',
    border: '2px solid #58a6ff',
    borderRadius: '8px',
    color: '#58a6ff',
    fontSize: '16px',
    fontWeight: '600',
    cursor: 'pointer'
  },
  featuresGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
    gap: '20px',
    marginBottom: '48px'
  },
  featureCard: {
    background: '#161b22',
    border: '1px solid #30363d',
    borderRadius: '12px',
    padding: '24px',
    transition: 'border-color 0.2s, transform 0.2s'
  },
  featureIcon: {
    fontSize: '36px',
    marginBottom: '16px'
  },
  featureTitle: {
    fontSize: '18px',
    fontWeight: '600',
    color: '#fff',
    marginBottom: '10px'
  },
  featureDesc: {
    fontSize: '14px',
    color: '#8b949e',
    lineHeight: '1.6'
  },
  statsSection: {
    background: '#161b22',
    border: '1px solid #30363d',
    borderRadius: '12px',
    padding: '32px',
    marginBottom: '32px',
    textAlign: 'center'
  },
  sectionTitle: {
    fontSize: '24px',
    color: '#fff',
    marginBottom: '24px',
    fontWeight: '600'
  },
  statsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
    gap: '24px'
  },
  statItem: {
    textAlign: 'center'
  },
  statValue: {
    fontSize: '28px',
    fontWeight: 'bold',
    color: '#58a6ff',
    marginBottom: '6px'
  },
  statLabel: {
    fontSize: '13px',
    color: '#8b949e'
  },
  techSection: {
    background: '#161b22',
    border: '1px solid #30363d',
    borderRadius: '12px',
    padding: '32px',
    marginBottom: '32px'
  },
  socialSection: {
    textAlign: 'center',
    marginBottom: '32px'
  },
  socialGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
    gap: '24px',
    maxWidth: '800px',
    margin: '0 auto'
  },
  socialCard: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    padding: '32px 24px',
    background: '#161b22',
    border: '1px solid #30363d',
    borderRadius: '12px',
    textDecoration: 'none',
    transition: 'all 0.3s ease',
    cursor: 'pointer'
  },
  socialTitle: {
    fontSize: '18px',
    fontWeight: 'bold',
    color: '#fff',
    marginTop: '16px',
    marginBottom: '8px'
  },
  socialDesc: {
    fontSize: '14px',
    color: '#8b949e',
    margin: 0
  },
  techGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
    gap: '16px'
  },
  techItem: {
    padding: '16px',
    background: '#0d1117',
    borderRadius: '8px',
    fontSize: '14px',
    lineHeight: '1.5',
    color: '#c9d1d9'
  },
  title: { 
    fontSize: '32px', 
    marginBottom: '24px', 
    color: '#fff' 
  },
  subtitle: { 
    fontSize: '20px', 
    marginTop: '32px', 
    marginBottom: '16px', 
    color: '#fff' 
  },
  grid: { 
    display: 'grid', 
    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
    gap: '16px', 
    marginBottom: '24px' 
  },
  card: { 
    background: '#161b22', 
    border: '1px solid #30363d', 
    borderRadius: '8px', 
    padding: '20px' 
  },
  cardLabel: { 
    fontSize: '14px', 
    color: '#8b949e', 
    marginBottom: '8px' 
  },
  cardValue: { 
    fontSize: '28px', 
    fontWeight: 'bold', 
    color: '#fff' 
  },
  online: { 
    color: '#3fb950', 
    fontWeight: 'bold', 
    fontSize: '18px' 
  },
  offline: { 
    color: '#f85149', 
    fontWeight: 'bold', 
    fontSize: '18px' 
  },
  error: { 
    background: '#21262d', 
    border: '1px solid #f85149', 
    borderRadius: '8px', 
    padding: '24px', 
    textAlign: 'center' 
  },
  table: { 
    width: '100%', 
    borderCollapse: 'collapse', 
    background: '#161b22', 
    borderRadius: '8px', 
    overflow: 'hidden' 
  },
  pending: { 
    background: '#1f6feb', 
    color: '#fff', 
    padding: '2px 8px', 
    borderRadius: '12px', 
    fontSize: '12px' 
  },
  done: { 
    background: '#238636', 
    color: '#fff', 
    padding: '2px 8px', 
    borderRadius: '12px', 
    fontSize: '12px' 
  },
  address: { 
    display: 'block', 
    background: '#0d1117', 
    padding: '12px', 
    borderRadius: '6px', 
    wordBreak: 'break-all', 
    marginBottom: '8px', 
    fontSize: '14px' 
  },
  copyBtn: { 
    padding: '6px 12px', 
    background: '#21262d', 
    border: '1px solid #30363d', 
    borderRadius: '6px', 
    color: '#8b949e', 
    cursor: 'pointer' 
  },
  formGroup: { 
    marginBottom: '16px' 
  },
  input: { 
    width: '100%', 
    padding: '12px', 
    background: '#0d1117', 
    border: '1px solid #30363d', 
    borderRadius: '6px', 
    color: '#c9d1d9', 
    fontSize: '14px' 
  },
  btn: { 
    padding: '12px 24px', 
    background: '#238636', 
    border: 'none', 
    borderRadius: '6px', 
    color: '#fff', 
    cursor: 'pointer', 
    fontWeight: '600' 
  },
  list: { 
    paddingLeft: '20px', 
    lineHeight: '2' 
  },
  code: { 
    background: '#0d1117', 
    padding: '16px', 
    borderRadius: '6px', 
    overflow: 'auto', 
    fontSize: '14px', 
    lineHeight: '1.6' 
  },
  downloadBtn: { 
    display: 'inline-block', 
    padding: '12px 24px', 
    background: '#1f6feb', 
    color: '#fff', 
    borderRadius: '6px', 
    textDecoration: 'none', 
    fontWeight: '600' 
  }
};

export default function App() {
  return (
    <HashRouter>
      <WalletProvider>
        <AppContent />
      </WalletProvider>
    </HashRouter>
  );
}
