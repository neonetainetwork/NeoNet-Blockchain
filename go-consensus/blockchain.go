package main

import (
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"sync"
	"time"
)

type Block struct {
	Index     int    `json:"index"`
	Timestamp string `json:"timestamp"`
	Data      string `json:"data"`
	PrevHash  string `json:"prev_hash"`
	Hash      string `json:"hash"`
	Nonce     int    `json:"nonce"`
	PubKey    string `json:"pub_key"`
	Signature string `json:"signature"`
}

type Blockchain struct {
	chain []*Block
	mu    sync.RWMutex
}

func NewBlockchain() *Blockchain {
	return &Blockchain{chain: make([]*Block, 0)}
}

func (bc *Blockchain) CreateGenesis() {
	gen := &Block{
		Index:     0,
		Timestamp: time.Now().UTC().Format(time.RFC3339),
		Data:      "genesis",
		PrevHash:  "",
		Nonce:     0,
	}
	gen.Hash = calculateHash(gen)
	bc.chain = append(bc.chain, gen)
}

func (bc *Blockchain) Latest() *Block {
	bc.mu.RLock()
	defer bc.mu.RUnlock()
	if len(bc.chain) == 0 {
		return nil
	}
	return bc.chain[len(bc.chain)-1]
}

func calculateHash(b *Block) string {
	record := fmt.Sprintf("%d%s%s%s%d", b.Index, b.Timestamp, b.Data, b.PrevHash, b.Nonce)
	h := sha256.Sum256([]byte(record))
	return hex.EncodeToString(h[:])
}

const difficulty = 2

func mine(b *Block) {
	target := ""
	for i := 0; i < difficulty; i++ {
		target += "0"
	}
	for {
		b.Nonce++
		h := calculateHash(b)
		if len(h) >= difficulty && h[:difficulty] == target {
			b.Hash = h
			return
		}
	}
}

func (bc *Blockchain) GenerateBlock(data string) *Block {
	latest := bc.Latest()
	var idx int
	var prev string
	if latest == nil {
		idx = 1
		prev = ""
	} else {
		idx = latest.Index + 1
		prev = latest.Hash
	}
	b := &Block{
		Index:     idx,
		Timestamp: time.Now().UTC().Format(time.RFC3339),
		Data:      data,
		PrevHash:  prev,
		Nonce:     0,
	}
	mine(b)
	return b
}

func (bc *Blockchain) AddBlock(b *Block) bool {
	bc.mu.Lock()
	defer bc.mu.Unlock()
	latest := bc.Latest()
	if latest != nil {
		if b.Index != latest.Index+1 {
			return false
		}
		if b.PrevHash != latest.Hash {
			return false
		}
		if calculateHash(b) != b.Hash {
			return false
		}
	}
	bc.chain = append(bc.chain, b)
	return true
}

func (bc *Blockchain) ReplaceChain(newChain []*Block) bool {
	bc.mu.Lock()
	defer bc.mu.Unlock()
	if len(newChain) <= len(bc.chain) {
		return false
	}
	for i := 1; i < len(newChain); i++ {
		prev := newChain[i-1]
		curr := newChain[i]
		if curr.PrevHash != prev.Hash || calculateHash(curr) != curr.Hash {
			return false
		}
	}
	bc.chain = newChain
	return true
}

func (bc *Blockchain) SaveToFile(fn string) error {
	bc.mu.RLock()
	defer bc.mu.RUnlock()
	bts, err := json.MarshalIndent(bc.chain, "", "  ")
	if err != nil {
		return err
	}
	tmp := fn + ".tmp"
	if err := os.WriteFile(tmp, bts, 0644); err != nil {
		return err
	}
	return os.Rename(tmp, fn)
}

func (bc *Blockchain) LoadFromFile(fn string) error {
	bc.mu.Lock()
	defer bc.mu.Unlock()
	bts, err := os.ReadFile(fn)
	if err != nil {
		return err
	}
	var arr []*Block
	if err := json.Unmarshal(bts, &arr); err != nil {
		return err
	}
	if len(arr) == 0 {
		return fmt.Errorf("empty chain")
	}
	for i := 1; i < len(arr); i++ {
		prev := arr[i-1]
		curr := arr[i]
		if curr.PrevHash != prev.Hash || calculateHash(curr) != curr.Hash {
			return fmt.Errorf("invalid chain at index %d", i)
		}
	}
	bc.chain = arr
	return nil
}

func SignBlock(b *Block, privHex string) error {
	priv, err := hex.DecodeString(privHex)
	if err != nil {
		return err
	}
	if len(priv) != ed25519.PrivateKeySize {
		return fmt.Errorf("bad priv size")
	}
	msg := calculateHash(b)
	sig := ed25519.Sign(ed25519.PrivateKey(priv), []byte(msg))
	b.Signature = hex.EncodeToString(sig)
	pub := ed25519.PrivateKey(priv).Public().(ed25519.PublicKey)
	b.PubKey = hex.EncodeToString(pub)
	return nil
}

func VerifyBlockSignature(b *Block) bool {
	if b.Signature == "" || b.PubKey == "" {
		return true // Allow unsigned blocks
	}
	pub, err := hex.DecodeString(b.PubKey)
	if err != nil {
		return false
	}
	sig, err := hex.DecodeString(b.Signature)
	if err != nil {
		return false
	}
	msg := calculateHash(b)
	return ed25519.Verify(ed25519.PublicKey(pub), []byte(msg), sig)
}
