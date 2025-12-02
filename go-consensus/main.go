package main

import (
	"flag"
	"log"
	"net/http"
)

func main() {
	port := flag.String("port", "50051", "P2P listen port")
	peers := flag.String("peers", "", "comma-separated peer addresses (host:port)")
	httpPort := flag.String("http", "8080", "HTTP API port")
	flag.Parse()

	cfg := Config{
		Port:  *port,
		Peers: *peers,
	}
	node := NewNode(cfg)
	log.Printf("=== NeoNet Go Consensus Starting ===")
	log.Printf("P2P Port: %s", cfg.Port)
	log.Printf("HTTP API Port: %s", *httpPort)

	if err := node.LoadChainFromFile("chain_store.json"); err != nil {
		log.Printf("No persisted chain loaded: %v", err)
	}

	go func() {
		mux := http.NewServeMux()
		mux.HandleFunc("/chain", node.HandleGetChain)
		mux.HandleFunc("/peers", node.HandleGetPeers)
		mux.HandleFunc("/tx", node.HandlePostTx)
		mux.HandleFunc("/vote", node.HandleVote)
		mux.HandleFunc("/relay_execute", node.HandleRelayExecute)
		mux.HandleFunc("/pbft/preprepare", node.HandlePBFTPrePrepare)
		mux.HandleFunc("/pbft/prepare", node.HandlePBFTPrepare)
		mux.HandleFunc("/pbft/commit", node.HandlePBFTCommit)
		log.Printf("HTTP API listening on :%s", *httpPort)
		log.Fatal(http.ListenAndServe(":"+*httpPort, mux))
	}()

	node.Start()
}
