# BGI Trident - Autonomous Consumption Agent for Swiggy

> Behavioral Graph Intelligence meets Swiggy's MCP platform.
> Three prediction prongs. One execution engine. Zero user friction.

## What This Is

An autonomous consumption agent powered by a **three-prong ensemble predictor** (Trident) that models user behavior across Swiggy's three domains (Food, Instamart, Dineout) and executes cross-domain orders through a multi-agent orchestration framework.

Unlike recommendation systems that suggest and wait, Trident **predicts, decides, and executes** -- closing the loop from behavioral signal to completed transaction.

## Why "Trident" -- Three Prongs of Prediction

The name is literal. Each prong captures a different signal class, and the ensemble combines them for predictions that no single model achieves alone.

```
                         TRIDENT ENSEMBLE
                              |
              +---------------+---------------+
              |               |               |
        Prong 1 (PyG)   Prong 2 (DGL)   Prong 3 (XGBoost)
        Graph Structure  Graph Dynamics   Tabular Features
              |               |               |
        ID-GNN on        R-GCN on         Gradient-boosted
        heterogeneous    temporal graph    hand-engineered
        consumption      with recency     consumption
        graph            decay weights    features
              |               |               |
        User-item        Time-aware        Frequency,
        structural       behavioral        recency, AOV,
        embeddings       embeddings        day-of-week,
        (128d)           (128d)            basket patterns
              |               |               |
              +---------------+---------------+
                              |
                    Ensemble Meta-Learner
                    (Stacked generalization)
                              |
                    Final prediction scores:
                    P(order from restaurant X) = 0.94
                    P(purchase product Y)      = 0.87
                    P(book venue Z)            = 0.72
```

### Why three prongs, not just one GNN?

**Prong 1: PyG (structural embeddings).** PyTorch Geometric with ID-GNN on the heterogeneous consumption graph. Captures *who orders what from where* -- the topological structure of user-restaurant-product-venue relationships. Strong at cold-start (new restaurants get embeddings from cuisine/location neighbors) and cross-domain discovery (OFTEN_PAIRED edges between Food and Instamart).

**Prong 2: DGL (temporal-behavioral embeddings).** Deep Graph Library with R-GCN on a temporal variant of the same graph, where edge weights decay with recency and edges carry time-of-day/day-of-week features. Captures *when and how often* -- the behavioral dynamics that PyG's static structure misses. A user who ordered biryani every Thursday for 6 weeks but stopped 3 weeks ago looks very different in DGL (decaying edge weight) than in PyG (strong structural connection). DGL catches the drift.

**Prong 3: XGBoost (tabular features).** Gradient-boosted trees on hand-engineered consumption features: order frequency, recency, average order value, basket size distribution, day-of-week patterns, cuisine diversity index, reorder intervals for Instamart SKUs. XGBoost handles the signals that GNNs are not great at: sharp thresholds (user never orders after 10 PM), multiplicative feature interactions (high AOV + weekend + group size > 3 = dineout signal), and tabular patterns that message-passing architectures smooth over.

**Ensemble.** A stacked generalization meta-learner (logistic regression or lightweight MLP) takes the prediction scores from all three prongs and produces final calibrated probabilities. The ensemble consistently outperforms any single prong because each captures orthogonal signal: structure (PyG), dynamics (DGL), and engineered features (XGBoost).

### How It Differs from Prior Art

DoorDash uses a single ID-GNN for notification personalization (heterogeneous user-restaurant graph, 7-day link prediction). Their system outputs a push notification. Single domain, single model, passive output.

Trident differs in three ways: (1) three-model ensemble captures structural, temporal, and tabular signals that a single GNN misses, (2) cross-domain heterogeneous graph spans Food, Instamart, and Dineout with 8 edge types, and (3) predictions drive autonomous execution through multi-agent orchestration, not just notifications.

## Provenance -- This Didn't Start Here

BGI Trident is not a weekend project built for this application. It is a Swiggy-specific integration of three platform layers developed at [AhinsaAI](https://ahinsaai.com) over the past 18+ months:

**BGI (Behavioral Graph Intelligence)** is the core research layer. BGI models user behavior as heterogeneous graphs and applies GNN-based prediction to domains where understanding consumption patterns drives business outcomes. The initial BGI work was in fraud detection -- modeling transaction networks as graphs to identify anomalous patterns, synthetic identities, and collusion rings in payments systems. The graph schema, feature engineering approach, and ensemble methodology in this repo descend directly from that production fraud graph work. The insight that a three-prong ensemble (structural + temporal + tabular) outperforms any single GNN was first validated in the fraud detection context, where PyG caught topological fraud rings, DGL caught velocity-based pattern shifts, and XGBoost caught threshold-based anomalies that message-passing architectures smoothed over.

**Graph Engine** is the graph engine layer that operationalizes BGI. It handles heterogeneous graph construction from raw interaction data, manages the training and serving pipeline for all three prongs, runs the ensemble meta-learner, and provides the online graph update mechanism that closes the predict-execute-learn loop. In this repo, everything under `src/bgi_trident/graph/` is the graph engine adapted for consumption prediction.

**Swar** is the vernacular voice AI infrastructure that AhinsaAI built for BFSI clients. Swar handles ASR/TTS provider abstraction, vernacular NLU for Indian languages including code-mixed input (Kannada-English, Hindi-English, Tamil-English), intent classification, and entity extraction. It has served regulated enterprise clients in production. The voice and NLU modules in this repo (`src/bgi_trident/voice/`, `src/bgi_trident/nlu/`) are integration points with Swar, not standalone reimplementations.

**Trident** is the orchestration layer that ties BGI predictions to multi-agent execution. The Trident coordinator, confirmation gates, fallback engine, and session state manager were designed as domain-agnostic components. Swiggy's three MCP servers are the first domain integration. The same Trident orchestrator can plug into any multi-service platform (food delivery, fintech, logistics) where cross-domain behavioral prediction drives autonomous action.

The repo you're reading is where all four layers converge on a single use case: predicting and executing consumption behavior across Swiggy's Food, Instamart, and Dineout.

## Architecture

```
                    +---------------------+
                    |   Input Modalities  |
                    |  Voice | Text | API |
                    +----------+----------+
                               |
                    +----------v----------+
                    | Voice Provider      |
                    | Abstraction Layer   |
                    | (Vapi/Bolna/Retell) |
                    +----------+----------+
                               |
                    +----------v----------+
                    |   Swar NLU Engine   |
                    | Vernacular Intent   |
                    | + Entity Extraction |
                    +----------+----------+
                               |
              +----------------v-----------------+
              |         BGI Trident Core          |
              |                                   |
              |  +--------Graph  ENGINE --------+ |
              |  |                               | |
              |  |  +-------+ +------+ +------+ | |
              |  |  | PyG   | | DGL  | |XGBst | | |
              |  |  |ID-GNN | |R-GCN | |Boost | | |
              |  |  |struct | |temprl| |tablr | | |
              |  |  +---+---+ +--+---+ +--+---+ | |
              |  |      |        |        |      | |
              |  |      +--------+--------+      | |
              |  |               |               | |
              |  |    Ensemble Meta-Learner      | |
              |  +---------------+---------------+ |
              |                  |                  |
              |  +---------------v---------------+  |
              |  |    Trident Orchestrator        |  |
              |  |  Multi-Agent Coordinator       |  |
              |  |  Constraint Resolution         |  |
              |  |  Confirmation Gates            |  |
              |  +-------------------------------+  |
              |                                     |
              +---+----------+----------+-----------+
                  |          |          |
           +------v--+ +----v----+ +---v-------+
           |  Food   | |Instamart| |  Dineout  |
           |  Agent  | |  Agent  | |   Agent   |
           +---------+ +---------+ +-----------+
                  |          |          |
           +------v--+ +----v----+ +---v-------+
           |Food MCP | |Instamart| |Dineout MCP|
           | Server  | |MCP Srvr | |  Server   |
           +---------+ +---------+ +-----------+
```

## Repo Structure

```
trident-consumption-graph/
|
+-- README.md
+-- pyproject.toml
+-- .github/
|   +-- workflows/
|       +-- ci.yml                    # Ruff lint + pytest + type check
|
+-- docs/
|   +-- architecture.md              # Detailed system design
|   +-- graph-schema.md              # Node/edge type definitions
|   +-- doordash-comparison.md       # Prior art analysis
|   +-- demo-script.md               # 2-min demo walkthrough
|   +-- trident-ensemble.md          # Why three prongs, ablation results
|
+-- src/
|   +-- bgi_trident/
|   |   +-- __init__.py
|   |   |
|   |   +-- graph/                   # Kumo: Behavioral Graph Engine
|   |   |   +-- __init__.py
|   |   |   +-- schema.py            # Node types, edge types, feature defs
|   |   |   +-- builder.py           # Heterogeneous graph construction (shared)
|   |   |   +-- updater.py           # Online graph update from completed orders
|   |   |   |
|   |   |   +-- pyg/                 # Prong 1: Structural embeddings
|   |   |   |   +-- __init__.py
|   |   |   |   +-- model.py         # ID-GNN encoder on HeteroData
|   |   |   |   +-- embeddings.py    # User/restaurant/product/venue embeddings (128d)
|   |   |   |   +-- link_predict.py  # Structural link prediction head
|   |   |   |
|   |   |   +-- dgl/                 # Prong 2: Temporal-behavioral embeddings
|   |   |   |   +-- __init__.py
|   |   |   |   +-- model.py         # R-GCN on DGLHeteroGraph
|   |   |   |   +-- temporal.py      # Recency decay weights + time-of-day attention
|   |   |   |   +-- embeddings.py    # Time-aware behavioral embeddings (128d)
|   |   |   |   +-- link_predict.py  # Temporal link prediction head
|   |   |   |
|   |   |   +-- xgboost/             # Prong 3: Tabular feature engineering
|   |   |   |   +-- __init__.py
|   |   |   |   +-- features.py      # Hand-engineered consumption features
|   |   |   |   +-- model.py         # XGBoost classifier (frequency, recency, AOV, etc.)
|   |   |   |
|   |   |   +-- ensemble/            # Meta-learner: stacked generalization
|   |   |       +-- __init__.py
|   |   |       +-- stacker.py       # Combines PyG + DGL + XGBoost scores
|   |   |       +-- calibration.py   # Probability calibration (Platt scaling)
|   |   |       +-- ablation.py      # Single-prong vs ensemble comparison
|   |   |
|   |   +-- orchestrator/            # Trident: Multi-Agent Orchestration
|   |   |   +-- __init__.py
|   |   |   +-- coordinator.py       # Intent decomposition + parallel dispatch
|   |   |   +-- state.py             # Cross-agent session state manager
|   |   |   +-- confirmation.py      # Payment/order confirmation gates
|   |   |   +-- fallback.py          # Graceful degradation + alternatives
|   |   |
|   |   +-- agents/                  # Domain Agents (one per MCP server)
|   |   |   +-- __init__.py
|   |   |   +-- base.py              # BaseAgent protocol
|   |   |   +-- food.py              # Food agent: search, menu, cart, order, track
|   |   |   +-- instamart.py         # Instamart agent: search, cart, checkout, track
|   |   |   +-- dineout.py           # Dineout agent: search, details, slots, book
|   |   |
|   |   +-- mcp/                     # MCP Server Interfaces
|   |   |   +-- __init__.py
|   |   |   +-- protocol.py          # MCPServer protocol (connect, call_tool, close)
|   |   |   +-- food_server.py       # Swiggy Food MCP client
|   |   |   +-- instamart_server.py  # Swiggy Instamart MCP client
|   |   |   +-- dineout_server.py    # Swiggy Dineout MCP client
|   |   |   +-- mock/                # Mock implementations for demo
|   |   |       +-- __init__.py
|   |   |       +-- food_mock.py     # Bangalore restaurant + menu data
|   |   |       +-- instamart_mock.py# Grocery product catalog
|   |   |       +-- dineout_mock.py  # Dineout venue + slot data
|   |   |       +-- fixtures/
|   |   |           +-- restaurants.json
|   |   |           +-- menus.json
|   |   |           +-- products.json
|   |   |           +-- venues.json
|   |   |
|   |   +-- voice/                   # Voice Provider Abstraction
|   |   |   +-- __init__.py
|   |   |   +-- protocol.py          # VoiceProvider protocol (5 methods)
|   |   |   +-- adapters/
|   |   |       +-- __init__.py
|   |   |       +-- vapi.py          # Vapi WebSocket adapter
|   |   |       +-- bolna.py         # Bolna adapter (stub)
|   |   |       +-- retell.py        # Retell adapter (stub)
|   |   |       +-- eleven.py        # ElevenLabs TTS + Deepgram ASR (stub)
|   |   |
|   |   +-- nlu/                     # Vernacular NLU (Swar integration point)
|   |   |   +-- __init__.py
|   |   |   +-- intent.py            # Intent classifier (order, restock, book, multi)
|   |   |   +-- entities.py          # Entity extractor (restaurant, product, venue, qty)
|   |   |   +-- codemix.py           # Code-mixed language handler (Hi-En, Ka-En, Ta-En)
|   |   |
|   |   +-- config.py                # Provider selection, API keys, feature flags
|   |
|   +-- data/                        # Synthetic training data
|       +-- generate_graph.py        # Synthetic Bangalore consumption graph generator
|       +-- bangalore_users.csv      # 500 synthetic user profiles
|       +-- bangalore_restaurants.csv# Real Bangalore restaurants (public data)
|       +-- bangalore_products.csv   # Common Instamart SKUs
|       +-- bangalore_venues.csv     # Dineout venues
|       +-- interactions/
|           +-- food_orders.csv      # 50K synthetic food orders (6 months)
|           +-- instamart_orders.csv # 20K synthetic grocery orders
|           +-- dineout_bookings.csv # 5K synthetic table bookings
|
+-- notebooks/
|   +-- 01_graph_construction.ipynb  # Build heterogeneous graph from synthetic data
|   +-- 02_pyg_training.ipynb        # Train PyG ID-GNN (Prong 1)
|   +-- 03_dgl_training.ipynb        # Train DGL R-GCN with temporal decay (Prong 2)
|   +-- 04_xgboost_features.ipynb   # Feature engineering + XGBoost training (Prong 3)
|   +-- 05_ensemble_ablation.ipynb   # Ensemble vs single-prong comparison
|   +-- 06_cross_domain_demo.ipynb   # End-to-end prediction + execution demo
|
+-- tests/
|   +-- conftest.py
|   +-- test_graph_builder.py
|   +-- test_pyg_model.py
|   +-- test_dgl_model.py
|   +-- test_xgboost_features.py
|   +-- test_ensemble.py
|   +-- test_orchestrator.py
|   +-- test_agents.py
|   +-- test_confirmation_gate.py
|   +-- test_fallback.py
|   +-- test_voice_protocol.py
|
+-- demo/
    +-- app.py                       # FastAPI demo server
    +-- ui/                          # React demo UI
    |   +-- App.jsx                  # Voice interaction + live graph visualization
    +-- scenarios/
        +-- thursday_biryani.json    # "Order usual Thursday dinner"
        +-- weekend_restock.json     # "Restock groceries + book Saturday table"
        +-- kannada_multiorder.json  # Full Kannada voice session across 3 servers
```

## Graph Schema

### Node Types

| Node Type   | Features                                           | Source        |
|-------------|-----------------------------------------------------|---------------|
| User        | location, cuisine_prefs, price_sensitivity, time_patterns | Profile + learned |
| Restaurant  | cuisine, rating, price_range, avg_delivery_time, location  | Food MCP      |
| Product     | category, brand, unit_size, price, reorder_frequency       | Instamart MCP |
| Venue       | cuisine, ambiance, price_range, capacity, location         | Dineout MCP   |
| TimeSlot    | hour_of_day, day_of_week, is_weekend, is_holiday           | Temporal       |
| Location    | geohash, area_name, city                                    | Geolocation    |

### Edge Types (Cross-Domain)

| Edge Type           | Source Node | Target Node | Weight Signal                     |
|---------------------|-------------|-------------|-----------------------------------|
| ORDERED_FROM        | User        | Restaurant  | order_count, recency, avg_spend   |
| PURCHASED           | User        | Product     | purchase_count, interval_days     |
| BOOKED_AT           | User        | Venue       | booking_count, party_size, rating |
| PREFERS_AT_TIME     | User        | TimeSlot    | order_frequency per slot          |
| LOCATED_NEAR        | User        | Location    | delivery_address frequency        |
| SERVES_CUISINE      | Restaurant  | Restaurant  | shared cuisine (implicit edge)    |
| OFTEN_PAIRED        | Restaurant  | Product     | co-occurrence in same-day orders  |
| FOLLOWED_BY_DINING  | Restaurant  | Venue       | food order within 48h of booking  |

The OFTEN_PAIRED and FOLLOWED_BY_DINING edges are what make this cross-domain. DoorDash's graph has no equivalent -- their heterogeneous graph is single-domain (users and restaurants only). These cross-domain edges enable predictions like: "User ordered from Meghana's (Food) and is likely to need Coke (Instamart) and book a similar restaurant (Dineout) this weekend."

## Key Design Decisions

### 1. Mock-to-Real MCP Swap

```python
# config.py
MCP_MODE = os.getenv("MCP_MODE", "mock")  # "mock" or "live"

# Each MCP server is instantiated via factory
def get_food_server() -> MCPServer:
    if MCP_MODE == "live":
        return SwiggyFoodMCP(credentials=SWIGGY_CREDS)
    return MockFoodMCP(fixtures_path="data/fixtures/")
```

When Swiggy grants API access, flip `MCP_MODE=live`. Zero code changes.

### 2. Voice Provider Independence

```python
class VoiceProvider(Protocol):
    async def start_session(self, config: SessionConfig) -> str: ...
    async def stream_audio_in(self, session_id: str, audio: bytes) -> None: ...
    async def receive_transcript(self, session_id: str) -> Transcript: ...
    async def synthesize_speech(self, text: str, language: str) -> bytes: ...
    async def end_session(self, session_id: str) -> None: ...
```

Swap providers via config. Orchestration layer imports `VoiceProvider`, never `vapi` or `retell`.

### 3. Confirmation Gate (Fraud/Risk Pattern)

No transactional MCP call (`place_food_order`, `checkout`, `book_table`) fires without passing through the confirmation gate. The gate:
- Reads back the order summary in the user's language
- States the total amount
- Requires explicit "yes"/"haan"/"sari" confirmation
- Logs the confirmation event for audit

This is a payments-grade guardrail, not a UX checkbox.

## Running the Demo

```bash
# Install dependencies
pip install -e ".[dev]"

# Generate synthetic graph data
python src/data/generate_graph.py

# Train all three prongs
python -m bgi_trident.graph.pyg.model --train          # Prong 1: PyG ID-GNN
python -m bgi_trident.graph.dgl.model --train           # Prong 2: DGL R-GCN
python -m bgi_trident.graph.xgboost.model --train       # Prong 3: XGBoost

# Fit ensemble meta-learner
python -m bgi_trident.graph.ensemble.stacker --fit

# Run ablation (optional -- shows ensemble vs single-prong lift)
python -m bgi_trident.graph.ensemble.ablation

# Start demo server (mock MCP mode)
MCP_MODE=mock python demo/app.py

# Run with live Swiggy APIs (requires credentials)
MCP_MODE=live SWIGGY_API_KEY=xxx python demo/app.py
```

## Tech Stack

- **Prong 1 (Structural)**: PyTorch Geometric (PyG) -- ID-GNN on HeteroData, heterogeneous message passing
- **Prong 2 (Temporal)**: Deep Graph Library (DGL) -- R-GCN on DGLHeteroGraph, recency-decayed edge weights
- **Prong 3 (Tabular)**: XGBoost -- gradient-boosted trees on engineered consumption features
- **Ensemble**: Stacked generalization meta-learner with Platt-scaled probability calibration
- **Orchestration**: asyncio-based multi-agent coordinator
- **Voice**: Provider-abstracted (Vapi default, Bolna/Retell/ElevenLabs swappable)
- **NLU**: Multilingual intent + entity extraction (Kannada, Hindi, Tamil, English)
- **API**: FastAPI for demo server
- **CI**: GitHub Actions (ruff, pytest, mypy)

## License

MIT

## Author

**Abhinav Khare** -- Cofounder and CTO, [AhinsaAI](https://ahinsaai.com)

20+ years in payments infrastructure, fraud/risk systems, and voice AI. Two exits. Production graph network experience in fraud detection. Built and scaled GCCs from 0 to 500 FTE. Scaled Asianet News Digital from 3M to 75M MAUs. ETH Zurich (M.S. Engineering), London Business School (MBA Finance). 12 active board seats across fintech, banking, and defence. Based in Bangalore.

- [LinkedIn](https://linkedin.com/in/abhinavkhare)
- [GitHub](https://github.com/AbhinavKhareTech)
