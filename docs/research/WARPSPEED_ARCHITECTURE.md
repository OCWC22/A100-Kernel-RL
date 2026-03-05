# WarpSpeed Architecture

**Source:** doubleAI WarpSpeed Technical Blog (March 3, 2026)

**GitHub:** https://github.com/double-ai/doubleGraph

---

## Executive Summary

WarpSpeed is doubleAI's AI system for GPU performance engineering, demonstrated by generating a hyper-optimized drop-in version of NVIDIA cuGraph:

**Key Results:**
- **3.6× average speedup** over expert-tuned cuGraph
- **100% of algorithms** run faster
- **55% achieve >2×** improvement
- **18% achieve >10×** improvement
- **192 kernels per GPU** architecture (576 total across A100, L4, A10G)

---

## Part 1: Core Philosophy

### 1.1 Artificial Expert Intelligence (AEI)

WarpSpeed represents a new paradigm: **Artificial Expert Intelligence**

| Characteristic | Traditional AI | Expert AI |
|---------------|----------------|-----------|
| Data Requirements | Abundant training data | Works with scarce data |
| Verification | Easy-to-check solutions | Hard-to-verify outputs |
| Reasoning | Short chains | Deep reasoning chains |
| Domain | General | Specialized |

**GPU Performance Engineering challenges all three:**
- **Scarce data:** Few CUDA experts, limited training examples
- **Hard verification:** Correctness is non-trivial, performance measurement is subtle
- **Deep reasoning:** Optimal performance emerges from long chains of interacting decisions

### 1.2 Scale + Skill

WarpSpeed targets two axes:

1. **Skill:** Finding optimizations that even the best engineers have missed
2. **Scale:** Applying them exhaustively across every algorithm and hardware target

**The combinatorial explosion:**

```
Algorithms × Configurations × GPUs = Kernels Required

cuGraph: ~24 algorithms × ~8 configurations × 3 GPUs = 576 kernels
```

**No human team would undertake this level of specialization.**

---

## Part 2: System Architecture

### 2.1 High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      WarpSpeed System                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │
│  │ Diligent     │   │ PAC          │   │ Meta-Pattern │        │
│  │ Learning     │   │ Reasoning    │   │ Engine       │        │
│  │              │   │              │   │              │        │
│  │ • Idea space │   │ • Correctness│   │ • 4-way      │        │
│  │   search     │   │ • Validation │   │   dispatch   │        │
│  │ • Efficient  │   │ • Grounding  │   │ • GPU-spec   │        │
│  │   exploration│   │              │   │   templates  │        │
│  └──────────────┘   └──────────────┘   └──────────────┘        │
│         │                  │                  │                 │
│         └──────────────────┼──────────────────┘                 │
│                            ▼                                    │
│                   ┌──────────────┐                              │
│                   │ Kernel       │                              │
│                   │ Generator    │                              │
│                   │              │                              │
│                   │ • Per-GPU    │                              │
│                   │   variants   │                              │
│                   │ • Optimized  │                              │
│                   │   builds     │                              │
│                   └──────────────┘                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Three Pillars

| Pillar | Purpose | Key Innovation |
|--------|---------|----------------|
| Diligent Learning | Search idea space efficiently | Navigate optimization decisions |
| PAC Reasoning | Determine correctness | Grounded verification |
| Meta-Pattern Engine | Generate optimized kernels | 4-way dispatch + GPU specialization |

---

## Part 3: Diligent Learning

### 3.1 The Idea Space

GPU kernel optimization involves a vast decision space:

```
┌─────────────────────────────────────────────────────────────┐
│                    Optimization Decisions                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Memory Layout          │  Warp Behavior                    │
│  ├── Data structures    │  ├── Warp-level primitives       │
│  ├── Access patterns    │  ├── Shuffle operations           │
│  └── Cache strategy     │  └── Synchronization              │
│                         │                                   │
│  Compute Strategy       │  Scheduling                       │
│  ├── Thread mapping     │  ├── Block size                   │
│  ├── Register usage     │  ├── Grid configuration           │
│  └── SIMD utilization   │  └── Launch bounds                │
│                         │                                   │
│  Algorithm Variant      │  Graph Structure                  │
│  ├── Direction (TD/BU)  │  ├── Degree distribution         │
│  ├── Serial/Parallel    │  ├── Component structure         │
│  └── Hybrid dispatch    │  └── Frontier dynamics            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Efficient Search Strategy

**Not brute force:** Diligent Learning navigates the idea space efficiently by:

1. **Learning from failures:** Prune unpromising branches
2. **Transfer across algorithms:** Apply insights from one algorithm to another
3. **Hardware-aware priors:** Use GPU architecture knowledge to guide search

---

## Part 4: PAC Reasoning

### 4.1 Probably Approximately Correct Verification

**Challenge:** How to verify correctness when:
- Multiple valid outputs exist (e.g., community detection)
- Numerical precision varies
- Non-determinism from parallel execution

**Solution:** PAC Reasoning — determine whether outputs are correct given a problem description

### 4.2 Algorithm-Specific Verification

**BFS/Connectivity:**
- **Invariant 1:** Component count matches reference
- **Invariant 2:** Every edge connects vertices in same component
- **Invariant 3:** Vertices in different components have different labels

**Louvain Community Detection:**
- **Challenge:** Result is inherently variable (depends on processing order)
- **Solution:** Stochastic Block Model (SBM) test graphs
  - Plant known community structure
  - Verify modularity score within acceptable range
  - Control signal-to-noise ratio

**PageRank:**
- **Invariant:** Converged solution satisfies `r = M × r` within tolerance
- **Verification:** Check convergence criterion directly

### 4.3 Performance Measurement

**Challenges with GPU timing:**

1. **Asynchronous execution:** Naive timing inflates speedups
2. **L2 cache effects:** Warm cache vs cold cache
3. **Thermal throttling:** Sustained performance vs burst
4. **Input distribution:** Dense vs sparse affects performance

**WarpSpeed's approach:**

```python
def measure_performance(kernel, inputs, warmup=100, runs=50):
    """Accurate GPU performance measurement."""
    # Warmup to eliminate cold-start effects
    for _ in range(warmup):
        run_kernel(kernel, inputs)
    cuda.synchronize()
    
    # Timed runs with events
    times = []
    for _ in range(runs):
        start = cuda.Event()
        end = cuda.Event()
        start.record()
        run_kernel(kernel, inputs)
        end.record()
        end.synchronize()
        times.append(elapsed(start, end))
    
    # Report median (robust to outliers)
    return median(times)
```

---

## Part 5: Meta-Pattern Engine

### 5.1 4-Way Dispatch Pattern

The core meta-pattern for graph algorithms:

```
┌─────────────────────────────────────────────────────────────┐
│                    4-Way Dispatch                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Input Characteristics                                     │
│   ├── Graph structure (dense/sparse/skewed)                │
│   ├── Frontier size (large/small)                          │
│   └── Degree distribution (uniform/power-law)              │
│                                                             │
│              │                                              │
│              ▼                                              │
│   ┌─────────────────────────────────────────┐              │
│   │         Dispatch Decision               │              │
│   │                                         │              │
│   │  ┌─────────┐ ┌─────────┐ ┌─────────┐   │              │
│   │  │ Serial  │ │ Thread   │ │ Warp    │   │              │
│   │  │ Tier    │ │ Tier     │ │ Tier    │   │              │
│   │  │         │ │          │ │         │   │              │
│   │  │ N≤200   │ │ deg<8    │ │ deg≥8   │   │              │
│   │  └─────────┘ └─────────┘ └─────────┘   │              │
│   │                                         │              │
│   │  ┌─────────────────────────────────┐   │              │
│   │  │      Cooperative Kernel         │   │              │
│   │  │      (Grid-wide sync)           │   │              │
│   │  └─────────────────────────────────┘   │              │
│   └─────────────────────────────────────────┘              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 BFS Direction-Optimizing

**Pattern:** Switch between Top-Down and Bottom-Up traversal based on frontier size

```cuda
// Top-Down: Efficient for small frontiers
// Check all edges from frontier vertices

// Bottom-Up: Efficient for large frontiers
// Check if any neighbor is in frontier

// Thresholds (from SKILLS.md)
#define TD_TO_BU_THRESHOLD(N) (N / 20)   // frontier > 5% of vertices
#define BU_TO_TD_THRESHOLD(N) (N / 200)  // frontier < 0.5% of vertices
```

**Performance Impact:**
- Pure Top-Down: Good for small frontiers, bad for large
- Pure Bottom-Up: Good for large frontiers, bad for small
- Hybrid: Best of both worlds — **up to 10× speedup**

### 5.3 Louvain 3-Tier Dispatch

**Pattern:** Choose parallelization strategy based on average degree

```cuda
// Tier 1: Serial (N ≤ 200)
// Single-threaded for tiny graphs
if (num_vertices <= 200) {
    louvain_serial(graph);
    return;
}

// Tier 2: Thread-level (avg_degree < 8)
// Each thread processes one vertex
if (avg_degree < 8) {
    louvain_thread_tier<<<blocks, threads>>>(graph);
    return;
}

// Tier 3: Warp-level (avg_degree >= 8)
// Warp cooperates on each vertex
louvain_warp_tier<<<blocks, WARPS_PER_BLOCK * 32>>>(graph);
```

**Why it works:**
- Low-degree graphs: Thread-level avoids synchronization overhead
- High-degree graphs: Warp-level amortizes work across lanes

---

## Part 6: GPU-Specific Specialization

### 6.1 Per-Architecture Kernels

WarpSpeed generates distinct kernels for each GPU:

| GPU | Architecture | Key Optimizations |
|-----|---------------|-------------------|
| A100 | sm_80 | L2 cache pinning, 40MB L2, 108 SMs |
| L4 | sm_89 | Similar to A100, smaller cache |
| A10G | sm_80 | Same as A100, different memory bandwidth |

### 6.2 A100-Specific Techniques

**L2 Cache Pinning:**

```cuda
// Pin parent array to L2 cache for Union-Find
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 30 * 1024 * 1024);  // 30MB

// Access pattern benefits from persistence
__device__ int find_root(int* parent, int x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];  // Path compression
        x = parent[x];
    }
    return x;
}
```

**Cooperative Kernels:**

```cuda
// Grid-wide synchronization for single-kernel algorithms
__global__ void __launch_bounds__(256) cooperative_kernel(...) {
    // Use cooperative groups for grid-wide sync
    cg::grid_group grid = cg::this_grid();
    
    // Phase 1: Local computation
    local_computation();
    grid.sync();  // All blocks synchronize
    
    // Phase 2: Global reduction
    global_reduction();
}
```

### 6.3 CachePool Pattern

**LRU cache for GPU-resident data:**

```cpp
// From cache_pool.hpp
template<typename T>
class CachePool {
    static constexpr int MAX_ENTRIES = 8;
    
    std::unordered_map<std::string, T> cache_;
    std::list<std::string> lru_order_;
    
public:
    T& get_or_create(const std::string& key, 
                     std::function<T()> factory) {
        if (cache_.count(key)) {
            // Move to back (most recently used)
            lru_order_.remove(key);
            lru_order_.push_back(key);
            return cache_[key];
        }
        
        // Evict if full
        if (cache_.size() >= MAX_ENTRIES) {
            std::string evict = lru_order_.front();
            lru_order_.pop_front();
            cache_.erase(evict);
        }
        
        // Create new entry
        cache_[key] = factory();
        lru_order_.push_back(key);
        return cache_[key];
    }
};
```

**Why it matters:**
- Avoids repeated malloc/free overhead
- Keeps frequently-used data GPU-resident
- Critical for RL evaluation loops

---

## Part 7: doubleGraph Results

### 7.1 Algorithm Coverage

| Algorithm Family | cuGraph Algorithms | Speedup Range |
|------------------|-------------------|----------------|
| Traversal | BFS, SSSP, BFS-iterative | 1.5× - 12× |
| Connectivity | WCC, SCC, KCores | 2× - 8× |
| Community | Louvain, Leiden, ECG | 1.8× - 6× |
| Centrality | PageRank, Katz, Betweenness | 2× - 15× |
| Similarity | Jaccard, Overlap | 3× - 10× |
| Structure | Triangle Count, K-Truss | 5× - 25× |

### 7.2 Speedup Distribution

```
Speedup Bins:
├── 1.0-1.5×:  8%   (minor improvements)
├── 1.5-2.0×:  37%  (moderate improvements)
├── 2.0-5.0×:  37%  (significant improvements)
├── 5.0-10.0×: 18%  (major improvements)
└── 10.0-100×: 18%  (transformative improvements)

Mean: 3.6×
```

### 7.3 Notable Optimizations

**BFS (up to 12×):**
- Direction-optimizing dispatch
- L2 cache pinning for frontier
- Warp-cooperative expansion

**Triangle Count (up to 25×):**
- Edge-centric parallelism
- Shared memory adjacency caching
- Load-balanced edge distribution

**PageRank (up to 15×):**
- Sparse-dense hybrid traversal
- Residual-based termination
- Warp-level reduction

---

## Part 8: Implications for KernelForge

### 8.1 What to Adopt

| Component | WarpSpeed | KernelForge Adoption |
|-----------|-----------|---------------------|
| 4-way dispatch | Core pattern | ✅ Curriculum phases |
| GPU-specific specialization | Per-arch kernels | ✅ A100 defaults |
| CachePool | LRU GPU cache | ✅ GPUCachePool class |
| PAC verification | Algorithm-specific | ✅ WCC invariants |
| L2 cache pinning | A100 optimization | ✅ SKILL.md guidance |

### 8.2 Key Code Patterns

**CachePool (Python):**

```python
class GPUCachePool:
    def __init__(self, max_entries: int = 8):
        self._max = max_entries
        self._cache = {}
        self._order = []

    def get_or_create(self, key: str, factory):
        if key in self._cache:
            self._order.remove(key)
            self._order.append(key)
            return self._cache[key]
        if len(self._cache) >= self._max:
            evict = self._order.pop(0)
            del self._cache[evict]
        val = factory()
        self._cache[key] = val
        self._order.append(key)
        return val
```

**BFS Dispatch (CUDA):**

```cuda
__global__ void bfs_hybrid(
    const int* row_ptr, const int* col_idx,
    int* frontier, int* next_frontier,
    int* distances, int n, int frontier_size
) {
    // Dispatch based on frontier size
    if (frontier_size > n / 20) {
        // Bottom-Up: Check if any neighbor is in frontier
        bfs_bottom_up<<<grid, block>>>(...);
    } else {
        // Top-Down: Expand from frontier
        bfs_top_down<<<grid, block>>>(...);
    }
}
```

### 8.3 SKILL.md Integration

WarpSpeed's SKILLS.md provides:

1. **Hardware specifications:** L2 cache, SMEM, registers per SM
2. **Dispatch thresholds:** N/20, N/200 for BFS direction
3. **Optimization patterns:** CachePool, cooperative groups
4. **Per-algorithm guidance:** BFS, Louvain, PageRank specifics

---

## Part 9: Technical Deep Dives

### 9.1 BFS Direction-Optimizing Detail

**Top-Down BFS:**
```
For each vertex v in frontier:
    For each neighbor u of v:
        If u not visited:
            Mark u visited
            Add u to next_frontier
```

**Complexity:** O(|E_frontier|) where E_frontier = edges from frontier

**Bottom-Up BFS:**
```
For each unvisited vertex u:
    For each neighbor v of u:
        If v in frontier:
            Mark u visited
            Add u to next_frontier
            Break
```

**Complexity:** O(|E_unvisited|) where E_unvisited = edges to unvisited

**Hybrid Decision:**
- Top-Down efficient when frontier is small (few edges to check)
- Bottom-Up efficient when frontier is large (many vertices can find neighbor quickly)

### 9.2 Louvain 3-Tier Detail

**Tier 1: Serial (N ≤ 200)**
- Single-threaded processing
- No synchronization overhead
- Optimal for tiny graphs

**Tier 2: Thread-Level (avg_degree < 8)**
- Each thread processes one vertex
- Low-degree = less work per thread
- Atomic operations for community updates

**Tier 3: Warp-Level (avg_degree ≥ 8)**
- Warp cooperates on each vertex
- High-degree = enough work to amortize sync
- Warp-level reductions for delta-Q computation

---

## Part 10: References

| Claim | Source |
|-------|--------|
| 3.6× average speedup | doubleAI blog, March 3, 2026 |
| 192 kernels per GPU | doubleGraph SKILLS.md |
| BFS TD→BU threshold N/20 | doubleGraph SKILLS.md Section 4.2 |
| BFS BU→TD threshold N/200 | doubleGraph SKILLS.md Section 4.2 |
| Louvain serial threshold N≤200 | doubleGraph SKILLS.md |
| Louvain thread tier avg_degree < 8 | doubleGraph SKILLS.md |
| CachePool 8 entries | doubleGraph cache_pool.hpp |
| 4-way dispatch | doubleGraph SKILLS.md Section 3 |
| A100 L2 cache 40MB | NVIDIA A100 whitepaper |
| L2 persistence 30MB (75% of L2) | doubleGraph pattern |

---

## Appendix: File Structure

```
doubleGraph/
├── cpp/src/aai/impl/
│   ├── a100/           # A100-specific kernels
│   │   ├── bfs.cu
│   │   ├── wcc.cu
│   │   ├── louvain.cu
│   │   └── ...
│   ├── a10g/           # A10G-specific kernels
│   ├── l4/             # L4-specific kernels
│   └── shared/         # Common utilities
│       ├── cache_pool.hpp
│       └── cooperative.cuh
├── SKILLS.md           # Optimization patterns
└── README.md
```
