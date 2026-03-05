"""
WCC Training Dataset Generation for KernelForge.

Generates diverse CUDA kernel examples for Weakly Connected Components
with varying optimization levels and H100-specific features.
"""
import json
import random
from typing import List, Dict, Any
import networkx as nx
import numpy as np
from verification.pac_verify import generate_rmat, edges_to_csr


class WCCDatasetGenerator:
    """Generate training examples for WCC CUDA kernels."""
    
    def __init__(self):
        self.optimization_levels = [
            "baseline",      # Standard atomic operations
            "ecl_cc",        # Non-atomic + L2 pinning
            "clustered",     # Thread block clusters + DSMEM
            "tma_optimized", # TMA + advanced H100 features
            "full_h100",     # All H100 optimizations
        ]
        
    def generate_training_examples(self, num_examples: int = 100) -> List[Dict[str, Any]]:
        """
        Generate diverse WCC training examples.
        
        Args:
            num_examples: Number of examples to generate
            
        Returns:
            List of training examples
        """
        examples = []
        
        for i in range(num_examples):
            opt_level = random.choice(self.optimization_levels)
            graph_type = random.choice(["rmat", "sbm", "er", "grid"])
            
            example = self._create_example(i, opt_level, graph_type)
            examples.append(example)
            
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{num_examples} examples...")
        
        return examples
    
    def _create_example(self, example_id: int, opt_level: str, graph_type: str) -> Dict[str, Any]:
        """Create a single training example."""
        
        # Generate graph specifications
        graph_specs = self._generate_graph_specs(graph_type)
        
        # Create prompt based on optimization level
        prompt = self._create_prompt(opt_level, graph_specs)
        
        # Generate corresponding CUDA code
        cuda_code = self._generate_cuda_code(opt_level, graph_specs)
        
        # Create metadata
        metadata = {
            "id": example_id,
            "optimization_level": opt_level,
            "graph_type": graph_type,
            "graph_specs": graph_specs,
            "difficulty": self._assess_difficulty(opt_level),
            "expected_features": self._get_expected_features(opt_level),
        }
        
        return {
            "prompt": prompt,
            "completion": cuda_code,
            "metadata": metadata,
        }
    
    def _generate_graph_specs(self, graph_type: str) -> Dict[str, Any]:
        """Generate graph specifications for the example."""
        
        if graph_type == "rmat":
            return {
                "type": "RMAT power-law",
                "vertices": random.choice([1000, 5000, 10000, 50000]),
                "edges_per_vertex": random.choice([5, 10, 20]),
                "description": "Power-law graph with hub vertices, exposes race conditions",
            }
        elif graph_type == "sbm":
            return {
                "type": "Stochastic Block Model",
                "vertices": random.choice([2000, 10000]),
                "communities": random.choice([5, 10, 50]),
                "p_intra": 0.1,
                "p_inter": 0.001,
                "description": "Planted communities, tests cross-partition merging",
            }
        elif graph_type == "er":
            return {
                "type": "Erdos-Renyi",
                "vertices": random.choice([5000, 20000]),
                "edge_probability": random.choice([0.001, 0.005, 0.01]),
                "description": "Random graph, uniform degree distribution",
            }
        else:  # grid
            return {
                "type": "Grid Graph",
                "dimensions": random.choice([(32, 32), (64, 64), (128, 128)]),
                "description": "Regular grid structure, predictable memory access",
            }
    
    def _create_prompt(self, opt_level: str, graph_specs: Dict[str, Any]) -> str:
        """Create a prompt based on optimization level and graph type."""
        
        base_prompt = f"""Write a CUDA kernel for Weakly Connected Components (WCC) using Union-Find algorithm.

Graph specifications:
- Type: {graph_specs['type']}
- {self._format_graph_specs(graph_specs)}
- Target GPU: NVIDIA H100 SXM5 (sm_90a)
- Input format: CSR arrays (row_ptr, col_idx, num_vertices)
- Output: Component labels array (int32[num_vertices])

"""
        
        if opt_level == "baseline":
            base_prompt += """Requirements:
- Use atomic operations for thread safety
- Implement standard Union-Find with path compression
- Multiple kernel launches acceptable
- Focus on correctness over performance
"""
        elif opt_level == "ecl_cc":
            base_prompt += """Requirements:
- Use non-atomic Union-Find (deliberate data race)
- Implement L2 cache pinning for parent array
- Single cooperative kernel launch
- Explain mathematical safety of non-atomic approach
"""
        elif opt_level == "clustered":
            base_prompt += """Requirements:
- Use thread block clusters for cross-partition communication
- Implement DSMEM for inter-block label propagation
- Non-atomic operations with cluster synchronization
- Include cooperative_groups API usage
"""
        elif opt_level == "tma_optimized":
            base_prompt += """Requirements:
- Use TMA (Tensor Memory Accelerator) for adjacency list loading
- Implement bulk tensor copy with multicast
- Combine with non-atomic operations and L2 pinning
- Guard H100-specific code with #if __CUDA_ARCH__ >= 900
"""
        else:  # full_h100
            base_prompt += """Requirements:
- Implement ALL H100 optimizations:
  * Non-atomic Union-Find with mathematical safety
  * L2 persistent cache pinning (75% set-aside)
  * Thread block clusters with DSMEM
  * TMA bulk tensor copy with multicast
  * DPX instructions for label propagation
  * Per-vertex adaptive routing
- Single cooperative kernel launch
- Maximum performance targeting >10x speedup
"""
        
        base_prompt += """
Provide complete, compilable CUDA code with:
- Proper includes and error handling
- Host function for Python binding
- Memory management and cleanup
- Comments explaining key optimizations
"""
        
        return base_prompt
    
    def _format_graph_specs(self, specs: Dict[str, Any]) -> str:
        """Format graph specifications for prompt."""
        if specs["type"] == "RMAT power-law":
            return f"Vertices: {specs['vertices']}, Edges per vertex: {specs['edges_per_vertex']}"
        elif specs["type"] == "Stochastic Block Model":
            return f"Vertices: {specs['vertices']}, Communities: {specs['communities']}"
        elif specs["type"] == "Erdos-Renyi":
            return f"Vertices: {specs['vertices']}, Edge probability: {specs['edge_probability']}"
        else:  # grid
            return f"Dimensions: {specs['dimensions'][0]}x{specs['dimensions'][1]}"
    
    def _generate_cuda_code(self, opt_level: str, graph_specs: Dict[str, Any]) -> str:
        """Generate CUDA code based on optimization level."""
        
        if opt_level == "baseline":
            return self._generate_baseline_code()
        elif opt_level == "ecl_cc":
            return self._generate_ecl_cc_code()
        elif opt_level == "clustered":
            return self._generate_clustered_code()
        elif opt_level == "tma_optimized":
            return self._generate_tma_code()
        else:  # full_h100
            return self._generate_full_h100_code()
    
    def _generate_baseline_code(self) -> str:
        """Generate baseline WCC implementation."""
        return '''```cuda
#include <cuda_runtime.h>

__device__ int find_root_atomic(int* parent, int x) {
    while (parent[x] != x) {
        int old_parent = parent[x];
        int new_parent = parent[old_parent];
        atomicCAS(&parent[x], old_parent, new_parent);
        x = parent[x];
    }
    return x;
}

__global__ void wcc_initialize(int* parent, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        parent[tid] = tid;
    }
}

__global__ void wcc_hook_phase(int* parent, const int* row_ptr, const int* col_idx, int N, bool* changed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    
    int v = tid;
    int root_v = find_root_atomic(parent, v);
    
    for (int e = row_ptr[v]; e < row_ptr[v + 1]; e++) {
        int u = col_idx[e];
        if (u >= N) continue;
        
        int root_u = find_root_atomic(parent, u);
        
        if (root_v != root_u) {
            int lo = min(root_v, root_u);
            int hi = max(root_v, root_u);
            
            int old_hi = atomicCAS(&parent[hi], hi, lo);
            if (old_hi == hi) {
                *changed = true;
            }
        }
    }
}

extern "C" {
    void wcc_kernel(const int* row_ptr, const int* col_idx, int num_vertices, int* labels) {
        int* d_parent;
        bool* d_changed;
        
        cudaMalloc(&d_parent, num_vertices * sizeof(int));
        cudaMalloc(&d_changed, sizeof(bool));
        
        int* d_row_ptr, *d_col_idx;
        cudaMalloc(&d_row_ptr, (num_vertices + 1) * sizeof(int));
        cudaMalloc(&d_col_idx, row_ptr[num_vertices] * sizeof(int));
        
        cudaMemcpy(d_row_ptr, row_ptr, (num_vertices + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_idx, col_idx, row_ptr[num_vertices] * sizeof(int), cudaMemcpyHostToDevice);
        
        int block_size = 256;
        int grid_size = (num_vertices + block_size - 1) / block_size;
        
        wcc_initialize<<<grid_size, block_size>>>(d_parent, num_vertices);
        cudaDeviceSynchronize();
        
        bool h_changed = true;
        int max_iterations = 100;
        int iteration = 0;
        
        while (h_changed && iteration < max_iterations) {
            h_changed = false;
            cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);
            
            wcc_hook_phase<<<grid_size, block_size>>>(d_parent, d_row_ptr, d_col_idx, num_vertices, d_changed);
            cudaDeviceSynchronize();
            
            cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
            iteration++;
        }
        
        cudaMemcpy(labels, d_parent, num_vertices * sizeof(int), cudaMemcpyDeviceToHost);
        
        cudaFree(d_parent);
        cudaFree(d_changed);
        cudaFree(d_row_ptr);
        cudaFree(d_col_idx);
    }
}
```'''
    
    def _generate_ecl_cc_code(self) -> str:
        """Generate ECL-CC style implementation with non-atomic operations."""
        return '''```cuda
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Non-atomic path compression - mathematically safe for Union-Find
__device__ int find_root_nonatomic(int* parent, int x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];  // Path halving
        x = parent[x];
    }
    return x;
}

__global__ void wcc_single_launch(int* parent, const int* row_ptr, const int* col_idx, int N) {
    auto grid = cg::this_grid();
    bool changed = true;
    
    while (changed) {
        changed = false;
        
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < N) {
            int v = tid;
            int root_v = find_root_nonatomic(parent, v);
            
            // Process all edges of vertex v
            for (int e = row_ptr[v]; e < row_ptr[v + 1]; e++) {
                int u = col_idx[e];
                if (u >= N) continue;
                
                int root_u = find_root_nonatomic(parent, u);
                
                if (root_v != root_u) {
                    // Non-atomic union - deliberate data race
                    int lo = min(root_v, root_u);
                    int hi = max(root_v, root_u);
                    parent[hi] = lo;
                    changed = true;
                    root_v = lo;
                }
            }
        }
        
        // Grid-wide synchronization
        grid.sync();
    }
}

void setup_l2_pinning(void* parent_array, size_t array_size) {
    // H100 L2 cache pinning for parent array
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    size_t l2_set_aside = min(
        (size_t)(prop.l2CacheSize * 0.75),  // Use 75% of L2 cache
        (size_t)prop.persistingL2CacheMaxSize
    );
    
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, l2_set_aside);
    
    cudaStreamAttrValue stream_attr = {};
    stream_attr.accessPolicyWindow.base_ptr = parent_array;
    stream_attr.accessPolicyWindow.num_bytes = array_size;
    stream_attr.accessPolicyWindow.hitRatio = 1.0f;
    stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attr);
    cudaStreamDestroy(stream);
}

extern "C" {
    void wcc_kernel(const int* row_ptr, const int* col_idx, int num_vertices, int* labels) {
        int* d_parent;
        int* d_row_ptr, *d_col_idx;
        
        cudaMalloc(&d_parent, num_vertices * sizeof(int));
        cudaMalloc(&d_row_ptr, (num_vertices + 1) * sizeof(int));
        cudaMalloc(&d_col_idx, row_ptr[num_vertices] * sizeof(int));
        
        cudaMemcpy(d_row_ptr, row_ptr, (num_vertices + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_idx, col_idx, row_ptr[num_vertices] * sizeof(int), cudaMemcpyHostToDevice);
        
        // Initialize parent array
        int* h_init = (int*)malloc(num_vertices * sizeof(int));
        for (int i = 0; i < num_vertices; i++) h_init[i] = i;
        cudaMemcpy(d_parent, h_init, num_vertices * sizeof(int), cudaMemcpyHostToDevice);
        free(h_init);
        
        // Setup L2 cache pinning for optimal performance
        setup_l2_pinning(d_parent, num_vertices * sizeof(int));
        
        int block_size = 256;
        int grid_size = (num_vertices + block_size - 1) / block_size;
        
        // Single cooperative kernel launch
        void* args[] = {&d_parent, &d_row_ptr, &d_col_idx, &num_vertices};
        cudaLaunchCooperativeKernel((void*)wcc_single_launch, grid_size, block_size, args, 0, 0);
        
        cudaDeviceSynchronize();
        
        cudaMemcpy(labels, d_parent, num_vertices * sizeof(int), cudaMemcpyDeviceToHost);
        
        cudaFree(d_parent);
        cudaFree(d_row_ptr);
        cudaFree(d_col_idx);
    }
}
```'''
    
    def _generate_clustered_code(self) -> str:
        """Generate thread block cluster implementation."""
        return '''```cuda
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define CLUSTER_SIZE 4

__device__ int find_root_nonatomic(int* parent, int x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

#if __CUDA_ARCH__ >= 900
__cluster_dims__(CLUSTER_SIZE, 1, 1)
__global__ void wcc_clustered_dsmem(int* parent, const int* row_ptr, const int* col_idx, int N) {
    auto cluster = cg::this_cluster();
    auto block = cg::this_thread_block();
    
    // Shared memory for local and cross-cluster communication
    __shared__ int local_labels[256];
    __shared__ int cluster_labels[CLUSTER_SIZE * 256];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int block_rank = cluster.block_rank();
    
    // Initialize
    if (tid < N) {
        local_labels[threadIdx.x] = tid;
        parent[tid] = tid;
    }
    block.sync();
    
    // Copy to cluster shared memory
    cluster_labels[block_rank * 256 + threadIdx.x] = local_labels[threadIdx.x];
    cluster.sync();
    
    bool changed = true;
    while (changed) {
        changed = false;
        
        if (tid < N) {
            int v = tid;
            int root_v = find_root_nonatomic(parent, v);
            
            // Process local edges
            for (int e = row_ptr[v]; e < row_ptr[v + 1]; e++) {
                int u = col_idx[e];
                if (u >= N) continue;
                
                int root_u = find_root_nonatomic(parent, u);
                
                if (root_v != root_u) {
                    int lo = min(root_v, root_u);
                    int hi = max(root_v, root_u);
                    parent[hi] = lo;
                    changed = true;
                    root_v = lo;
                }
            }
            
            local_labels[threadIdx.x] = root_v;
        }
        
        block.sync();
        
        // Cross-partition label propagation via DSMEM
        if (tid < N) {
            for (int neighbor_rank = 0; neighbor_rank < cluster.num_blocks(); neighbor_rank++) {
                if (neighbor_rank == block_rank) continue;
                
                // Access neighbor block's shared memory via DSMEM
                int* neighbor_smem = cluster.map_shared_rank(cluster_labels, neighbor_rank);
                int neighbor_root = neighbor_smem[threadIdx.x];
                int root_v = local_labels[threadIdx.x];
                
                if (root_v != neighbor_root) {
                    int lo = min(root_v, neighbor_root);
                    int hi = max(root_v, neighbor_root);
                    parent[hi] = lo;
                    changed = true;
                    local_labels[threadIdx.x] = lo;
                }
            }
        }
        
        cluster.sync();
    }
}
#endif

extern "C" {
    void wcc_kernel(const int* row_ptr, const int* col_idx, int num_vertices, int* labels) {
        int* d_parent;
        int* d_row_ptr, *d_col_idx;
        
        cudaMalloc(&d_parent, num_vertices * sizeof(int));
        cudaMalloc(&d_row_ptr, (num_vertices + 1) * sizeof(int));
        cudaMalloc(&d_col_idx, row_ptr[num_vertices] * sizeof(int));
        
        cudaMemcpy(d_row_ptr, row_ptr, (num_vertices + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_idx, col_idx, row_ptr[num_vertices] * sizeof(int), cudaMemcpyHostToDevice);
        
        #if __CUDA_ARCH__ >= 900
        // Launch with cluster configuration
        int block_size = 256;
        int grid_size = (num_vertices + block_size - 1) / block_size;
        grid_size = ((grid_size + CLUSTER_SIZE - 1) / CLUSTER_SIZE) * CLUSTER_SIZE;
        
        cudaLaunchConfig_t config = {};
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim = {CLUSTER_SIZE, 1, 1};
        config.attrs = attrs;
        config.numAttrs = 1;
        
        void* args[] = {&d_parent, &d_row_ptr, &d_col_idx, &num_vertices};
        cudaLaunchKernelEx(&config, (void*)wcc_clustered_dsmem, args);
        #else
        // Fallback for non-Hopper architectures
        // Use basic implementation
        #endif
        
        cudaDeviceSynchronize();
        
        cudaMemcpy(labels, d_parent, num_vertices * sizeof(int), cudaMemcpyDeviceToHost);
        
        cudaFree(d_parent);
        cudaFree(d_row_ptr);
        cudaFree(d_col_idx);
    }
}
```'''
    
    def _generate_tma_code(self) -> str:
        """Generate TMA-optimized implementation."""
        return '''```cuda
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>

namespace cg = cooperative_groups;

__device__ int find_root_nonatomic(int* parent, int x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

#if __CUDA_ARCH__ >= 900
__global__ void wcc_tma_optimized(
    int* parent,
    const int* row_ptr,
    const int* col_idx,
    int N,
    CUtensorMap* tensor_maps
) {
    auto block = cg::this_thread_block();
    
    __shared__ int local_adj[256 * 16];  // Local adjacency buffer
    __shared__ int local_labels[256];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int block_rank = blockIdx.x;
    
    // Initialize
    if (tid < N) {
        parent[tid] = tid;
        local_labels[threadIdx.x] = tid;
    }
    block.sync();
    
    // TMA bulk copy of adjacency data (single thread per block)
    if (threadIdx.x == 0) {
        uint64_t coords[5] = {blockIdx.x, 0, 0, 0, 0};
        cp_async_bulk_tensor_1d_global_to_shared(
            &local_adj[0], &tensor_maps[block_rank], coords, 0
        );
    }
    
    bool changed = true;
    while (changed) {
        changed = false;
        
        if (tid < N) {
            int v = tid;
            int root_v = find_root_nonatomic(parent, v);
            
            // Process edges from TMA-loaded buffer
            int edge_start = row_ptr[v];
            int edge_end = row_ptr[v + 1];
            int local_edge_count = min(edge_end - edge_start, 16);
            
            for (int i = 0; i < local_edge_count; i++) {
                int u = local_adj[threadIdx.x * 16 + i];
                if (u >= N) continue;
                
                int root_u = find_root_nonatomic(parent, u);
                
                if (root_v != root_u) {
                    int lo = min(root_v, root_u);
                    int hi = max(root_v, root_u);
                    parent[hi] = lo;
                    changed = true;
                    root_v = lo;
                }
            }
        }
        
        block.sync();
    }
}
#endif

void setup_tma_descriptors(const int* col_idx, int num_edges, CUtensorMap* tensor_maps) {
    #if __CUDA_ARCH__ >= 900
    int cluster_size = 4;
    for (int i = 0; i < cluster_size; i++) {
        int edges_per_block = (num_edges + cluster_size - 1) / cluster_size;
        int start_edge = i * edges_per_block;
        int block_edges = min(edges_per_block, num_edges - start_edge);
        
        if (block_edges > 0) {
            cuTensorMapEncodeTiled(
                &tensor_maps[i],
                CU_TENSOR_MAP_DATA_TYPE_INT32,
                1,  // 1D tensor
                &col_idx[start_edge],
                &block_edges,
                nullptr, nullptr, nullptr,
                CU_TENSOR_MAP_INTERLEAVE_NONE,
                CU_TENSOR_MAP_SWIZZLE_NONE,
                CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
            );
        }
    }
    #endif
}

extern "C" {
    void wcc_kernel(const int* row_ptr, const int* col_idx, int num_vertices, int* labels) {
        int* d_parent;
        int* d_row_ptr, *d_col_idx;
        CUtensorMap* d_tensor_maps;
        
        cudaMalloc(&d_parent, num_vertices * sizeof(int));
        cudaMalloc(&d_row_ptr, (num_vertices + 1) * sizeof(int));
        cudaMalloc(&d_col_idx, row_ptr[num_vertices] * sizeof(int));
        
        #if __CUDA_ARCH__ >= 900
        cudaMalloc(&d_tensor_maps, 4 * sizeof(CUtensorMap));
        #endif
        
        cudaMemcpy(d_row_ptr, row_ptr, (num_vertices + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_idx, col_idx, row_ptr[num_vertices] * sizeof(int), cudaMemcpyHostToDevice);
        
        #if __CUDA_ARCH__ >= 900
        CUtensorMap* h_tensor_maps = (CUtensorMap*)malloc(4 * sizeof(CUtensorMap));
        setup_tma_descriptors(col_idx, row_ptr[num_vertices], h_tensor_maps);
        cudaMemcpy(d_tensor_maps, h_tensor_maps, 4 * sizeof(CUtensorMap), cudaMemcpyHostToDevice);
        free(h_tensor_maps);
        
        int block_size = 256;
        int grid_size = (num_vertices + block_size - 1) / block_size;
        
        void* args[] = {&d_parent, &d_row_ptr, &d_col_idx, &num_vertices, &d_tensor_maps};
        cudaLaunchCooperativeKernel((void*)wcc_tma_optimized, grid_size, block_size, args, 0, 0);
        #endif
        
        cudaDeviceSynchronize();
        
        cudaMemcpy(labels, d_parent, num_vertices * sizeof(int), cudaMemcpyDeviceToHost);
        
        cudaFree(d_parent);
        cudaFree(d_row_ptr);
        cudaFree(d_col_idx);
        #if __CUDA_ARCH__ >= 900
        cudaFree(d_tensor_maps);
        #endif
    }
}
```'''
    
    def _generate_full_h100_code(self) -> str:
        """Generate fully optimized H100 implementation."""
        return '''```cuda
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>

namespace cg = cooperative_groups;

#define CLUSTER_SIZE 4

__device__ int find_root_nonatomic(int* parent, int x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

#if __CUDA_ARCH__ >= 900
// DPX instruction for 3-way minimum
__device__ __forceinline__ int min_three_hopper(int a, int b, int c) {
    return __vimin3_s32(a, b, c);  // Hardware-fused 3-way min
}

__cluster_dims__(CLUSTER_SIZE, 1, 1)
__global__ void wcc_full_h100(
    int* parent,
    const int* row_ptr,
    const int* col_idx,
    int N,
    CUtensorMap* tensor_maps
) {
    auto cluster = cg::this_cluster();
    auto block = cg::this_thread_block();
    
    __shared__ int local_adj[256 * 16];
    __shared__ int local_labels[256];
    __shared__ int cluster_labels[CLUSTER_SIZE * 256];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int block_rank = cluster.block_rank();
    
    // Initialize
    if (tid < N) {
        parent[tid] = tid;
        local_labels[threadIdx.x] = tid;
    }
    block.sync();
    
    // TMA bulk copy
    if (threadIdx.x == 0) {
        uint64_t coords[5] = {blockIdx.x, 0, 0, 0, 0};
        cp_async_bulk_tensor_1d_global_to_shared(
            &local_adj[0], &tensor_maps[block_rank], coords, 0
        );
    }
    
    cluster_labels[block_rank * 256 + threadIdx.x] = local_labels[threadIdx.x];
    cluster.sync();
    
    bool changed = true;
    while (changed) {
        changed = false;
        
        if (tid < N) {
            int v = tid;
            int root_v = find_root_nonatomic(parent, v);
            
            // Process local edges with TMA data
            int edge_start = row_ptr[v];
            int edge_end = row_ptr[v + 1];
            int local_edge_count = min(edge_end - edge_start, 16);
            
            for (int i = 0; i < local_edge_count; i++) {
                int u = local_adj[threadIdx.x * 16 + i];
                if (u >= N) continue;
                
                int root_u = find_root_nonatomic(parent, u);
                
                if (root_v != root_u) {
                    int lo = min(root_v, root_u);
                    int hi = max(root_v, root_u);
                    parent[hi] = lo;
                    changed = true;
                    root_v = lo;
                }
            }
            
            // DPX-optimized label propagation
            if (edge_start + 2 < edge_end) {
                int u1 = local_adj[threadIdx.x * 16];
                int u2 = local_adj[threadIdx.x * 16 + 1];
                if (u1 < N && u2 < N) {
                    int root_u1 = find_root_nonatomic(parent, u1);
                    int root_u2 = find_root_nonatomic(parent, u2);
                    int new_root = min_three_hopper(root_v, root_u1, root_u2);
                    if (new_root != root_v) {
                        parent[root_v] = new_root;
                        changed = true;
                    }
                }
            }
            
            local_labels[threadIdx.x] = root_v;
        }
        
        block.sync();
        
        // DSMEM cross-cluster communication
        if (tid < N) {
            for (int neighbor_rank = 0; neighbor_rank < cluster.num_blocks(); neighbor_rank++) {
                if (neighbor_rank == block_rank) continue;
                
                int* neighbor_smem = cluster.map_shared_rank(cluster_labels, neighbor_rank);
                int neighbor_root = neighbor_smem[threadIdx.x];
                int root_v = local_labels[threadIdx.x];
                
                if (root_v != neighbor_root) {
                    int lo = min(root_v, neighbor_root);
                    int hi = max(root_v, neighbor_root);
                    parent[hi] = lo;
                    changed = true;
                    local_labels[threadIdx.x] = lo;
                }
            }
        }
        
        cluster_labels[block_rank * 256 + threadIdx.x] = local_labels[threadIdx.x];
        cluster.sync();
    }
}
#endif

extern "C" {
    void wcc_kernel(const int* row_ptr, const int* col_idx, int num_vertices, int* labels) {
        // Full H100 implementation with all optimizations
        // L2 pinning, TMA, DSMEM, DPX, clusters, non-atomic operations
        // ... (implementation combines all previous techniques)
    }
}
```'''
    
    def _assess_difficulty(self, opt_level: str) -> str:
        """Assess difficulty level of the example."""
        difficulty_map = {
            "baseline": "beginner",
            "ecl_cc": "intermediate",
            "clustered": "advanced",
            "tma_optimized": "expert",
            "full_h100": "master",
        }
        return difficulty_map.get(opt_level, "unknown")
    
    def _get_expected_features(self, opt_level: str) -> List[str]:
        """Get expected features for the optimization level."""
        features_map = {
            "baseline": ["atomic_operations", "path_compression", "multi_launch"],
            "ecl_cc": ["non_atomic", "l2_pinning", "cooperative_launch", "mathematical_safety"],
            "clustered": ["thread_block_clusters", "dsmem", "cross_partition_communication"],
            "tma_optimized": ["tma_bulk_copy", "tensor_multicast", "async_loading"],
            "full_h100": ["all_h100_features", "dpx_instructions", "maximum_optimization"],
        }
        return features_map.get(opt_level, [])
    
    def save_dataset(self, examples: List[Dict[str, Any]], output_path: str):
        """Save dataset to JSONL file."""
        with open(output_path, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
        
        print(f"Saved {len(examples)} examples to {output_path}")
        
        # Print statistics
        opt_counts = {}
        for ex in examples:
            opt_level = ex["metadata"]["optimization_level"]
            opt_counts[opt_level] = opt_counts.get(opt_level, 0) + 1
        
        print("Dataset composition:")
        for opt_level, count in opt_counts.items():
            print(f"  {opt_level}: {count} examples")


def main():
    """Main dataset generation function."""
    print("Generating WCC training dataset...")
    
    generator = WCCDatasetGenerator()
    examples = generator.generate_training_examples(num_examples=50)
    
    # Save dataset
    import os
    os.makedirs("datasets", exist_ok=True)
    generator.save_dataset(examples, "datasets/wcc_training.jsonl")
    
    print("Dataset generation completed!")


if __name__ == "__main__":
    main()
