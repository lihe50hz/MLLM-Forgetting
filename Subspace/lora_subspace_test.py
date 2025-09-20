import torch
import numpy as np
import json
import os
from typing import Dict, Tuple, List
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from safetensors import safe_open
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("Warning: safetensors not available, will try to load PyTorch .bin files")

class LoRAOrthogonalityTester:
    """
    Test orthogonality between LoRA adapters as described in O-LoRA paper.
    Computes dot products of A matrices from corresponding target modules.
    """
    
    def __init__(self, adapter1_path: str, adapter2_path: str):
        """
        Initialize with paths to two LoRA adapter directories.
        
        Args:
            adapter1_path: Path to first LoRA adapter (e.g., AD directory)
            adapter2_path: Path to second LoRA adapter (e.g., Sci directory)
        """
        self.adapter1_path = Path(adapter1_path)
        self.adapter2_path = Path(adapter2_path)
        self.adapter1_weights = {}
        self.adapter2_weights = {}
        self.adapter1_config = {}
        self.adapter2_config = {}
        
    def load_adapter_config(self, adapter_path: Path) -> Dict:
        """Load adapter configuration from adapter_config.json"""
        config_path = adapter_path / "adapter_config.json"
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def load_adapter_weights(self, adapter_path: Path) -> Dict:
        """Load adapter weights from adapter_model.safetensors or .bin files"""
        weights = {}
        
        # Try safetensors first
        safetensors_path = adapter_path / "adapter_model.safetensors"
        if SAFETENSORS_AVAILABLE and safetensors_path.exists():
            with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)
            print(f"Loaded weights from {safetensors_path}")
        else:
            # Fallback to PyTorch .bin files
            bin_path = adapter_path / "adapter_model.bin"
            if bin_path.exists():
                weights = torch.load(bin_path, map_location="cpu")
                print(f"Loaded weights from {bin_path}")
            else:
                # Look for pytorch_model.bin as another fallback
                pytorch_path = adapter_path / "pytorch_model.bin"
                if pytorch_path.exists():
                    weights = torch.load(pytorch_path, map_location="cpu")
                    print(f"Loaded weights from {pytorch_path}")
                else:
                    raise FileNotFoundError(f"No adapter weights found in {adapter_path}. "
                                          f"Looked for: adapter_model.safetensors, adapter_model.bin, pytorch_model.bin")
        
        return weights
    
    def load_adapters(self):
        """Load both LoRA adapters"""
        print("Loading LoRA adapters...")
        
        # Load configurations
        self.adapter1_config = self.load_adapter_config(self.adapter1_path)
        self.adapter2_config = self.load_adapter_config(self.adapter2_path)
        
        # Load weights
        self.adapter1_weights = self.load_adapter_weights(self.adapter1_path)
        self.adapter2_weights = self.load_adapter_weights(self.adapter2_path)
        
        print(f"Adapter 1 ({self.adapter1_path.name}): {len(self.adapter1_weights)} weight tensors")
        print(f"Adapter 2 ({self.adapter2_path.name}): {len(self.adapter2_weights)} weight tensors")
        
        # Print some basic info
        print(f"Adapter 1 rank: {self.adapter1_config['r']}")
        print(f"Adapter 2 rank: {self.adapter2_config['r']}")
        print(f"Adapter 1 alpha: {self.adapter1_config['lora_alpha']}")
        print(f"Adapter 2 alpha: {self.adapter2_config['lora_alpha']}")
    
    def extract_a_matrices(self) -> Tuple[Dict, Dict]:
        """
        Extract A matrices from both adapters for corresponding target modules.
        LoRA decomposition: ΔW = B @ A, where A is the "down-projection" matrix.
        """
        a_matrices_1 = {}
        a_matrices_2 = {}
        
        # Get A matrices (lora_A) from adapter 1
        for key, tensor in self.adapter1_weights.items():
            if '.lora_B.' in key:
                # Extract module name (remove .lora_A.weight suffix)
                module_name = key.replace('.lora_B.weight', '')
                a_matrices_1[module_name] = tensor
        
        # Get A matrices from adapter 2
        for key, tensor in self.adapter2_weights.items():
            if '.lora_B.' in key:
                module_name = key.replace('.lora_B.weight', '')
                a_matrices_2[module_name] = tensor
        
        print(f"Found {len(a_matrices_1)} A matrices in adapter 1")
        print(f"Found {len(a_matrices_2)} A matrices in adapter 2")
        
        return a_matrices_1, a_matrices_2
    
    def compute_orthogonality_metrics(self, a_matrices_1: Dict, a_matrices_2: Dict) -> Dict:
        """
        Compute orthogonality metrics between corresponding A matrices.
        
        Returns:
            Dictionary containing orthogonality metrics for each module pair
        """
        results = {}
        
        # Find common modules
        common_modules = set(a_matrices_1.keys()) & set(a_matrices_2.keys())
        print(f"Found {len(common_modules)} common target modules")
        
        for module_name in common_modules:
            a1 = a_matrices_1[module_name].float()
            a2 = a_matrices_2[module_name].float()
            
            # Ensure matrices have same shape
            if a1.shape != a2.shape:
                print(f"Warning: Shape mismatch for {module_name}: {a1.shape} vs {a2.shape}")
                continue

            # Flatten matrices for dot product computation
            a1_flat = a1.flatten()
            a2_flat = a2.flatten()
            
            # Compute metrics
            dot_product = torch.dot(a1_flat, a2_flat).item()
            norm_a1 = torch.norm(a1_flat).item()
            norm_a2 = torch.norm(a2_flat).item()
            
            # Cosine similarity (normalized dot product)
            cosine_sim = dot_product / (norm_a1 * norm_a2) if (norm_a1 * norm_a2) > 0 else 0
            
            # Frobenius inner product
            frobenius_inner = torch.sum(a1 * a2).item()
            
            results[module_name] = {
                'dot_product': dot_product,
                'cosine_similarity': cosine_sim,
                'frobenius_inner_product': frobenius_inner,
                'norm_a1': norm_a1,
                'norm_a2': norm_a2,
                'shape': a1.shape
            }
        
        return results
    
    def compute_random_baseline(self, a_matrices_1: Dict, a_matrices_2: Dict) -> Dict:
        """
        Compute orthogonality metrics for random vectors with same shapes as A matrices.
        This provides a baseline to compare against - shows what orthogonality we'd expect by chance.
        
        Args:
            a_matrices_1: A matrices from first adapter (for shape reference)
            a_matrices_2: A matrices from second adapter (for shape reference)
            
        Returns:
            Dictionary containing orthogonality metrics for random vector pairs
        """
        results = {}
        
        # Find common modules (same as main analysis)
        common_modules = set(a_matrices_1.keys()) & set(a_matrices_2.keys())
        
        print(f"Computing random baseline for {len(common_modules)} modules...")
        
        for module_name in common_modules:
            # Get shapes from original matrices
            shape = a_matrices_1[module_name].shape
            
            # Generate random matrices with same shape
            # Using normal distribution (mean=0, std=1)
            random_a1 = torch.randn(shape, dtype=torch.float32)
            random_a2 = torch.randn(shape, dtype=torch.float32)
            
            # Flatten matrices for dot product computation
            a1_flat = random_a1.flatten()
            a2_flat = random_a2.flatten()
            
            # Compute metrics (same as main analysis)
            dot_product = torch.dot(a1_flat, a2_flat).item()
            norm_a1 = torch.norm(a1_flat).item()
            norm_a2 = torch.norm(a2_flat).item()
            
            # Cosine similarity (normalized dot product)
            cosine_sim = dot_product / (norm_a1 * norm_a2) if (norm_a1 * norm_a2) > 0 else 0
            
            # Frobenius inner product
            frobenius_inner = torch.sum(random_a1 * random_a2).item()
            
            results[module_name] = {
                'dot_product': dot_product,
                'cosine_similarity': cosine_sim,
                'frobenius_inner_product': frobenius_inner,
                'norm_a1': norm_a1,
                'norm_a2': norm_a2,
                'shape': shape
            }
        
        return results
    
    def analyze_orthogonality(self, results: Dict, random_baseline: Dict = None) -> Dict:
        """Analyze overall orthogonality statistics"""
        if not results:
            return {}
        
        cosine_similarities = [r['cosine_similarity'] for r in results.values()]
        dot_products = [r['dot_product'] for r in results.values()]
        frobenius_products = [r['frobenius_inner_product'] for r in results.values()]
        
        analysis = {
            'num_modules': len(results),
            'cosine_similarity': {
                'mean': np.mean(cosine_similarities),
                'std': np.std(cosine_similarities),
                'min': np.min(cosine_similarities),
                'max': np.max(cosine_similarities),
                'median': np.median(cosine_similarities)
            },
            'dot_product': {
                'mean': np.mean(dot_products),
                'std': np.std(dot_products),
                'min': np.min(dot_products),
                'max': np.max(dot_products),
                'median': np.median(dot_products)
            },
            'frobenius_inner_product': {
                'mean': np.mean(frobenius_products),
                'std': np.std(frobenius_products),
                'min': np.min(frobenius_products),
                'max': np.max(frobenius_products),
                'median': np.median(frobenius_products)
            }
        }
        
        # Add random baseline analysis if provided
        if random_baseline:
            baseline_cosine_similarities = [r['cosine_similarity'] for r in random_baseline.values()]
            baseline_dot_products = [r['dot_product'] for r in random_baseline.values()]
            baseline_frobenius_products = [r['frobenius_inner_product'] for r in random_baseline.values()]
            
            analysis['random_baseline'] = {
                'cosine_similarity': {
                    'mean': np.mean(baseline_cosine_similarities),
                    'std': np.std(baseline_cosine_similarities),
                    'min': np.min(baseline_cosine_similarities),
                    'max': np.max(baseline_cosine_similarities),
                    'median': np.median(baseline_cosine_similarities)
                },
                'dot_product': {
                    'mean': np.mean(baseline_dot_products),
                    'std': np.std(baseline_dot_products),
                    'min': np.min(baseline_dot_products),
                    'max': np.max(baseline_dot_products),
                    'median': np.median(baseline_dot_products)
                },
                'frobenius_inner_product': {
                    'mean': np.mean(baseline_frobenius_products),
                    'std': np.std(baseline_frobenius_products),
                    'min': np.min(baseline_frobenius_products),
                    'max': np.max(baseline_frobenius_products),
                    'median': np.median(baseline_frobenius_products)
                }
            }
            
            # Add comparison metrics
            analysis['comparison_to_random'] = {
                'cosine_similarity_ratio': analysis['cosine_similarity']['mean'] / analysis['random_baseline']['cosine_similarity']['mean'] if analysis['random_baseline']['cosine_similarity']['mean'] != 0 else float('inf'),
                'is_more_orthogonal_than_random': abs(analysis['cosine_similarity']['mean']) < abs(analysis['random_baseline']['cosine_similarity']['mean'])
            }
        
        return analysis
    
    def visualize_results(self, results: Dict, analysis: Dict, random_baseline: Dict = None, save_path: str = None):
        """Visualize orthogonality results with optional random baseline comparison"""
        if not results:
            print("No results to visualize")
            return
        
        # For now, use simple layout (will enhance later with baseline comparison)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        title = 'LoRA A-Matrix Orthogonality Analysis'
        if random_baseline:
            title += ' (with Random Baseline)'
        fig.suptitle(title, fontsize=16)
        
        # Extract data
        modules = list(results.keys())
        cosine_sims = [results[m]['cosine_similarity'] for m in modules]
        dot_products = [results[m]['dot_product'] for m in modules]
        frobenius_products = [results[m]['frobenius_inner_product'] for m in modules]
        
        # Plot 1: Cosine similarity distribution
        axes[0, 0].hist(cosine_sims, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7, label='Perfect Orthogonality')
        axes[0, 0].set_xlabel('Cosine Similarity')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Cosine Similarity Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Dot product distribution
        axes[0, 1].hist(dot_products, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.7, label='Perfect Orthogonality')
        axes[0, 1].set_xlabel('Dot Product')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Dot Product Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Module-wise cosine similarity
        module_indices = range(len(modules))
        axes[1, 0].scatter(module_indices, cosine_sims, alpha=0.7, color='purple')
        axes[1, 0].axhline(0, color='red', linestyle='--', alpha=0.7, label='Perfect Orthogonality')
        axes[1, 0].set_xlabel('Module Index')
        axes[1, 0].set_ylabel('Cosine Similarity')
        axes[1, 0].set_title('Per-Module Cosine Similarity')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics
        metrics = ['Mean', 'Std', 'Min', 'Max', 'Median']
        cosine_stats = [analysis['cosine_similarity'][k.lower()] for k in metrics]
        
        axes[1, 1].bar(metrics, cosine_stats, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].set_ylabel('Cosine Similarity')
        axes[1, 1].set_title('Summary Statistics')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            # Save with default name if no path provided
            default_path = "lora_orthogonality_visualization.png"
            plt.savefig(default_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {default_path}")
        
        plt.close()  # Close the figure to free memory
    
    def run_orthogonality_test(self, visualize: bool = True, save_results: str = None, compute_baseline: bool = True):
        """
        Run the complete orthogonality test pipeline.
        
        Args:
            visualize: Whether to create visualizations
            save_results: Path to save results (optional)
            compute_baseline: Whether to compute random baseline for comparison
        """
        print("="*60)
        print("LoRA Orthogonality Test - O-LoRA Paper Implementation")
        print("="*60)
        
        # Load adapters
        self.load_adapters()
        print()
        
        # Extract A matrices
        a_matrices_1, a_matrices_2 = self.extract_a_matrices()
        print()
        
        # Compute orthogonality metrics
        print("Computing orthogonality metrics...")
        results = self.compute_orthogonality_metrics(a_matrices_1, a_matrices_2)
        print()
        
        # Compute random baseline if requested
        random_baseline = None
        if compute_baseline:
            print("Computing random baseline for comparison...")
            random_baseline = self.compute_random_baseline(a_matrices_1, a_matrices_2)
            print()
        
        # Analyze results
        analysis = self.analyze_orthogonality(results, random_baseline)
        
        # Print summary
        print("ORTHOGONALITY ANALYSIS SUMMARY")
        print("-" * 40)
        print(f"Number of analyzed modules: {analysis['num_modules']}")
        print(f"Mean cosine similarity: {analysis['cosine_similarity']['mean']:.6f}")
        print(f"Std cosine similarity: {analysis['cosine_similarity']['std']:.6f}")
        print(f"Range: [{analysis['cosine_similarity']['min']:.6f}, {analysis['cosine_similarity']['max']:.6f}]")
        
        if random_baseline:
            print()
            print("RANDOM BASELINE COMPARISON")
            print("-" * 40)
            print(f"Random baseline mean cosine similarity: {analysis['random_baseline']['cosine_similarity']['mean']:.6f}")
            print(f"Random baseline std cosine similarity: {analysis['random_baseline']['cosine_similarity']['std']:.6f}")
            print(f"LoRA vs Random ratio: {analysis['comparison_to_random']['cosine_similarity_ratio']:.3f}")
            print(f"LoRA more orthogonal than random: {analysis['comparison_to_random']['is_more_orthogonal_than_random']}")
        
        print()
        
        # Interpret results
        mean_cosine = analysis['cosine_similarity']['mean']
        if random_baseline:
            baseline_mean = analysis['random_baseline']['cosine_similarity']['mean']
            if abs(mean_cosine) < abs(baseline_mean) * 0.5:
                orthogonality_level = "Excellent orthogonality (much better than random)"
            elif abs(mean_cosine) < abs(baseline_mean):
                orthogonality_level = "Good orthogonality (better than random)"
            elif abs(mean_cosine) < abs(baseline_mean) * 1.5:
                orthogonality_level = "Moderate orthogonality (comparable to random)"
            else:
                orthogonality_level = "Poor orthogonality (worse than random)"
        else:
            # Original interpretation without baseline
            if abs(mean_cosine) < 0.1:
                orthogonality_level = "High orthogonality (close to O-LoRA ideal)"
            elif abs(mean_cosine) < 0.3:
                orthogonality_level = "Moderate orthogonality"
            else:
                orthogonality_level = "Low orthogonality (high correlation)"
        
        print(f"Interpretation: {orthogonality_level}")
        print()
        
        # Detailed results for top/bottom modules
        sorted_modules = sorted(results.items(), key=lambda x: abs(x[1]['cosine_similarity']))
        
        print("TOP 5 MOST ORTHOGONAL MODULES:")
        for i, (module, metrics) in enumerate(sorted_modules[:5]):
            baseline_note = ""
            if random_baseline and module in random_baseline:
                baseline_cosine = random_baseline[module]['cosine_similarity']
                baseline_note = f" (baseline: {baseline_cosine:.6f})"
            print(f"{i+1}. {module}: cosine_sim = {metrics['cosine_similarity']:.6f}{baseline_note}")
        print()
        
        print("TOP 5 LEAST ORTHOGONAL MODULES:")
        for i, (module, metrics) in enumerate(sorted_modules[-5:]):
            baseline_note = ""
            if random_baseline and module in random_baseline:
                baseline_cosine = random_baseline[module]['cosine_similarity']
                baseline_note = f" (baseline: {baseline_cosine:.6f})"
            print(f"{i+1}. {module}: cosine_sim = {metrics['cosine_similarity']:.6f}{baseline_note}")
        print()
        
        # Visualize if requested
        if visualize:
            viz_path = None
            if save_results:
                viz_path = save_results.replace('.json', '_visualization.png')
            self.visualize_results(results, analysis, random_baseline, viz_path)
        
        # Save results if requested
        if save_results:
            # Convert analysis to JSON-serializable format
            def make_json_serializable(obj):
                if isinstance(obj, dict):
                    return {k: make_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (bool, np.bool_)):
                    return bool(obj)  # Convert both Python and numpy booleans
                elif isinstance(obj, (int, float, np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, str):
                    return str(obj)
                elif hasattr(obj, 'item'):  # Handle numpy scalars
                    return obj.item()
                else:
                    return obj
            
            output = {
                'adapter1_path': str(self.adapter1_path),
                'adapter2_path': str(self.adapter2_path),
                'analysis': make_json_serializable(analysis),
                'detailed_results': {k: {kk: make_json_serializable(vv) 
                                       for kk, vv in v.items() if kk != 'shape'} 
                                   for k, v in results.items()}
            }
            
            # Add random baseline results if computed
            if random_baseline:
                output['random_baseline_results'] = {k: {kk: make_json_serializable(vv) 
                                                        for kk, vv in v.items() if kk != 'shape'} 
                                                    for k, v in random_baseline.items()}
            
            with open(save_results, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"Results saved to {save_results}")
        
        return results, analysis, random_baseline


def test_lora_orthogonality(ad_path: str, sci_path: str, visualize: bool = True, save_results: str = None, compute_baseline: bool = True):
    """
    Convenience function to test LoRA orthogonality between AD and Sci adapters.
    
    Args:
        ad_path: Path to AD LoRA adapter directory
        sci_path: Path to Sci LoRA adapter directory  
        visualize: Whether to create visualizations
        save_results: Path to save results JSON file
        compute_baseline: Whether to compute random baseline for comparison
    
    Returns:
        Tuple of (detailed_results, analysis_summary, random_baseline_results)
    """
    tester = LoRAOrthogonalityTester(ad_path, sci_path)
    return tester.run_orthogonality_test(visualize=visualize, save_results=save_results, compute_baseline=compute_baseline)


if __name__ == "__main__":
    # Example usage
    ad_adapter_path = "/pasteur/u/lihe50hz/VLMTrans/working/llava-1.5-7b/lora-sft-cl-align-projector-lr1e-4-warm0.1-rank8/AD"
    sci_adapter_path = "/pasteur/u/lihe50hz/VLMTrans/working/llava-1.5-7b/lora-sft-cl-align-projector-lr1e-4-warm0.1-rank8/Sci"
    
    # Run the orthogonality test with random baseline comparison
    results, analysis, baseline = test_lora_orthogonality(
        ad_adapter_path, 
        sci_adapter_path,
        visualize=True,
        save_results="lora_orthogonality_results.json",
        compute_baseline=True
    )
