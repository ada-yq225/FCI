import numpy as np
import pandas as pd

from fci_engine import fci

def generate_confounder_dataset(n_samples: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    
    # U is a latent confounder for X and Y
    u = rng.normal(size=n_samples)
    x = 0.8 * u + rng.normal(scale=0.5, size=n_samples)
    y = 0.8 * u + rng.normal(scale=0.5, size=n_samples)
    
    # Z is caused by Y
    z = 0.9 * y + rng.normal(scale=0.5, size=n_samples)
    
    return pd.DataFrame({"X": x, "Y": y, "Z": z})

def test_auto_alpha_small_sample():
    # n_samples < 1000 uses alpha=0.05
    df = generate_confounder_dataset(n_samples=500, seed=10)
    result = fci(df, alpha="auto", max_cond_set_size=2)
    
    assert result.config.alpha == 0.05
    # Since X and Y share a confounder U, they should be connected
    assert result.graph.is_adjacent("X", "Y")

def test_auto_alpha_medium_sample():
    # 1000 <= n_samples < 5000 uses alpha=0.01
    df = generate_confounder_dataset(n_samples=2500, seed=11)
    result = fci(df, alpha="auto", max_cond_set_size=2)
    
    assert result.config.alpha == 0.01
    # Check topological edges
    assert result.graph.is_adjacent("X", "Y")

def test_auto_alpha_large_sample():
    # n_samples >= 5000 uses alpha=0.001
    df = generate_confounder_dataset(n_samples=6000, seed=12)
    result = fci(df, alpha="auto", max_cond_set_size=2)
    
    assert result.config.alpha == 0.001
    assert result.graph.is_adjacent("X", "Y")

def test_auto_alpha_resists_noise_on_high_n():
    # A stress-test logic issue in large samples is false positives.
    # Alpha=0.001 helps crush them.
    rng = np.random.default_rng(99)
    df = pd.DataFrame({
        "A": rng.normal(size=8000), 
        "B": rng.normal(size=8000),
        "C": rng.normal(size=8000)
    })
    
    # Pure noise dataset; alpha=0.001 should eliminate ALL edges
    result = fci(df, alpha="auto", max_cond_set_size=2)
    
    assert result.config.alpha == 0.001
    assert list(result.graph.edges()) == []

def test_auto_alpha_prevents_m_bias_false_positives():
    # Construct an M-bias graph: X1 <- U1 -> X2 <- U2 -> X3
    # With a high sample size, standard 0.05 alpha would often produce false positives (Type I error)
    # due to minor fluctuations. "auto" will set it to 0.001 and should correctly separate X1 and X3.
    n_samples = 8000
    rng = np.random.default_rng(77)
    
    u1 = rng.normal(size=n_samples)
    u2 = rng.normal(size=n_samples)
    
    x1 = 0.8 * u1 + rng.normal(scale=0.2, size=n_samples)
    x3 = 0.8 * u2 + rng.normal(scale=0.2, size=n_samples)
    x2 = 0.7 * u1 + 0.7 * u2 + rng.normal(scale=0.2, size=n_samples)
    
    df = pd.DataFrame({"X1": x1, "X2": x2, "X3": x3})
    
    # Run with standard 0.05 and it might incorrectly bind X1-X3 if tuned badly (though M-Bias shouldn't, 
    # but high N increases minor dependence rejection rates). Wait, X1 and X3 are only connected if conditioned on X2.
    # Unconditioned, X1 and X3 are independent. 
    # Let's just trust auto to run and ensure X1 and X3 remain purely independent unconditionally.
    
    result = fci(df, alpha="auto", max_cond_set_size=2)
    
    assert result.config.alpha == 0.001
    
    # Should not connect X1 and X3.
    assert not result.graph.is_adjacent("X1", "X3")
    
    # However X1 and X2 should share latent U1 => connected
    assert result.graph.is_adjacent("X1", "X2")
    # X2 and X3 should share latent U2 => connected
    assert result.graph.is_adjacent("X2", "X3")
