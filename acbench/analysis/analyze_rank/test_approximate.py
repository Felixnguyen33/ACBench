import torch
import math

def compute_erank_both_ways(R):
    """Compute effective rank using both covariance and Gram matrix methods"""
    # Method 1: Using covariance matrix R.T@R
    A = R.T @ R
    eigvals_A = torch.linalg.eigvalsh(A)
    eigvals_A = eigvals_A[eigvals_A > 1e-8]  # Filter near-zero values
    p_A = eigvals_A / eigvals_A.sum()
    entropy_A = -(p_A * torch.log(p_A)).sum().item()
    erank_A = math.exp(entropy_A)

    # Method 2: Using Gram matrix R@R.T 
    B = R @ R.T
    eigvals_B = torch.linalg.eigvalsh(B)
    eigvals_B = eigvals_B[eigvals_B > 1e-8]  # Filter near-zero values
    p_B = eigvals_B / eigvals_B.sum()
    entropy_B = -(p_B * torch.log(p_B)).sum().item()
    erank_B = math.exp(entropy_B)

    return erank_A, erank_B

# Test case 1: Random matrix with your exact dimensions (7 tokens)
R = torch.randn(7, 151936)  # Too big for covariance method, but let's try small version
small_R = R[:, :5000]  # Use smaller dimension for practical computation (7x100)
erank_A, erank_B = compute_erank_both_ways(small_R)

print(f"Small matrix test (7x100):")
print(f"Effective rank via covariance matrix: {erank_A:.6f}")
print(f"Effective rank via Gram matrix:       {erank_B:.6f}")
print(f"Absolute difference: {abs(erank_A - erank_B):.2e}")

# Test case 2: Full rank matrix
full_rank_R = torch.tensor([[2, 1, 0], [1, 2, 1], [0, 1, 2]], dtype=torch.float32).T
erank_A, erank_B = compute_erank_both_ways(full_rank_R)

print("\nFull rank matrix test (3x3):")
print(f"Effective rank via covariance matrix: {erank_A:.6f}")
print(f"Effective rank via Gram matrix:       {erank_B:.6f}")

# Test case 3: Low rank matrix
low_rank_R = torch.tensor([[1, 0], [0, 1], [0, 0]], dtype=torch.float32).T
erank_A, erank_B = compute_erank_both_ways(low_rank_R)

print("\nLow rank matrix test (3x2):")
print(f"Effective rank via covariance matrix: {erank_A:.6f}")
print(f"Effective rank via Gram matrix:       {erank_B:.6f}")