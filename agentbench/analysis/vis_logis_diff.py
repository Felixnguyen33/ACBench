import numpy as np 
import matplotlib.pyplot as plt 


# Load the logits from the numpy files
logits_after_quant = np.load("./logits_after_quant.npy")
logits_before_quant = np.load("./logits_before_quant.npy")

# Calculate the difference between the logits
logits_diff = logits_after_quant - logits_before_quant

N = logits_diff.shape[0]
# Plot 5 subfigures for the differences
plt.figure(figsize=(9, 15))
cmap = plt.get_cmap('viridis')
num_tokens = logits_diff.shape[1] // 100
for i in range(N):
    plt.subplot(N, 1, i+1)
    colors = cmap(np.linspace(0, 1, num_tokens))  # Generate colors for all tokens from the viridis colormap
    plt.bar(range(num_tokens), logits_diff[i][:num_tokens], color=colors)
    # plt.xlabel('Token Index')
    plt.ylabel(f'Position{i}')
    # plt.title(f'Logits Difference Visualization for Sequence Position {i}')
    plt.grid(True)

plt.savefig("./logits_difference_distribution.png")

from scipy.stats import spearmanr, kendalltau

# Calculate Spearman's rank correlation and Kendall's tau for each sequence position
spearman_correlations = []
kendall_taus = []
top_k = [5, 10, 20, 50, 80, 100]  # Define the top-k values

# Initialize lists to store top-k correlations for each sequence position
top_k_spearman_correlations = {k: [] for k in top_k}
top_k_kendall_taus = {k: [] for k in top_k}

for i in range(N):
    # Flatten the logits for the current sequence position
    logits_after_flat = logits_after_quant[i].flatten()
    logits_before_flat = logits_before_quant[i].flatten()
    
    # Calculate Spearman's rank correlation
    spearman_corr, _ = spearmanr(logits_after_flat, logits_before_flat)
    spearman_correlations.append(spearman_corr)
    
    # Calculate Kendall's tau
    kendall_tau, _ = kendalltau(logits_after_flat, logits_before_flat)
    kendall_taus.append(kendall_tau)
    
    # Calculate top-k Spearman's rank correlation and Kendall's tau
    for k in top_k:
        top_k_indices = np.argsort(logits_after_flat)[-k:]
        top_k_spearman_corr, _ = spearmanr(logits_after_flat[top_k_indices], logits_before_flat[top_k_indices])
        top_k_kendall_tau, _ = kendalltau(logits_after_flat[top_k_indices], logits_before_flat[top_k_indices])
        
        top_k_spearman_correlations[k].append(top_k_spearman_corr)
        top_k_kendall_taus[k].append(top_k_kendall_tau)
    
    # Print the results
    print(f"Sequence Position {i}: Spearman's Rank Correlation = {spearman_corr}, Kendall's Tau = {kendall_tau}")
    for k in top_k:
        print(f"Sequence Position {i}: Top-{k} Spearman's Rank Correlation = {top_k_spearman_correlations[k][-1]}, Top-{k} Kendall's Tau = {top_k_kendall_taus[k][-1]}")

# Plotting the results
plt.figure(figsize=(15, 6))

# Plot for Spearman's Rank Correlation
plt.subplot(2, 1, 1)
plt.plot(spearman_correlations, label='Spearman\'s Rank Correlation')
for k in top_k:
    plt.plot(top_k_spearman_correlations[k], label=f'Top-{k} Spearman\'s Rank Correlation')
plt.xlabel('Sequence Position')
plt.ylabel('Spearman\'s Rank Correlation')
plt.title('Spearman\'s Rank Correlation')
plt.legend()

# Plot for Kendall's Tau
plt.subplot(2, 1, 2)
plt.plot(kendall_taus, label='Kendall\'s Tau')
for k in top_k:
    plt.plot(top_k_kendall_taus[k], label=f'Top-{k} Kendall\'s Tau')
plt.xlabel('Sequence Position')
plt.ylabel('Kendall\'s Tau')
plt.title('Kendall\'s Tau')
plt.legend()

plt.tight_layout()
plt.grid('--')
plt.savefig("./correlation_distribution.png")