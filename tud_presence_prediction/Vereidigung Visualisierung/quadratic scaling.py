import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --- Configuration ---
OUTPUT_DIR = "visualizations"
FILENAME = "transformer_quadratic_scaling.png"

# Colors
COLOR_QUADRATIC = "#1F77B4"  # Blue for Standard Transformer
COLOR_FLASH = "#8B3E8C" # Purple for flash Transformer
COLOR_AXIAL = "#299912"     # Green for axial Transformer
COLOR_LONG_CONTEXT = "#FF7F0E" # Orange for long context curve

def plot_scaling_comparison():
    """
    Creates and saves a plot comparing the quadratic compute scaling of
    standard Transformers with a more efficient approach.
    """
    # X-axis: Context Length (powers of 2 are common)
    context_lengths = np.arange(1000, 10001, 1000)

    # Standard Transformer (should be approximately quadratic: O(n^2))
    # Normalize the compute times so the first data point starts at 1
    base_compute_times = (context_lengths / context_lengths[0])**2

    # # Efficient Transformer (should be closer to linear or n*log(n))
    # noise_factor_eff = 1 + np.random.uniform(-0.05, 0.05, size=len(context_lengths))
    # efficient_compute_times = efficient_scaling_factor * context_lengths * np.log2(context_lengths) * noise_factor_eff

    # --- Create Plot ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot data points and lines
    ax.plot(context_lengths, base_compute_times, 'o-', color=COLOR_QUADRATIC,
            label="Standard Transformer O(n^2)", markersize=8, linewidth=2.5)

    # Configure axes and title
    ax.set_xlabel("Context Length (Number of Tokens)", fontsize=12)
    ax.set_ylabel("Relative Compute Time", fontsize=12)
    ax.set_title("Computational Cost vs. Context Length in Transformers", fontsize=14, fontweight='bold')

    # Format axis ticks to improve readability
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xticks(context_lengths) # Ensure our data points are ticks
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Add legend
    ax.legend(loc="upper left", frameon=True, shadow=True, fontsize=11)

    # Save the plot
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outfile = os.path.join(OUTPUT_DIR, FILENAME)
    plt.show()
    # plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Plot saved at: {os.path.abspath(outfile)}")

def main():
    """Main function to run the plot creation."""
    plot_scaling_comparison()

if __name__ == "__main__":
    main()