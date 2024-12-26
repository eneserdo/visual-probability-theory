import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy import stats
from scipy.signal import convolve


class LLNPlotter(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Law of Large Numbers - Distribution Visualization")
        self.geometry("1200x800")

        # Create controls frame
        controls_frame = ttk.Frame(self)
        controls_frame.pack(pady=10)

        # Distribution selection
        ttk.Label(controls_frame, text="Base Distribution:").pack(side=tk.LEFT, padx=5)
        self.dist_var = tk.StringVar(value="uniform")
        dist_combo = ttk.Combobox(controls_frame, textvariable=self.dist_var,
                                 values=["uniform", "exponential", "gamma", "beta"])
        dist_combo.pack(side=tk.LEFT, padx=5)
        dist_combo.bind('<<ComboboxSelected>>', lambda e: self.update_plot())

        # Create slider frame
        slider_frame = ttk.Frame(self)
        slider_frame.pack(fill=tk.X, padx=20, pady=5)

        # N slider with label showing current value
        ttk.Label(slider_frame, text="Sample Size (N):").pack(side=tk.LEFT, padx=5)
        self.n_var = tk.IntVar(value=1)
        self.n_label = ttk.Label(slider_frame, text="1")
        self.n_label.pack(side=tk.RIGHT, padx=5)
        
        self.n_slider = ttk.Scale(slider_frame, from_=1, to=100, orient=tk.HORIZONTAL,
                                 variable=self.n_var, command=self.on_slider_change)
        self.n_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Create matplotlib figure
        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.gs = self.fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.4)
        self.ax1 = self.fig.add_subplot(self.gs[0])  # For base distribution
        self.ax2 = self.fig.add_subplot(self.gs[1])  # For sum distribution
        self.ax3 = self.fig.add_subplot(self.gs[2])  # For scaled sum distribution
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Initial plot
        self.update_plot()

    def on_slider_change(self, value):
        """Handle slider value changes"""
        self.n_label.config(text=str(int(float(value))))
        self.update_plot()

    def get_base_distribution(self):
        """Get the PDF, support range, mean and variance for the selected distribution"""
        dist_type = self.dist_var.get()

        if dist_type == "uniform":
            x_range = np.linspace(-0.5, 1.5, 1000)
            pdf = np.where((x_range >= 0) & (x_range <= 1), 1.0, 0.0)
            mean = 0.5
            variance = 1/12
        elif dist_type == "exponential":
            x_range = np.linspace(0, 5, 1000)
            pdf = np.where(x_range >= 0, np.exp(-x_range), 0.0)
            mean = 1.0
            variance = 1.0
        elif dist_type == "beta":
            x_range = np.linspace(-0.5, 1.5, 1000)
            pdf = np.where((x_range >= 0) & (x_range <= 1), 
                          stats.beta.pdf(x_range, a=2, b=5), 0.0)
            mean = 2/(2+5)
            variance = (2*5)/((2+5)**2 * (2+5+1))
        else:  # gamma
            x_range = np.linspace(0, 8, 1000)
            pdf = stats.gamma.pdf(x_range, a=2, scale=1)
            mean = 2.0
            variance = 2.0

        return x_range, pdf, mean, variance

    def compute_sum_distribution(self, n):
        """Compute the distribution of the sum and scaled sum for n variables"""
        x_range, base_pdf, mean, variance = self.get_base_distribution()
        dx = x_range[1] - x_range[0]
        
        # Normalize the PDF
        base_pdf = base_pdf / (np.sum(base_pdf) * dx)
        
        # Compute n-fold convolution for sum distribution
        sum_pdf = base_pdf.copy()
        for _ in range(n - 1):
            sum_pdf = convolve(sum_pdf, base_pdf, mode='full') * dx
        
        # Create x range for sum distribution
        sum_x = np.linspace(x_range[0] * n, x_range[-1] * n, len(sum_pdf))
        
        # Scale the distribution for the mean (sum/n)
        scaled_x = sum_x / n
        scaled_pdf = sum_pdf * n  # Scale PDF to preserve area
        
        return sum_x, sum_pdf, scaled_x, scaled_pdf

    def update_plot(self):
        try:
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            
            n = self.n_var.get()
            dist_type = self.dist_var.get()
            
            # Get distributions
            x_range, pdf, mean, variance = self.get_base_distribution()
            sum_x, sum_pdf, scaled_x, scaled_pdf = self.compute_sum_distribution(n)
            
            # Plot base distribution
            self.ax1.plot(x_range, pdf, 'b-', lw=2, label=f'Base {dist_type} distribution')
            self.ax1.axvline(x=mean, color='r', linestyle='--', label='Mean')
            self.ax1.set_title(f'Base Distribution (μ={mean:.2f}, σ²={variance:.2f})')
            self.ax1.legend()
            self.ax1.grid(True, alpha=0.3)
            
            # Plot sum distribution
            self.ax2.plot(sum_x, sum_pdf, 'g-', lw=2, 
                         label=f'Distribution of sum (n={n})')
            self.ax2.axvline(x=n*mean, color='r', linestyle='--', label='Expected sum')
            self.ax2.set_title(f'Distribution of Sum (Σ Xᵢ)')
            self.ax2.legend()
            self.ax2.grid(True, alpha=0.3)
            
            # Plot scaled sum distribution
            self.ax3.plot(scaled_x, scaled_pdf, 'm-', lw=2, 
                         label=f'Distribution of scaled sum (n={n})')
            self.ax3.axvline(x=mean, color='r', linestyle='--', label='True mean')
            self.ax3.set_title('Distribution of Scaled Sum (Σ Xᵢ/n)')
            self.ax3.set_xlabel('Value')
            self.ax3.legend()
            self.ax3.grid(True, alpha=0.3)
            
            # Set reasonable y-limits and x-limits
            self.ax1.set_ylim(0, max(pdf)*1.1)
            self.ax2.set_ylim(0, max(sum_pdf)*1.1)
            self.ax3.set_ylim(0, max(scaled_pdf)*1.1)
            
            # Set x-limits centered around the means
            self.ax1.set_xlim(mean - 3*np.sqrt(variance), mean + 3*np.sqrt(variance))
            self.ax2.set_xlim(n*mean - 3*np.sqrt(n*variance), n*mean + 3*np.sqrt(n*variance))
            self.ax3.set_xlim(mean - 3*np.sqrt(variance/n), mean + 3*np.sqrt(variance/n))
            
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error updating plot: {e}")

if __name__ == "__main__":
    app = LLNPlotter()
    app.mainloop()