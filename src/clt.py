import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy import stats
from scipy.signal import convolve

# TODO: to make it consistent with the rest of the project, do not inherit and take master as an argument
class CLTPlotter(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Central Limit Theorem Visualization")
        self.geometry("1000x700")

        # Create controls frame
        controls_frame = ttk.Frame(self)
        controls_frame.pack(pady=10)

        # Mode selection
        ttk.Label(controls_frame, text="Mode:").pack(side=tk.LEFT, padx=5)
        self.mode_var = tk.StringVar(value="sampling")
        mode_combo = ttk.Combobox(controls_frame, textvariable=self.mode_var,
                                 values=["sampling", "distribution"])
        mode_combo.pack(side=tk.LEFT, padx=5)
        mode_combo.bind('<<ComboboxSelected>>', lambda e: self.reset_and_update())

        # Distribution selection
        ttk.Label(controls_frame, text="Base Distribution:").pack(side=tk.LEFT, padx=5)
        self.dist_var = tk.StringVar(value="uniform")
        dist_combo = ttk.Combobox(controls_frame, textvariable=self.dist_var,
                                 values=["bernoulli", "binomial", "uniform", "exponential", "poisson"])
        dist_combo.pack(side=tk.LEFT, padx=5)
        dist_combo.bind('<<ComboboxSelected>>', lambda e: self.reset_and_update())

        # Sample count control (only for sampling mode)
        ttk.Label(controls_frame, text="Sample Size:").pack(side=tk.LEFT, padx=5)
        self.sample_size_var = tk.StringVar(value="10000")
        sample_entry = ttk.Entry(controls_frame, textvariable=self.sample_size_var, width=8)
        sample_entry.pack(side=tk.LEFT, padx=5)
        self.sample_size_var.trace_add("write", lambda *args: self.update_plot())

        # Resolution control
        ttk.Label(controls_frame, text="Resolution:").pack(side=tk.LEFT, padx=5)
        self.resolution_var = tk.StringVar(value="10000")
        resolution_entry = ttk.Entry(controls_frame, textvariable=self.resolution_var, width=8)
        resolution_entry.pack(side=tk.LEFT, padx=5)
        self.resolution_var.trace_add("write", lambda *args: self.update_plot())

        # Create slider frame
        slider_frame = ttk.Frame(self)
        slider_frame.pack(fill=tk.X, padx=20, pady=5)

        # N slider with label showing current value
        ttk.Label(slider_frame, text="Number of Variables (N):").pack(side=tk.LEFT, padx=5)
        self.n_var = tk.IntVar(value=1)
        self.n_label = ttk.Label(slider_frame, text="1")
        self.n_label.pack(side=tk.RIGHT, padx=5)
        
        self.n_slider = ttk.Scale(slider_frame, from_=1, to=100, orient=tk.HORIZONTAL,
                                 variable=self.n_var, command=self.on_slider_change)
        self.n_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Initial plot
        self.update_plot()

    def reset_and_update(self):
        """Reset N to 1 and update plot when mode or distribution changes"""
        self.n_var.set(1)
        self.n_label.config(text="1")
        self.update_plot()

    def on_slider_change(self, value):
        """Handle slider value changes"""
        self.n_label.config(text=str(int(float(value))))
        self.update_plot()

    def get_base_distribution(self):
        """Get the PDF and support range for the selected base distribution"""
        try:
            resolution = int(self.resolution_var.get())
        except ValueError:
            resolution = 1000
            self.resolution_var.set("1000")
            
        x = np.linspace(-10, 10, resolution)
        dist_type = self.dist_var.get()

        if dist_type == "bernoulli":
            # Bernoulli distribution with p=0.5
            pdf = np.zeros_like(x)
            pdf[np.abs(x) < 0.1] = 0.5  # Approximate delta functions at 0 and 1
            pdf[np.abs(x-1) < 0.1] = 0.5
            x_range = np.linspace(-0.5, 1.5, resolution)
        elif dist_type == "binomial":
            # Binomial distribution with n=10, p=0.5
            x_range = np.linspace(-1, 12, resolution)
            pdf = stats.binom.pmf(np.round(x_range), n=10, p=0.5)
        elif dist_type == "uniform":
            # Uniform distribution on [0,1]
            pdf = np.where((x >= 0) & (x <= 1), 1.0, 0.0)
            x_range = np.linspace(-0.5, 1.5, resolution)
        elif dist_type == "exponential":
            # Exponential distribution with rate 1
            pdf = np.where(x >= 0, np.exp(-x), 0.0)
            x_range = np.linspace(0, 5, resolution)
        else:  # poisson
            # Poisson distribution with lambda=2
            x_range = np.linspace(-1, 10, resolution)
            pdf = stats.poisson.pmf(np.round(x_range), mu=2)

        return x_range, pdf

    def convolve_distributions(self, n):
        """Compute the n-fold convolution of the base distribution"""
        x_range, base_pdf = self.get_base_distribution()
        dx = x_range[1] - x_range[0]
        
        # Normalize the PDF
        base_pdf = base_pdf / (np.sum(base_pdf) * dx)
        
        # Perform n-fold convolution
        result_pdf = base_pdf.copy()
        for _ in range(n - 1):
            result_pdf = convolve(result_pdf, base_pdf, mode='full') * dx
            
        # Adjust x-range for the convolution
        new_x = np.linspace(x_range[0] * n, x_range[-1] * n, len(result_pdf))
        
        # Standardize the distribution
        mean = np.sum(new_x * result_pdf) * dx
        variance = np.sum((new_x - mean)**2 * result_pdf) * dx
        std = np.sqrt(variance)
        
        standardized_x = (new_x - mean) / std
        result_pdf = result_pdf * std  # Preserve area under curve
        
        return standardized_x, result_pdf

    def generate_samples(self):
        """Generate samples for the sampling-based approach"""
        n = self.n_var.get()
        try:
            sample_size = int(self.sample_size_var.get())
        except ValueError:
            sample_size = 1000  # Default if invalid input
            self.sample_size_var.set("1000")

        dist_type = self.dist_var.get()

        if dist_type == "bernoulli":
            samples = np.random.binomial(1, 0.5, (sample_size, n))
        elif dist_type == "binomial":
            samples = np.random.binomial(10, 0.5, (sample_size, n))
        elif dist_type == "uniform":
            samples = np.random.uniform(0, 1, (sample_size, n))
        elif dist_type == "exponential":
            samples = np.random.exponential(1, (sample_size, n))
        else:  # poisson
            samples = np.random.poisson(2, (sample_size, n))

        sums = np.sum(samples, axis=1)
        standardized_sums = (sums - np.mean(sums)) / np.std(sums)
        return standardized_sums

    def update_plot(self):
        try:
            self.ax.clear()
            
            mode = self.mode_var.get()
            n = self.n_var.get()
            
            try:
                resolution = int(self.resolution_var.get())
            except ValueError:
                resolution = 1000
                self.resolution_var.set("1000")
            
            if mode == "sampling":
                # Sampling mode
                standardized_sums = self.generate_samples()
                self.ax.hist(standardized_sums, bins=50, density=True, alpha=0.7,
                            label=f'Sum of {n} {self.dist_var.get()} RVs')
                plot_title = f'CLT Demonstration (Sampling)\nSum of {n} {self.dist_var.get()} random variables'
            else:
                # Distribution mode
                x, pdf = self.convolve_distributions(n)
                self.ax.plot(x, pdf, 'b-', lw=2,
                            label=f'PDF of sum of {n} {self.dist_var.get()} RVs')
                plot_title = f'CLT Demonstration (Distribution)\nPDF of sum of {n} {self.dist_var.get()} random variables'

            # Plot standard normal PDF for comparison
            x = np.linspace(-4, 4, resolution)
            self.ax.plot(x, stats.norm.pdf(x), 'r--', lw=2,
                        label='Standard Normal PDF')

            self.ax.set_title(plot_title)
            self.ax.set_xlabel('Standardized Value')
            self.ax.set_ylabel('Density')
            self.ax.legend()
            self.ax.grid(True, alpha=0.3)
            self.ax.set_xlim(-4, 4)
            
            self.canvas.draw()
        except Exception as e:
            print(f"Error updating plot: {e}")

if __name__ == "__main__":
    app = CLTPlotter()
    app.mainloop()