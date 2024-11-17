import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import multivariate_normal
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk
from numpy import trapezoid

class CustomNavigationToolbar(NavigationToolbar2Tk):
    def __init__(self, canvas, window):
        super().__init__(canvas, window)
        
    def pack(self, *args, **kwargs):
        # Prevent the pack geometry manager from being used
        pass

class ConditionalPlotApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Interactive Probability Distributions")
        
        # Create main frame
        self.main_frame = ttk.Frame(self.master, padding="10")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(0, weight=1)
        self.main_frame.rowconfigure(1, weight=0)
        self.main_frame.rowconfigure(2, weight=0)
        
        # Create matplotlib figure with subplots
        self.fig = plt.figure(figsize=(12, 6))
        self.gs = self.fig.add_gridspec(3, 2, width_ratios=[2, 1])
        
        # Create subplots
        self.ax_3d = self.fig.add_subplot(self.gs[:, 0], projection='3d')
        self.ax_conditional = self.fig.add_subplot(self.gs[0, 1])
        self.ax_joint = self.fig.add_subplot(self.gs[1, 1])
        self.ax_marginal = self.fig.add_subplot(self.gs[2, 1])
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        
        # Add toolbar
        self.toolbar = CustomNavigationToolbar(self.canvas, self.main_frame)
        self.toolbar.grid(row=1, column=0, columnspan=2, sticky=tk.W)
        
        # Create slider frame
        self.slider_frame = ttk.Frame(self.main_frame, padding="5")
        self.slider_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # Create y-value slider
        ttk.Label(self.slider_frame, text="Y value:").grid(row=0, column=0, padx=5)
        self.y_value = tk.DoubleVar(value=0.0)
        self.y_slider = ttk.Scale(
            self.slider_frame,
            from_=-3.0,
            to=3.0,
            orient=tk.HORIZONTAL,
            length=300,
            variable=self.y_value,
            command=self.update_plot
        )
        self.y_slider.grid(row=0, column=1, padx=5)
        
        # Create initial plot
        self.create_plot()
        
        # Configure grid weights
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(0, weight=1)
        
        # Set initial window size
        self.master.geometry("1200x800")
        self.master.attributes('-zoomed', True)

    def create_plot(self):
        # Clear all axes
        self.ax_3d.clear()
        self.ax_conditional.clear()
        self.ax_joint.clear()
        self.ax_marginal.clear()
        
        # Create grid of points
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)
        
        # Define parameters for multiple bivariate normal distributions
        means = [[1, 1], [-1, -1]]  # Centers of the two modes
        covs = [[[0.5, 0.2], [0.2, 0.5]],  # Covariance for first mode
                [[0.5, -0.2], [-0.2, 0.5]]]  # Covariance for second mode
        weights = [0.5, 0.5]  # Equal weights for both modes
        
        # Create the mixture of bivariate normal distributions
        pos = np.dstack((X, Y))
        Z = np.zeros_like(X)
        for mean, cov, weight in zip(means, covs, weights):
            rv = multivariate_normal(mean, cov)
            Z += weight * rv.pdf(pos)
        
        # Plot the 3D surface
        self.surf = self.ax_3d.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.7)
        
        # Store the data for later use
        self.x, self.y, self.Z = x, y, Z
        
        # Calculate marginal distribution p(X)
        self.marginal_x = np.sum(Z, axis=0) * (y[1] - y[0])  # Numerical integration
        
        # Update all plots
        self.update_all_plots()
        
        # Customize 3D plot
        self.ax_3d.set_xlabel('x')
        self.ax_3d.set_ylabel('y')
        self.ax_3d.set_zlabel('density')
        self.ax_3d.view_init(elev=20, azim=-45)
        self.ax_3d.set_title('3D Joint Distribution')
        
        # Set consistent axis limits for 3D plot
        self.ax_3d.set_xlim(-3, 3)
        self.ax_3d.set_ylim(-3, 3)
        self.ax_3d.set_zlim(0, np.max(Z) * 1.1)

    def update_all_plots(self):
        y_slice = self.y_value.get()
        idx_y = np.abs(self.y - y_slice).argmin()
        
        # Update 3D plot lines
        if hasattr(self, 'joint_line'):
            self.joint_line.remove()
        if hasattr(self, 'conditional_line'):
            self.conditional_line.remove()
        
        # Plot 3D slices
        self.joint_line, = self.ax_3d.plot(
            self.x, 
            [y_slice] * len(self.x), 
            self.Z[:, idx_y], 
            color='blue', 
            linewidth=2, 
            label='p(X,Y=y)'
        )
        
        conditional = self.Z[:, idx_y] / np.max(self.Z[:, idx_y]) * np.max(self.Z)
        self.conditional_line, = self.ax_3d.plot(
            self.x, 
            [y_slice] * len(self.x), 
            conditional, 
            color='red', 
            linewidth=2, 
            label='p(X|Y=y)'
        )
        
        # Update 2D plots
        self.ax_conditional.clear()
        self.ax_joint.clear()
        self.ax_marginal.clear()
        
        # Plot conditional probability p(X|Y=y)
        conditional_normalized = self.Z[:, idx_y] / trapezoid(self.Z[:, idx_y], self.x)
        self.ax_conditional.plot(self.x, conditional_normalized, 'r-', linewidth=2)
        self.ax_conditional.set_title(f'p(X|Y={y_slice:.2f})')
        self.ax_conditional.set_xlim(-3, 3)
        self.ax_conditional.set_ylim(0, 1)
        
        # Plot joint probability p(X,Y=y)
        self.ax_joint.plot(self.x, self.Z[:, idx_y], 'b-', linewidth=2)
        self.ax_joint.set_title(f'p(X,Y={y_slice:.2f})')
        self.ax_joint.set_xlim(-3, 3)
        self.ax_joint.set_ylim(0, 1)
        
        # Plot marginal probability p(X)
        self.ax_marginal.plot(self.x, self.marginal_x, 'g-', linewidth=2)
        self.ax_marginal.set_title('p(X)')
        self.ax_marginal.set_xlim(-3, 3)
        self.ax_marginal.set_ylim(0, 1)
        
        # Add legend to 3D plot
        self.ax_3d.legend()
        
        # Adjust layout
        self.fig.tight_layout()

    def update_plot(self, *args):
        self.update_all_plots()
        self.canvas.draw()

def main():
    root = tk.Tk()
    app = ConditionalPlotApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()