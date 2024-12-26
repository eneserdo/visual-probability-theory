import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import multivariate_normal
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk

class ProbabilityPlotApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Marginal Probability Visualization")

        self.geometry("800x1000")
        self.bind('<q>', self.quit_application)
        self.bind('<s>', self.start_animation)
        self.bind('<r>', self.reset_animation)
        
        # Create main frame
        self.main_frame = ttk.Frame(self, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create matplotlib figure with subplots
        self.fig = plt.figure(figsize=(20, 8))
        self.gs = self.fig.add_gridspec(1, 2, width_ratios=[1, 1])
        
        # Create subplots
        self.ax_3d = self.fig.add_subplot(self.gs[0, 0], projection='3d')
        self.ax_marginal = self.fig.add_subplot(self.gs[0, 1])
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create a frame for the toolbar
        self.toolbar_frame = ttk.Frame(self.main_frame)
        self.toolbar_frame.grid(row=1, column=0, columnspan=2, sticky=tk.W)
        
        # Add toolbar to the toolbar frame
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        
        # Create control frame
        self.control_frame = ttk.Frame(self.main_frame, padding="5")
        self.control_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # Create start/stop button
        self.running = False
        self.start_button = ttk.Button(
            self.control_frame,
            text="Start Animation",
            command=self.toggle_animation
        )
        self.start_button.grid(row=0, column=0, padx=5)
        
        # Create reset button
        self.reset_button = ttk.Button(
            self.control_frame,
            text="Reset",
            command=self.reset_animation
        )
        self.reset_button.grid(row=0, column=1, padx=5)
        
        # Initialize the plot
        self.create_plot()
        
        # Configure grid weights
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(0, weight=1)

    def create_plot(self):
        # Clear axes
        self.ax_3d.clear()
        self.ax_marginal.clear()
        
        # Create grid of points
        self.x = np.linspace(-3, 3, 100)
        self.y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(self.x, self.y)
        
        # Define parameters for bivariate normal distribution
        mean = [0, 0]
        cov = [[1, 0.5], [0.5, 1]]
        
        # Create the bivariate normal distribution
        pos = np.dstack((X, Y))
        rv = multivariate_normal(mean, cov)
        self.Z = rv.pdf(pos)
        
        # Plot the 3D surface
        self.surf = self.ax_3d.plot_surface(X, Y, self.Z, cmap=cm.viridis, alpha=0.7)
        
        # Initialize marginal plot
        self.current_sum = np.zeros_like(self.x)
        self.marginal_line, = self.ax_marginal.plot(self.x, self.current_sum, 'b-', linewidth=2)
        
        # Add slice line and contribution area
        self.current_y_idx = 0
        self.slice_line = None
        self.contribution_line = None
        
        # Customize plots
        self.ax_3d.set_xlabel('x')
        self.ax_3d.set_ylabel('y')
        self.ax_3d.set_zlabel('density')
        self.ax_3d.view_init(elev=20, azim=-45)
        self.ax_3d.set_title('Joint Distribution p(X,Y)')
        
        self.ax_marginal.set_xlabel('x')
        self.ax_marginal.set_ylabel('density')
        self.ax_marginal.set_title('Marginal Distribution p(X)')
        
        # Set axis limits
        self.ax_3d.set_xlim(-3, 3)
        self.ax_3d.set_ylim(-3, 3)
        self.ax_3d.set_zlim(0, np.max(self.Z) * 1.1)
        
        max_marginal = np.sum(self.Z, axis=0) * (self.y[1] - self.y[0])
        self.ax_marginal.set_xlim(-3, 3)
        self.ax_marginal.set_ylim(0, np.max(max_marginal) * 1.1)
        
        # Add grid
        self.ax_marginal.grid(True, linestyle='--', alpha=0.6)
        
        self.fig.tight_layout()

    def update_animation(self):
        if not self.running:
            return
            
        # Remove previous slice line
        if hasattr(self, 'slice_line') and self.slice_line:
            for line in self.slice_line:  # slice_line is a tuple of Line objects
                line.remove()
        
        # Remove previous contribution line
        if hasattr(self, 'contribution_line') and self.contribution_line:
            self.contribution_line.remove()
            
        # Plot current slice
        y_val = self.y[self.current_y_idx]
        self.slice_line = self.ax_3d.plot(
            self.x, 
            [y_val] * len(self.x),
            self.Z[self.current_y_idx, :],
            'r-',
            linewidth=2,
            label='Current Slice'
        )
        
        # Update running sum
        dy = self.y[1] - self.y[0]
        self.current_sum += self.Z[self.current_y_idx, :] * dy
        
        # Update marginal plot
        self.marginal_line.set_ydata(self.current_sum)
        
        # Plot current contribution
        self.contribution_line, = self.ax_marginal.plot(
            self.x,
            self.Z[self.current_y_idx, :] * dy,
            'r-',
            alpha=0.3,
            label='Current Contribution'
        )
        
        # Update canvas
        self.canvas.draw()
        
        # Increment counter
        self.current_y_idx += 1
        
        # Check if animation should continue
        if self.current_y_idx < len(self.y):
            self.master.after(50, self.update_animation)  # Increased delay to 50ms
        else:
            self.running = False
            self.start_button.config(text="Start Animation")

    def toggle_animation(self):
        if self.running:
            self.running = False
            self.start_button.config(text="Start Animation")
        else:
            if self.current_y_idx >= len(self.y):
                self.reset_animation()
            self.running = True
            self.start_button.config(text="Stop Animation")
            self.update_animation()

    def reset_animation(self):
        self.running = False
        self.start_button.config(text="Start Animation")
        self.current_y_idx = 0
        self.current_sum = np.zeros_like(self.x)
        self.create_plot()


if __name__ == "__main__":
    app = ProbabilityPlotApp()
    app.mainloop()