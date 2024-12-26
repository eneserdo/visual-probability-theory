import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Constants for x-axis range


class ConvolutionPlotter(tk.Tk):
    def __init__(self):
        super().__init__()

        # You can change these signals to see how they affect the convolution
        signal1_dict= {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25} # will be still
        signal2_dict= {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4} # will be reversed and shifted

        # find the limits of the domain
        extra_space = 3
        self.max_x = np.max(list(signal1_dict.keys())) + np.max(list(signal2_dict.keys())) + extra_space
        self.min_x = -np.max(list(signal2_dict.keys())) - extra_space
        self.title("Convolution of two discrete pdf")

        assert self.min_x < 0, "min_x must be less than 0, min/max calculation might be wrong, set them manually"
        assert self.max_x > 0, "max_x must be greater than 0, min/max calculation might be wrong, set them manually"

        # Add key binding for 'q'
        self.bind('<q>', self.quit_application)
        self.bind('<r>', self.reset_shift)
        self.bind('<d>', self.increase_shift)
        self.bind('<a>', self.decrease_shift)
        
        
        # Initialize signals
        self.domain = np.arange(self.min_x, self.max_x + 1)
        self.signal1 = np.array([signal1_dict.get(x, 0) for x in self.domain])
        self.signal2 = np.array([signal2_dict.get(x, 0) for x in self.domain])
        
        self.shift = 0
        self.third_plot_dict = {d:0 for d in self.domain}

        # Create main container
        main_container = ttk.Frame(self)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Create left panel for plots
        plot_panel = ttk.Frame(main_container)
        plot_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create right panel for explanation
        explanation_panel = ttk.Frame(main_container)
        explanation_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        
        # Create text widget for explanation
        self.explanation_text = tk.Text(explanation_panel, width=40, wrap=tk.WORD, font=('TkDefaultFont', 9))
        self.explanation_text.pack(fill=tk.BOTH, expand=True)

        # Create matplotlib figure (move to plot_panel)
        self.fig = Figure(figsize=(10, 8))
        self.ax1 = self.fig.add_subplot(411) # PDF of X
        self.ax2 = self.fig.add_subplot(412) # PDF of Y
        self.ax3 = self.fig.add_subplot(413) # PDF of Y (reversed and shifted by shift)
        self.ax4 = self.fig.add_subplot(414) # PDF of Z (convolution of X and Y)

        # Create canvas (update master to plot_panel)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_panel)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Initial plot
        self.init_plots()
        self.update_plots()

    def reset_shift(self, event=None):
        self.shift = 0
        self.third_plot_dict = {d:0 for d in self.domain}
        self.init_plots()
        self.update_plots()

    def increase_shift(self, event=None):
        self.shift += 1
        self.shift = min(self.shift, self.max_x)
        self.update_plots()

    def decrease_shift(self, event=None):
        self.third_plot_dict[self.shift] = 0
        self.shift -= 1
        self.shift = max(self.shift, self.min_x)
        self.update_plots()

    def reverse_signal(self, signal):
        # y(x) = y(-x) but only for the domain (pad with 0s or truncate if necessary)
        left = abs(self.min_x)
        right = abs(self.max_x)
    
        if left == right:
            return signal[::-1]
    
        elif left < right:
            return np.array([0] * (right - left) + signal[:-(right - left)].tolist())[::-1]

        else:
            return np.array(signal[(left-right):].tolist() + [0] * (left - right))[::-1]
    
    def shift_signal(self, signal, shift):
        if shift == 0:
            return signal
        return np.array([0]*(shift) + signal[:-shift].tolist())


    def init_plots(self):
        # Clear all axes
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()

        # Plot PDF of X
        self.ax1.stem(self.domain, self.signal1, 'b')
        self.ax1.set_title('PDF of X')
        self.ax1.grid(True)
        self.ax1.set_xlim(self.min_x, self.max_x)
        self.ax1.set_xticks(range(self.min_x, self.max_x + 1))  # Add integer ticks

        # Plot PDF of Y
        self.ax2.stem(self.domain, self.signal2, 'g')
        self.ax2.set_title('PDF of Y')
        self.ax2.grid(True)
        self.ax2.set_xlim(self.min_x, self.max_x)
        self.ax2.set_xticks(range(self.min_x, self.max_x + 1))  # Add integer ticks

    def update_plots(self):
        # Clear all axes
        self.ax3.clear()
        self.ax4.clear()

        # Plot PDF of Y (reversed and shifted by shift)
        reversed_signal2 = self.reverse_signal(self.signal2)
        shifted_signal2 = self.shift_signal(reversed_signal2, self.shift)

        self.ax3.stem(self.domain, shifted_signal2, 'r')
        self.ax3.set_title(f'PDF of Y (reversed and shifted by {self.shift})')
        self.ax3.grid(True)
        self.ax3.set_xlim(self.min_x, self.max_x)
        self.ax3.set_xticks(range(self.min_x, self.max_x + 1))  # Add integer ticks

        # Calculate and plot signal 3
        third_signal = sum(self.signal1 * shifted_signal2)
        self.third_plot_dict[self.shift] = third_signal

        # Plot all historical third signals
        self.ax4.stem(self.domain, list(self.third_plot_dict.values()), 'g')
        self.ax4.set_title('PDF of Z (convolution of X and Y)')
        self.ax4.grid(True)
        self.ax4.set_xlim(self.min_x, self.max_x)
        self.ax4.set_xticks(range(self.min_x, self.max_x + 1))  # Add integer ticks

        # Update explanation text
        self.update_explanation()

        # Update canvas
        self.fig.tight_layout()
        self.canvas.draw()

    def update_explanation(self):
        self.explanation_text.delete('1.0', tk.END)
        explanation = f"""Explanation:

In this example, random variables represent the probabilities for dice rolls (X and Y). And we want to find the probability of the sum of the two dices.

To do this, we need to use the convolution of the two pdfs. To find the convolution step by step for every shift values we multiply the first and third plot and sum the products (or area under the curve for continuous functions).

To find the probability of the sum of the two dices are z={self.shift}, we need to consider every possible value of the first dice and the second dice.
For example, if we want to find the probability of the sum of the two dices are 3, we need to consider the following:
P(z=3) = P(x=1) * P(y=2) + P(x=2) * P(y=1)

And convolution is actually doing this for every possible shift values (z values).

Controls:
- 'a': Decrease shift
- 'd': Increase shift
- 'r': Reset shift
- 'q': Quit application
"""
        self.explanation_text.insert('1.0', explanation)

    def quit_application(self, event=None):
        self.quit()

if __name__ == "__main__":
    root = ConvolutionPlotter()
    root.mainloop()
