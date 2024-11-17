import tkinter as tk
from tkinter import ttk
import importlib
import sys
from typing import Dict, Type, Callable

# Import visualization classes
from src import ConditionalPlotApp, ProbabilityPlotApp, CLTVisualizer

class VisualizationLauncher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Probability Visualizations")
        self.root.geometry("400x500")
        
        # Dictionary mapping visualization names to their launch functions
        self.visualizations: Dict[str, Callable] = {
            "Marginalization": self.launch_marginalization,
            "Conditional Distributions": self.launch_conditional,
            "Central Limit Theorem": self.launch_clt
        }
        
        self.create_widgets()

    def create_widgets(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="Probability Visualization Tools",
            font=('Helvetica', 16, 'bold')
        )
        title_label.grid(row=0, column=0, pady=(0, 20))
        
        # Description
        desc_label = ttk.Label(
            main_frame,
            text="Select a visualization to launch:",
            font=('Helvetica', 10)
        )
        desc_label.grid(row=1, column=0, pady=(0, 20))
        
        # Create buttons for each visualization
        for idx, (name, func) in enumerate(self.visualizations.items()):
            btn = ttk.Button(
                main_frame,
                text=name,
                command=func,
                width=30
            )
            btn.grid(row=idx+2, column=0, pady=5)
        
        # Quit button
        quit_btn = ttk.Button(
            main_frame,
            text="Quit",
            command=self.root.quit,
            width=30
        )
        quit_btn.grid(row=len(self.visualizations)+2, column=0, pady=(20, 0))

    def launch_visualization(self, VisualizerClass: Type, *args, **kwargs):
        """Generic method to launch a visualization window"""
        new_window = tk.Toplevel(self.root)
        app = VisualizerClass(new_window, *args, **kwargs)
        return app

    def launch_marginalization(self):
        """Launch marginalization visualization"""
        self.launch_visualization(ProbabilityPlotApp)

    def launch_conditional(self):
        """Launch conditional distribution visualization"""
        self.launch_visualization(ConditionalPlotApp)

    def launch_clt(self):
        """Launch CLT visualization"""
        # CLT visualizer doesn't need a master window
        CLTVisualizer()

    def run(self):
        self.root.mainloop()

def main():
    app = VisualizationLauncher()
    app.run()

if __name__ == "__main__":
    main() 