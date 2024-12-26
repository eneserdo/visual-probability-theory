import tkinter as tk
from tkinter import ttk
from typing import Dict, Type, Callable

# Import visualization classes
from src import ConditionalPlotter, ProbabilityPlotApp, CLTPlotter, ConvolutionPlotter, LLNPlotter

class VisualizationLauncher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Probability Visualizations")
        self.root.geometry("400x500")

        self.root.bind('<q>', self.quit_application)
        
        # Dictionary mapping visualization names to their launch functions
        self.visualizations: Dict[str, Callable] = {
            # "Marginalization": ProbabilityPlotApp, # FIXME ProbabilityPlotApp is not working
            "Conditional Distributions": ConditionalPlotter,
            "Central Limit Theorem": CLTPlotter,
            "Convolution of two discrete pdf": ConvolutionPlotter, # ok
            "Law of Large Numbers": LLNPlotter
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


    def run(self):
        self.root.mainloop()

def main():
    app = VisualizationLauncher()
    app.run()

if __name__ == "__main__":
    main() 