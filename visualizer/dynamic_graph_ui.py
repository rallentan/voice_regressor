import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class DynamicGraphUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Dynamic 2D Graph")

        # Create a figure and an axes for the plot
        self.figure, self.ax = plt.subplots()
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)

        # Create a canvas to embed the figure in the Tkinter window
        self.canvas = FigureCanvasTkAgg(self.figure, master)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Plot an initial dot
        self.dot, = self.ax.plot(0, 0, 'bo')

        # Update function to change the position of the dot
        # self.update_dot_position(0, 0)

    def update_dot_position(self, x, y):
        # Update the data for the dot
        self.dot.set_data(x, y)

        # Redraw the canvas with the new dot position
        self.canvas.draw_idle()
