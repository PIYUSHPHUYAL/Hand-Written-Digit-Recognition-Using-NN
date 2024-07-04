import tkinter as tk
from tkinter import ttk, font
import numpy as np
from Prediction import predict
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageDraw

class DigitRecognitionApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Handwritten Digit Recognition")
        self.geometry("1000x600")
        self.configure(bg="#2C3E50")

        self.custom_font = font.Font(family="Roboto", size=12)
        self.title_font = font.Font(family="Roboto", size=28, weight="bold")

        self.drawing = np.zeros((28, 28), dtype=np.float32)
        self.setup_ui()

    def setup_ui(self):
        # Title
        tk.Label(self, text="Handwritten Digit Recognition", font=self.title_font, bg="#2C3E50", fg="#ECF0F1").pack(pady=20)

        # Main frame
        main_frame = ttk.Frame(self, style="Main.TFrame")
        main_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

        # Canvas frame
        canvas_frame = ttk.Frame(main_frame, style="Canvas.TFrame")
        canvas_frame.pack(side=tk.LEFT, padx=(0, 20))

        self.canvas = tk.Canvas(canvas_frame, width=280, height=280, bg="#34495E", cursor="cross", highlightthickness=0)
        self.canvas.pack(padx=10, pady=10)

        self.canvas.bind("<Button-1>", self.activate_event)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

        # Button frame
        button_frame = ttk.Frame(main_frame, style="Button.TFrame")
        button_frame.pack(side=tk.LEFT, fill=tk.Y)

        clear_btn = ttk.Button(button_frame, text="Clear Canvas", command=self.clear_canvas, style="Custom.TButton")
        clear_btn.pack(fill=tk.X, pady=(0, 10))

        predict_btn = ttk.Button(button_frame, text="Predict", command=self.predict, style="Custom.TButton")
        predict_btn.pack(fill=tk.X)

        # Result frame
        result_frame = ttk.Frame(main_frame, style="Result.TFrame")
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.result_label = ttk.Label(result_frame, text="Digit: ", font=self.custom_font, style="Result.TLabel")
        self.result_label.pack(pady=10)

        # Matplotlib figure for visualization
        self.fig, self.ax = plt.subplots(figsize=(4, 4))
        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=result_frame)
        self.canvas_widget.get_tk_widget().pack()

        self.style_config()

    def style_config(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("Main.TFrame", background="#2C3E50")
        style.configure("Canvas.TFrame", background="#2C3E50")
        style.configure("Button.TFrame", background="#2C3E50")
        style.configure("Result.TFrame", background="#2C3E50")

        style.configure("Custom.TButton", 
                        background="#3498DB", 
                        foreground="#ECF0F1", 
                        font=("Roboto", 12),
                        padding=10)
        style.map("Custom.TButton", 
                  background=[('active', '#2980B9')])

        style.configure("Result.TLabel", 
                        background="#2C3E50", 
                        foreground="#ECF0F1", 
                        font=("Roboto", 16, "bold"))

    def activate_event(self, event):
        self.lastx, self.lasty = event.x, event.y

    def draw_lines(self, event):
        x, y = event.x, event.y
        self.canvas.create_line((self.lastx, self.lasty, x, y), width=20, fill='#ECF0F1', capstyle=tk.ROUND, smooth=tk.TRUE, splinesteps=12)
        
        # Update drawing array
        x1, y1 = int(self.lastx * 28 / 280), int(self.lasty * 28 / 280)
        x2, y2 = int(x * 28 / 280), int(y * 28 / 280)
        
        t = np.linspace(0, 1, 100)
        x_line = np.round(x1 * (1-t) + x2 * t).astype(int)
        y_line = np.round(y1 * (1-t) + y2 * t).astype(int)
        
        x_line = np.clip(x_line, 0, 27)
        y_line = np.clip(y_line, 0, 27)
        
        self.drawing[y_line, x_line] = 1.0
        
        self.lastx, self.lasty = x, y

    def clear_canvas(self):
        self.canvas.delete("all")
        self.drawing.fill(0)
        self.result_label.config(text="Digit: ")
        self.ax.clear()
        self.canvas_widget.draw()

    def predict(self):
        smoothed_drawing = gaussian_filter(self.drawing, sigma=0.5)
        vec = smoothed_drawing.reshape(1, 784) / np.max(smoothed_drawing)

        Theta1 = np.loadtxt('Theta1.txt')
        Theta2 = np.loadtxt('Theta2.txt')

        pred = predict(Theta1, Theta2, vec)

        self.result_label.config(text=f"Predicted Digit: {pred[0]}")

        self.ax.clear()
        self.ax.imshow(smoothed_drawing, cmap='gray')
        self.ax.axis('off')
        self.fig.patch.set_facecolor('#2C3E50')
        self.canvas_widget.draw()

if __name__ == "__main__":
    app = DigitRecognitionApp()
    app.mainloop()