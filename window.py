import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button

class Window:
    def __init__(self):
        mpl.rcParams['toolbar'] = 'None'
        self.window, self.cartesian_plane = plt.subplots()
        self.window.canvas.manager.set_window_title('Perceptron')
        self.window.set_size_inches(8, 7, forward=True)
        plt.subplots_adjust(bottom=0.150, top=0.850)
        self.cartesian_plane.set_xlim(-1.0,1.0)
        self.cartesian_plane.set_ylim(-1.0,1.0)

        self.textbox_learning_rate = TextBox(plt.axes([0.200, 0.9, 0.100, 0.03]), "Learning Rate:")
        self.textbox_epochs = TextBox(plt.axes([0.440, 0.9, 0.100, 0.03]), "Epochs:")
        weights_button = Button(plt.axes([0.025, 0.05, 0.125, 0.03]), "Initialize Weights")
        fit_button = Button(plt.axes([0.160, 0.05, 0.1, 0.03]), "Fit")
        evaluate_button = Button(plt.axes([0.270, 0.05, 0.1, 0.03]), "Evaluate")
        clean_button = Button(plt.axes([0.380, 0.05, 0.1, 0.03]), "Clean")
        plt.show()

if __name__ == '__main__':
    Window()