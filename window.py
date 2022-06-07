import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button

from perceptron import Perceptron
from constants import *

class Window:
    def __init__(self):
        self.learning_rate = LEARNING_RATE
        self.epochs = EPOCHS
        self.X = np.array([[],[]])
        self.Y = np.array([])
        self.neuron = None

        mpl.rcParams['toolbar'] = 'None'
        self.window, self.cartesian_plane = plt.subplots()
        self.window.canvas.manager.set_window_title('Perceptron')
        self.window.set_size_inches(8, 7, forward=True)
        plt.subplots_adjust(bottom=0.150, top=0.9)
        self.cartesian_plane.set_xlim(XL,XU)
        self.cartesian_plane.set_ylim(XL,XU)

        self.textbox_has_converged = TextBox(plt.axes([0.250, 0.93, 0.500, 0.03]), '')
        self.textbox_learning_rate = TextBox(plt.axes([0.200, 0.05, 0.100, 0.03]), 'Learning Rate')
        self.textbox_epochs = TextBox(plt.axes([0.375, 0.05, 0.100, 0.03]), 'Epochs:')
        fit_button = Button(plt.axes([0.550, 0.05, 0.1, 0.03]), 'Fit')
        degraded_button = Button(plt.axes([0.650, 0.05, 0.1, 0.03]), 'Degraded')
        clean_button = Button(plt.axes([0.800, 0.05, 0.1, 0.03]), 'Clean')

        self.textbox_learning_rate.on_submit(self.validateEpochs)
        self.textbox_epochs.on_submit(self.validateLearningRate)
        fit_button.on_clicked(self.fit)
        degraded_button.on_clicked(self.degraded)
        clean_button.on_clicked(self.clean)
        self.window.canvas.mpl_connect('button_press_event', self.__onclick)
        plt.show()

    def validateEpochs(self, value):
        try:
            x = int(value)
            self.epochs = x
        except ValueError:
            self.epochs = EPOCHS

    def validateLearningRate(self, value):
        try:
            x = float(value)
            if(x > 0 and x < 1):
                self.learning_rate = x
            else:
                self.learning_rate = LEARNING_RATE  
        except ValueError:
            self.learning_rate = LEARNING_RATE

    def fit(self, event):
        neuron = Perceptron(self.X.shape[0], self.learning_rate) # dimension and learning rate
        neuron.fitness(self.X, self.Y, self.epochs)
        if neuron.hasConverged():
            self.textbox_has_converged.set_val('Perceptron has converged in ' + str(neuron.epochsReached()) + ' epochs')
        else:
            self.textbox_has_converged.set_val('Perceptron has not converged')
        self.graphLine(neuron.get_w(), neuron.get_b())
        self.neuron = neuron

    def clean(self, event):
        self.learning_rate = LEARNING_RATE
        self.epochs = EPOCHS
        self.X = np.array([[],[]])
        self.Y = np.array([])
        self.neuron = None
        self.cartesian_plane.clear()
        self.cartesian_plane.set_xlim(XL,XU)
        self.cartesian_plane.set_ylim(XL,XU)
        self.textbox_has_converged.set_val(EMPTY_CHAR)

    def __onclick(self, event):
        if event.inaxes == self.cartesian_plane:
            current_point = np.array([[event.xdata], [event.ydata]])
            self.X = np.append(self.X, current_point, axis = 1)
            is_left_click = event.button == 1            
            self.Y = np.append(self.Y, 0 if is_left_click else 1)
            self.cartesian_plane.plot(event.xdata, event.ydata, 'b.' if is_left_click else 'rx')
        self.window.canvas.draw()
    
    def graphLine(self, w, bias): 
        x1 = (1/w[1]) * (-w[0]*(XU)-bias)
        x2 = (1/w[1]) * (-w[0]*(XL)-bias) 
        self.cartesian_plane.plot([XL,XU], [x2,x1], 'y-')
        self.window.canvas.draw()
        
    def degraded(self, event):
        x = XL
        y = XL
        while(x <= XU):
            y = XL
            while y <= XU:
                point = np.dot(self.neuron.get_w(), np.array([x,y])) + self.neuron.get_b() 
                if self.neuron.pw(point):
                    self.cartesian_plane.plot(x,y, color=(0, 0.6, 0, 0.1), marker='o')
                else:
                    self.cartesian_plane.plot(x,y, color=(1, 0.6, 0, 0.1), marker='o')
                y += STEP
            x += STEP
        self.window.canvas.draw()

if __name__ == '__main__':
    Window()