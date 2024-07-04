import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import statistics


class EasyPlot:
    def __init__(self):
        self.points = {}
        
    def add_point(self, x, y, label):
        if label not in self.points:
            self.points[label] = {'x': [], 'y': []}
        self.points[label]['x'].append(x)
        self.points[label]['y'].append(y)
        
    def plot(self, writer, epoch_count):
        for label, coordinates in self.points.items():
            new_fig = plt.figure(figsize=(7, 3))
            plt.plot(coordinates['x'], coordinates['y'], label=label)
            if('reward' in label):
                writer.add_figure(f"matplot_reward/{label}", new_fig, epoch_count)
            else:
                writer.add_figure(f"matplot/{label}", new_fig, epoch_count)
            plt.close(new_fig)
    
    def plot_mean(self, writer, epoch_count):
        for label, coordinates in self.points.items():
            if('reward' in label):
                writer.add_scalar(f"scaler_reward/{label}", statistics.mean(coordinates['y']), epoch_count)
            else:
                writer.add_scalar(f"scaler/{label}", statistics.mean(coordinates['y']), epoch_count)