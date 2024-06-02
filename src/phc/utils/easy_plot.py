import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

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
        #     plt.plot(coordinates['x'], coordinates['y'], marker='o', label=label)
        # plt.xlabel('X-axis')
        # plt.ylabel('Y-axis')
        # plt.title('Easy Plot')
        # plt.legend()
        # plt.show()