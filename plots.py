import numpy as np
import matplotlib.pyplot as plt

# from stack exchange
class scatter():
    def __init__(self,x,y,ax,size=1,**kwargs):
        self.n = len(x)
        self.ax = ax
        self.ax.figure.canvas.draw()
        self.size_data=size
        self.size = size
        self.sc = ax.scatter(x,y,s=self.size,**kwargs)
        self._resize()
        self.cid = ax.figure.canvas.mpl_connect('draw_event', self._resize)

    def _resize(self,event=None):
        ppd=72./self.ax.figure.dpi
        trans = self.ax.transData.transform
        s =  ((trans((1,self.size_data))-trans((0,0)))*ppd)[1]
        if s != self.size:
            self.sc.set_sizes(s**2*np.ones(self.n))
            self.size = s
            self._redraw_later()
    
    def _redraw_later(self):
        self.timer = self.ax.figure.canvas.new_timer(interval=10)
        self.timer.single_shot = True
        self.timer.add_callback(lambda : self.ax.figure.canvas.draw_idle())
        self.timer.start()


# custom triangular lattice plotter
def show_wavefunction(ax, coordinates, values, vmin=None, vmax=None, colorbar=True, cmap=mpl.colormaps['viridis']):
    if vmin is None: vmin = np.min(values)
    if vmax is None: vmax = np.max(values)
    colors = mpl.colormaps['viridis']((values - vmin) / (vmax - vmin))
    cNorm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    scatter(coordinates[0].reshape(-1), coordinates[1].reshape(-1), ax, size=1.1, c=colors.reshape(-1, 4), marker="h")
    if colorbar:
        plt.colorbar(cm.ScalarMappable(norm=cNorm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
    ax.tick_params(axis='both', left=False, labelleft=False, which='both', bottom=False, top=False, labelbottom=False) 
    
    return None


