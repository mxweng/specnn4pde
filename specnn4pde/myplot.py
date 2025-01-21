__all__ = ['show_colors', 'cmaps', 'colors',
           'ax_config', 'ax3d_config', 'zaxis_sci_formatter', 
           'latex_render','colorbar_config', ]

import pkg_resources
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable


class color:
    def __init__(self):
        pass

# Load the colormaps from the matlab file
mat_file_path = pkg_resources.resource_filename(__name__, 'colormaps.mat')
mat_data = scipy.io.loadmat(mat_file_path)

# Define the colormaps from matlab
cmaps = color()
cmaps.parula = ListedColormap(mat_data['parula'])
cmaps.turbo = ListedColormap(mat_data['turbo'])
cmaps.hsv = ListedColormap(mat_data['hsv'])
cmaps.hot = ListedColormap(mat_data['hot'])
cmaps.cool = ListedColormap(mat_data['cool'])
cmaps.spring = ListedColormap(mat_data['spring'])
cmaps.summer = ListedColormap(mat_data['summer'])
cmaps.autumn = ListedColormap(mat_data['autumn'])
cmaps.winter = ListedColormap(mat_data['winter'])
cmaps.gray = ListedColormap(mat_data['gray'])
cmaps.bone = ListedColormap(mat_data['bone'])
cmaps.copper = ListedColormap(mat_data['copper'])
cmaps.pink = ListedColormap(mat_data['pink'])
cmaps.sky = ListedColormap(mat_data['sky'])
cmaps.abyss = ListedColormap(mat_data['abyss'])
cmaps.jet = ListedColormap(mat_data['jet'])
cmaps.lines = ListedColormap(mat_data['lines'])
cmaps.colorcube = ListedColormap(mat_data['colorcube'])
cmaps.prism = ListedColormap(mat_data['prism'])
cmaps.flag = ListedColormap(mat_data['flag'])
cmaps.white = ListedColormap(mat_data['white'])

colors = color()
# Custom color schemes
# from tikz datavisualization
colors.tikz = [(r/255, g/255, b/255) for r, g, b in [(0,0,204),(0,153,0),(204,102,0),(204,0,0)]]
# from https://zhuanlan.zhihu.com/p/691027218
colors.cartoon1 = [(r/255, g/255, b/255) for r, g, b in [(14,62,135),(52,108,172),(222,234,234),(247,228,116),(216,178,58)]]
colors.cartoon2 = [(r/255, g/255, b/255) for r, g, b in [(70,120,142),(120,183,201),(246,224,147),(151,179,25),(229,139,123)]]
colors.cartoon3 = [(r/255, g/255, b/255) for r, g, b in [(5, 89,18),(20,130,107),(163,209,205),(250,222,188),(217,149,74),(115,57,23)]]


def show_colors(colors):
    """
    Display the colors as color blocks.

    Parameters
    ----------
    colors: list
        List of colors to display.
    """
    fig, ax = plt.subplots(1, 1, figsize=(len(colors), 0.8))

    for i, color in enumerate(colors):
        rect = patches.FancyBboxPatch((i, 0), 0.4, 0.4, boxstyle="round,pad=0.2", facecolor=color, edgecolor='w')
        ax.add_patch(rect)

    ax.axis('off')  # Hide axes
    ax.axis('equal')  # Equal aspect ratio
    plt.show()



def ax_config(ax,  
            title=None, title_fontsize=12., title_y=1., title_pad=6.,
            xlabel=None, xlabel_fontsize=12.,
            ylabel=None, ylabel_fontsize=12.,
            xlims=None, ylims=None,
            spine_width=0.8, spine_color='gray',
            tick_major=True, 
            xtick_major=None, ytick_major=None,
            tick_major_direction='in', tick_major_color='gray',
            tick_major_labelsize=10., tick_major_labelcolor='black',
            tick_major_length=3., tick_major_width=0.8, tick_major_pad=4.,
            tick_minor=True, 
            xtick_minor=None, ytick_minor=None, 
            tick_minor_direction='in', tick_minor_color='gray',
            tick_minor_labelsize=10., tick_minor_labelcolor='black',
            tick_minor_length=2., tick_minor_width=0.5, tick_minor_pad=2.,
            grid_major=True, 
            grid_major_linewidth=0.5, grid_major_linestyle='--', 
            grid_major_color='lightgray', grid_major_alpha=1.,
            grid_minor=True, 
            grid_minor_linewidth=0.5, grid_minor_linestyle='--',
            grid_minor_color='lightgray', grid_minor_alpha=1.,
            legend=True, 
            legend_loc='best', legend_bbox_to_anchor=None, 
            legend_edgecolor='C0', legend_facecolor='1', legend_framealpha=0.3, 
            legend_fontsize=10., legend_ncol=1, legend_handlelength=2.,
            sci_fmt=False, scilimits=(0, 0)):
    """
    Configure the plot with title, labels, spine, tick, grid, and legend parameters for a given Axes object.

    Parameters
    ----------
    ax (matplotlib.axes.Axes): The Axes object to configure.
    title (str): Title of the plot.
    title_fontsize (float): Font size of the title.
    title_y (float): Position of the title along the y-axis.
    title_pad (float): Padding of the title from the plot.
    xlabel (str): Label of the x-axis.
    xlabel_fontsize (float): Font size of the x-axis label.
    ylabel (str): Label of the y-axis.
    ylabel_fontsize (float): Font size of the y-axis label.
    xlims (2-tuple): Limits of the x-axis.
    ylims (2-tuple): Limits of the y-axis.
    spine_width (float): Width of the spines.
    spine_color (str): Color of the spines.
    tick_major (bool): Whether to configure major ticks.
    xtick_major (list): List of major ticks for the x-axis.
    ytick_major (list): List of major ticks for the y-axis.
    tick_major_direction (str): Direction of major ticks.
    tick_major_color (str): Color of major ticks.
    tick_major_labelsize (float): Label size of major ticks.
    tick_major_labelcolor (str): Label color of major ticks.
    tick_major_length (float): Length of major ticks.
    tick_major_width (float): Width of major ticks.
    tick_major_pad (float): Padding of major ticks.
    tick_minor (bool): Whether to configure minor ticks.
    xtick_minor (list): List of minor ticks for the x-axis.
    ytick_minor (list): List of minor ticks for the y-axis.
    tick_minor_direction (str): Direction of minor ticks.
    tick_minor_color (str): Color of minor ticks.
    tick_minor_labelsize (float): Label size of minor ticks.
    tick_minor_labelcolor (str): Label color of minor ticks.
    tick_minor_length (float): Length of minor ticks.
    tick_minor_width (float): Width of minor ticks.
    tick_minor_pad (float): Padding of minor ticks.
    grid_major (bool): Whether to configure major grid.
    grid_major_linewidth (float): Line width of major grid.
    grid_major_linestyle (str): Line style of major grid.
    grid_major_color (str): Color of major grid.
    grid_major_alpha (float): Alpha transparency of major grid.
    grid_minor (bool): Whether to configure minor grid.
    grid_minor_linewidth (float): Line width of minor grid.
    grid_minor_linestyle (str): Line style of minor grid.
    grid_minor_color (str): Color of minor grid.
    grid_minor_alpha (float): Alpha transparency of minor grid.
    legend (bool): Whether to configure legend.
    legend_loc (str): Location of the legend.
    legend_bbox_to_anchor (tuple or None): Bounding box anchor for the legend.
    legend_edgecolor (str): Edge color of the legend.
    legend_facecolor (str): Face color of the legend.
    legend_framealpha (float): Frame alpha transparency of the legend.
    legend_fontsize (float): Font size of the legend.
    legend_ncol (int): Number of columns in the legend.
    legend_handlelength (float): Length of the legend handles.
    sci_fmt (bool): Whether to use scientific format for the y-axis ticks.
    scilimits (2-tuple): Limits for scientific notation.
    """
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize, y=title_y, pad=title_pad)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    if xlims:
        ax.set_xlim(xlims[0], xlims[1])
    if ylims:
        ax.set_ylim(ylims[0], ylims[1])

    if ax.name == 'polar':
        ax.spines['polar'].set_linewidth(spine_width)
        ax.spines['polar'].set_edgecolor(spine_color)
    else:
        for spine in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine].set_linewidth(spine_width)
            ax.spines[spine].set_color(spine_color)
    
    if tick_major:     
        ax.tick_params(axis='both', which='major', direction=tick_major_direction, pad=tick_major_pad,
                       labelcolor=tick_major_labelcolor, labelsize=tick_major_labelsize, 
                       length=tick_major_length, width=tick_major_width,
                       color=tick_major_color, top=tick_major, right=tick_major)
        
    if xtick_major != None:
        ax.set_xticks(xtick_major)
    if ytick_major != None:
        ax.set_yticks(ytick_major)
    
    if tick_minor:
        ax.tick_params(axis='both', which='minor', direction=tick_minor_direction, pad=tick_minor_pad,
                       labelcolor=tick_minor_labelcolor, labelsize=tick_minor_labelsize,
                       length=tick_minor_length, width=tick_minor_width,
                       color=tick_minor_color, top=tick_minor, right=tick_minor)
    
    if xtick_minor != None:
        ax.set_xticks(xtick_minor, minor=True)
    if ytick_minor != None:
        ax.set_yticks(ytick_minor, minor=True)

    if grid_major:
        ax.grid(grid_major, which='major', color=grid_major_color, linestyle=grid_major_linestyle, 
                linewidth=grid_major_linewidth, alpha=grid_major_alpha)
    
    if grid_minor:
        ax.grid(grid_minor, which='minor', color=grid_minor_color, linestyle=grid_minor_linestyle, 
                linewidth=grid_minor_linewidth, alpha=grid_minor_alpha)
    
    handles, labels = ax.get_legend_handles_labels()
    if legend and labels:
        ax.legend(loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor, ncol=legend_ncol, 
                  edgecolor=legend_edgecolor, facecolor=legend_facecolor, 
                  framealpha=legend_framealpha, fontsize=legend_fontsize,
                  handlelength=legend_handlelength)
        
    if sci_fmt:    
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=scilimits)


def ax3d_config(ax, axis3don=True, view_angle=[5, 45], 
                box_aspect=None, axis_limits=None,
                title=None, title_size=12., title_y=1., title_pad=0.,
                xlabel=None, ylabel=None, zlabel=None,
                labelsize=12, labelpad=[-5,-5,-5], label_rotation=[0,0,-90],
                pane_color=(1,1,1,0), spine_color='grey', spine_width=0.5,
                tick_labelsize=10, tick_pad=[-5,-4,-1.5], tick_color='k',
                tick_inward_length=0, tick_outward_length=0.3, tick_linewidth=0.5, 
                grid_color='lightgray', grid_linewidth=0.5, grid_linestyle=':',
                projscale=[1,1,1,1], movescale=[0,0,0,0]):
    """
    Configure the plot with title, tick, grid parameters for a given 3D Axes object.

    Parameters
    ----------
    ax : 3D axis object
    axis3don (bool): turn on/off 3D axis
    view_angle (list): [elevation, azimuth] in degrees
    box_aspect (list): aspect ratio of the box
    axis_limits (list): limits of the x, y, and z axes, [xmin, xmax, ymin, ymax, zmin, zmax]
    title (str): title of the plot
    title_size (float): font size of the title
    title_y (float): position of the title along the y-axis
    title_pad (int): padding for the title
    xlabel (str): x-axis label
    ylabel (str): y-axis label
    zlabel (str): z-axis label
    labelsize (int/list): label font size, if int, apply to all labels, if list, apply to each label
    labelpad (list): padding for each axis label
    label_rotation (list): label rotation in degrees
    pane_color (str): color of the pane
    spine_color (str): color of the axis lines
    spine_width (float): width of the axis lines
    tick_pad (list): padding for each tick label
    tick_labelsize (int): font size of the tick labels
    tick_color (str): color of the ticks and tick labels
    tick_inward_length (float): inward length for the ticks
    tick_outward_length (float): outward length for the ticks
    tick_linewidth (float): linewidth of the ticks
    grid_color (str): color of the grid lines
    grid_linewidth (float): linewidth of the grid lines
    grid_linestyle (str): linestyle of the grid lines
    projscale (list): scale of the projection in direction [x, y, z, w]
                        'w' is Homogeneous Coordinates
    movescale (list): move the position of the plot in [x, y, width, height]                   
    """
    
    ax._axis3don = axis3don
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    ax.tick_params(labelsize=tick_labelsize, colors=tick_color)
    ax.xaxis.set_tick_params(pad=tick_pad[0])
    ax.yaxis.set_tick_params(pad=tick_pad[1])
    ax.zaxis.set_tick_params(pad=tick_pad[2])

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_pane_color(pane_color)
        axis.line.set_linewidth(spine_width)
        axis.line.set_color(spine_color)

        axis._axinfo["tick"]['inward_factor'] = tick_inward_length
        axis._axinfo["tick"]['outward_factor'] = tick_outward_length
        axis._axinfo["tick"]['linewidth'][True] = tick_linewidth

        axis._axinfo["grid"]['color'] = grid_color
        axis._axinfo["grid"]['linewidth'] = grid_linewidth
        axis._axinfo["grid"]['linestyle'] = grid_linestyle
    
    if title is not None:
        ax.set_title(title, fontsize=title_size, y=title_y, pad=title_pad)

    if isinstance(labelsize, int) or isinstance(labelsize, float):
        labelsize = [labelsize]*3 
    if xlabel is not None:
        ax.set_xlabel(xlabel, labelpad=labelpad[0], fontsize=labelsize[0], rotation=label_rotation[0])
    if ylabel is not None:
        ax.set_ylabel(ylabel, labelpad=labelpad[1], fontsize=labelsize[1], rotation=label_rotation[1])
    if zlabel is not None:
        ax.set_zlabel(zlabel, labelpad=labelpad[2], fontsize=labelsize[2], rotation=label_rotation[2])

    if axis_limits:
        ax.set_xlim(*axis_limits[:2])
        ax.set_ylim(*axis_limits[2:4])
        ax.set_zlim(*axis_limits[4:])
    if box_aspect:
        ax.set_box_aspect(box_aspect)

    if projscale != [1,1,1,1]:
        ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag(projscale))
    if movescale != [0,0,0,0]:
        pos = ax.get_position()
        ax.set_position([pos.x0 + movescale[0], pos.y0 + movescale[1], 
                         pos.width + movescale[2], pos.height + movescale[3]])


def latex_render(flag=True, packages=[]):
    """
    Enable or disable LaTeX rendering in matplotlib plots.

    Parameters
    ----------
    flag (bool): Whether to enable or disable LaTeX rendering.
                If True, enable LaTeX rendering and set the font to 'Computer Modern Roman',
                        which is the default font used in LaTeX.
                If False, disable LaTeX rendering and reset the font to default settings of matplotlib. 
    packages (list): List of LaTeX packages to include in the preamble.
                    Default packages include 'amssymb', 'amsmath', 'amsthm', 'bm', 'bbm'.

    Example
    ----------
    >>> plot_latex_render(True)
    """
    if flag:
        plt.rcParams['text.usetex'] = True
        packages = list(set(packages+['amssymb', 'amsmath', 'amsthm', 'bm', 'bbm']))
        plt.rcParams['text.latex.preamble'] = r'\usepackage{' + ','.join(packages) + '}'
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Computer Modern Roman', 'Times New Roman'] + plt.rcParams['font.serif']
    else:
        plt.rc('text', usetex=False)
        plt.rcParams['font.family'] = 'sans-serif'


def colorbar_config(img, cax=None, ax=None, label=None, labelsize=10, 
                    shrink=0.7, aspect=20, pad=0.1, orientation='vertical', 
                    fraction=0.1, size='5%', loc='right',
                    position=None,
                    img_clim=None,
                    ticks=None, tick_labels=None, tick_length=2, tick_width=0.5, 
                    outline_visible=True, 
                    outline_linewidth=0.5, outline_edgecolor='black', outline_linestyle='-',
                    sci_fmt=True, powerlimits=(-1, 1),
                    scalar_pos=(1, 1.05)):
    """
    Add colorbar to the figure and configure the appearance and label of the colorbar.

    Parameters
    ----------
    img : ScalarMappable object
        The image object created by ax.imshow() or similar functions.
    cax : `~matplotlib.axes.Axes`, optional
        Axes into which the colorbar will be drawn.  If `None`, then a new
        Axes is created and the space for it will be stolen from the Axes(s)
        specified in *ax*.
    ax : `~matplotlib.axes.Axes` or iterable or `numpy.ndarray` of Axes, optional
        The one or more parent Axes from which space for a new colorbar Axes
        will be stolen. This parameter is only used if *cax* is not set.
        Defaults to the Axes that contains the mappable used to create the
        colorbar.
    label : str
        The label for the colorbar.
    labelsize : int, optional
        The font size of the ticks label, default is 10.
    shrink : float, optional
        The shrink factor of the colorbar, default is 0.7.
    aspect : int, optional
        The aspect ratio of the colorbar, default is 20.
    pad : float, optional
        The padding between the colorbar and the axes, default is 0.1.
    orientation : str, optional
        The orientation of the colorbar, default is 'vertical'.
    loc : str, optional
        The location of the colorbar, default is 'right'.
    size: str, optional
        The size of the colorbar, default is '5%'.
    fraction : float, optional
        The fraction of the axes that the colorbar occupies, default is 0.1.
    position : list or tuple, optional
        The position of the colorbar [x0, y0, width, height], default is None.
        If ax is not None, the position is relative to the position of ax, i.e.,
                ax.get_position() with elementwise addition of position.
        If ax is None, the position is relative to the figure.
    img_clim : list or tuple, optional
        The limits of color rendering and the colorbar, values outside the limits will be set to the limits.
    ticks : list, optional
        The ticks to be shown on the colorbar, default is None.
    tick_labels : list, optional
        The labels for the ticks, only used when ticks is not None, should be the same length as ticks.
    tick_length : float, optional
        The length of the ticks, default is 2.
    tick_width : float, optional
        The width of the ticks, default is 0.5.
    outline_visible : bool, optional
        Whether to show the colorbar outline, default is True.
    outline_linewidth : float, optional
        The linewidth of the colorbar outline, default is 0.5.
    outline_edgecolor : str, optional
        The edge color of the colorbar outline, default is 'black'.
    outline_linestyle : str, optional
        The linestyle of the colorbar outline, default is '-'.
    sci_fmt : bool, optional
        Whether to use scientific format for the colorbar ticks, default is True.
    powerlimits : tuple, optional
        The power limits for scientific notation, default is (-1, 1), 
        i.e. use scientific notation for numbers outside range 1e-1 to 1e1.
    scalar_pos : tuple, optional
        The position of the scalar text, default is (1,1.05).
        The first element represents the x position of the right side of the scientific notation text, where 0 is the left edge and 1 is the right edge.
        The second element represents the y position of the bottom side of the scientific notation text, where 0 is the bottom edge and 1 is the top edge.
    """
    
    # Add colorbar
    if ax.name == 'rectilinear':
        if cax is None and ax is not None:        
            divider = make_axes_locatable(ax)
            # add a new axes for the colorbar, the same size as the main axes
            cax = divider.append_axes(loc, size=size, pad=pad)
        cbar = plt.colorbar(img, cax=cax, ax=ax, shrink=shrink, aspect=aspect, pad=pad, orientation=orientation, fraction=fraction)
    else:
        # for 'polar' and '3d' axes
        cbar = plt.colorbar(img, cax=cax, ax=ax, shrink=shrink, aspect=aspect, pad=pad, orientation=orientation, fraction=fraction)

    if position is not None:
        if ax is not None:
            pos = ax.get_position()
            cbar.ax.set_position([pos.x0 + pos.width + position[0], pos.y0+position[1], 
                                    position[2], pos.height+position[3]])
        else:
            cbar.ax.set_position(position)

    cbar.ax.tick_params(labelsize=labelsize, length=tick_length, width=tick_width)

    if img_clim:
        img.set_clim(*img_clim)

    if ticks is not None:
        cbar.set_ticks(ticks)
        if tick_labels:
            if len(tick_labels) != len(ticks):
                raise ValueError("The length of tick_labels and ticks should be the same.")
            cbar.set_ticklabels(tick_labels)
    if label:
        cbar.set_label(label, fontsize=labelsize)

    if outline_visible:
        cbar.outline.set_visible(True)
        cbar.outline.set_linewidth(outline_linewidth)
        cbar.outline.set_edgecolor(outline_edgecolor)
        cbar.outline.set_linestyle(outline_linestyle)
    else:
        cbar.outline.set_visible(False)

    if sci_fmt:
        # set the colorbar ticks to scientific format
        cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        cbar.ax.yaxis.get_major_formatter().set_powerlimits(powerlimits)

    cbar.ax.yaxis.get_offset_text().set_x(scalar_pos[0])
    cbar.ax.yaxis.get_offset_text().set_y(scalar_pos[1])
    
    return cbar


def zaxis_sci_formatter(fig, ax, scalar_pos=None):
    """
    Format the z-axis of a 3D plot to use scientific notation.
    
    Attention
    ---------
    This function should be called after the plot is created, i.e., after ax.plot_surface(), ax.scatter(), etc.

    Parameters
    ----------
    fig : Figure object
        The figure object associated with the 3D plot.
    ax : Axes3D object
        The 3D axes object associated with the plot.
    scalar_pos : tuple, optional
        The position of the scalar text, default is None.
    """
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax.zaxis.set_major_formatter(formatter)
    # redraws the figure to update the z-axis
    fig.canvas.draw()

    ax.get_zaxis().offsetText.set_visible(False)
    if scalar_pos is None:
        scalar_pos = ax.get_zaxis().offsetText.get_position()
    ax.text2D(*scalar_pos, ax.get_zaxis().offsetText.get_text(), transform=ax.transAxes)