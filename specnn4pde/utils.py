__all__ = ['pkg_system_info', 'svg2pdf', 'func_timer', 'timer',
           'ax_config', 'ax3d_config', 'plot_latex_render', 'colorbar_config',
           ]

import os
import platform
import psutil
import pandas as pd
from datetime import datetime
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import GPUtil
import importlib
import subprocess

import time
from functools import wraps

def pkg_system_info(packages, show_pkg=True, show_system=True, show_gpu=True):
    """
    This function takes a list of package names as input, imports each package dynamically, 
    and displays the version information of each package and the system information.

    Parameters
    ----------
    packages : list of str
        A list of package names to import and get version information.
    show_pkg : bool
        Whether to show package version information. Default is True.
    show_system : bool
        Whether to show system information. Default is True.
    show_gpu : bool
        Whether to show GPU information. Default is True.

    Returns
    ----------
    None

    Example
    ----------
    >>> pkg_system_info(['numpy', 'pandas', 'scipy', 'qiskit'], show_pkg=True, show_gpu=True, show_system=False)
    """

    def get_cpu_info():
        # Get CPU information on Linux
        cpu_info = subprocess.check_output("lscpu", shell=True).decode()
        architecture = subprocess.check_output("uname -m", shell=True).decode().strip()
        lines = cpu_info.split('\n')
        info_dict = {}
        for line in lines:
            if "Vendor ID:" in line:
                info_dict['Vendor ID'] = line.split(':')[1].strip()
            if "CPU family:" in line:
                info_dict['CPU family'] = line.split(':')[1].strip()
            if "Model:" in line:
                info_dict['Model'] = line.split(':')[1].strip()
            if "Stepping:" in line:
                info_dict['Stepping'] = line.split(':')[1].strip()
        return architecture, info_dict


    if show_pkg:
        # Get packages version information
        pkg_versions = []
        for pkg_name in packages:
            try:
                pkg = importlib.import_module(pkg_name)
                version = pkg.__version__
            except AttributeError:
                version = "Version not available"
            pkg_versions.append((pkg.__name__, version))
        
        pkg_versions_df = pd.DataFrame(pkg_versions, columns=['Package', 'Version'])
        display(HTML(pkg_versions_df.to_html(index=False)))

    if show_gpu:
        # Get GPU information
        gpus = GPUtil.getGPUs()
        gpu_info_list = []
        if gpus:
            for gpu in gpus:
                gpu_info = [gpu.name, f"{round(gpu.memoryTotal / 1024, 1)} Gb", 1]
                for existing_gpu_info in gpu_info_list:
                    if existing_gpu_info[0] == gpu_info[0] and existing_gpu_info[1] == gpu_info[1]:
                        existing_gpu_info[2] += 1
                        break
                else:
                    gpu_info_list.append(gpu_info)
        else:
            gpu_info_list = [['No GPU detected', 'N/A', 'N/A']]

        gpu_info_df = pd.DataFrame(gpu_info_list, columns=['GPU Version', 'GPU Memory', 'Count'])
        display(HTML(gpu_info_df.to_html(index=False)))

    if show_system:
        # Get system information
        system_info = {
            'Python version': platform.python_version(),
            'Python compiler': platform.python_compiler(),
            'Python build': platform.python_build(),
            'OS': platform.system(),
            'CPU Version': platform.processor(),
            'CPU Number': psutil.cpu_count(),
            'CPU Memory': f"{round(psutil.virtual_memory().total / (1024.0 **3), 1)} Gb",
            'Time': datetime.now().strftime("%a %b %d %H:%M:%S %Y %Z")
        }

        if system_info['OS'] == 'Linux':
            architecture, cpu_info = get_cpu_info()
            system_info['CPU Version'] = f"{architecture} Family {cpu_info['CPU family']} Model {cpu_info['Model']} Stepping {cpu_info['Stepping']}, {cpu_info['Vendor ID']}"

        system_info_df = pd.DataFrame(list(system_info.items()), columns=['System Information', 'Details'])
        display(HTML(system_info_df.to_html(index=False)))


def svg2pdf(directory, inkscape_path=None):
    """
    Convert all SVG files in a directory to PDF using Inkscape.

    This function is only tested on Windows and it requires 
    Inkscape to be installed on your system. 
    You can download it from https://inkscape.org/release/ and install it.

    Parameters:
    ----------
    directory (str): The directory that contains the SVG files.
    inkscape_path (str, optional): The path to the Inkscape executable. 
        If not provided, the function will use "inkscape" as the default value, 
        assuming that Inkscape is in the system's PATH.

    Example:
    ----------
    >>> convert_svg_to_pdf(r'D:/path/to/your/directory')
    """
    if inkscape_path is None:
        inkscape_path = "inkscape"
    for filename in os.listdir(directory):
        if filename.endswith(".svg"):
            svg_file = os.path.join(directory, filename)
            pdf_file = os.path.join(directory, os.path.splitext(filename)[0] + ".pdf")
            command = f'"{inkscape_path}" "{svg_file}" --export-filename="{pdf_file}"'
            subprocess.run(command, shell=True)


def func_timer(function):
    """
    This is a timer decorator. It calculates the execution time of the function.
    
    Args
    ----------
    function : callable
        The function to be timed.

    Returns
    ----------
    function : callable
        The decorated function which will print its execution time when called.

    Example
    ----------
    >>> @func_timer
    >>> def my_function(n):
    >>>     return sum(range(n))
    >>> my_function(1000000)
    """

    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print ("Running time of %s: %.3e seconds" % (function.__name__, t1-t0))
        return result
    return function_timer


class timer:
    """
    A simple timer class.
    
    Attributes
    ----------
    start_time : float
        The time when the timer was started.
    last_lap_time : float
        The time when the last lap was recorded.

    Methods
    -------
    __init__():
        Initializes the timer.
    __str__():
        Returns a string representation of the timer.
    __repr__():
        Returns a formal string representation of the timer.
    reset():
        Resets the timer.
    update():
        Updates the last lap time without printing anything.
    lap():
        Records a lap time and prints the time difference since the last lap.
    stop():
        Prints the total time.
    """
    
    def __init__(self):
        self.start_time = time.time()
        self.last_lap_time = self.start_time

    def __str__(self):
        return 'Timer(start_time=%.3e, last_lap_time=%.3e)' % (self.start_time, self.last_lap_time)

    def __repr__(self):
        return self.__str__()

    def reset(self):
        self.start_time = time.time()
        self.last_lap_time = self.start_time

    def update(self):
        self.last_lap_time = time.time()

    def lap(self):
        current_time = time.time()
        lap_time = current_time - self.last_lap_time
        self.last_lap_time = current_time
        print('Lap time: %.3e s' % lap_time)

    def stop(self):
        total_time = time.time() - self.start_time
        return print('Total time: %.3e s' % total_time)
    

def ax_config(ax,  
            title=None, title_fontsize=13.,
            xlabel=None, xlabel_fontsize=13.,
            ylabel=None, ylabel_fontsize=13.,
            spine_width=0.8, spine_color='gray',
            tick_major=True, 
            xtick_major=None, ytick_major=None,
            tick_major_direction='in', tick_major_color='gray',
            tick_major_labelsize=12., tick_major_labelcolor='black',
            tick_major_length=3., tick_major_width=0.8, tick_major_pad=4.,
            tick_minor=True, 
            xtick_minor=None, ytick_minor=None, 
            tick_minor_direction='in', tick_minor_color='gray',
            tick_minor_labelsize=12., tick_minor_labelcolor='black',
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
            legend_fontsize=12., legend_ncol=1):
    """
    Configure the plot with title, labels, spine, tick, grid, and legend parameters for a given Axes object.

    Parameters:
    ----------
    ax (matplotlib.axes.Axes): The Axes object to configure.
    title (str): Title of the plot.
    title_fontsize (float): Font size of the title.
    xlabel (str): Label of the x-axis.
    xlabel_fontsize (float): Font size of the x-axis label.
    ylabel (str): Label of the y-axis.
    ylabel_fontsize (float): Font size of the y-axis label.
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
    """
    if title:
        ax.set_title(title, fontsize=title_fontsize)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)

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
    
    if legend:
        ax.legend(loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor, ncol=legend_ncol, 
                  edgecolor=legend_edgecolor, facecolor=legend_facecolor, 
                  framealpha=legend_framealpha, fontsize=legend_fontsize)


def ax3d_config(ax, axis3don=True, view_angle=[5, 45], 
                box_aspect=None, axis_limits=None,
                title=None, title_size=12., title_pad=0.,
                xlabel=None, ylabel=None, zlabel=None,
                labelsize=12, labelpad=[-5,-5,-5], label_rotation=[0,0,-90],
                pane_color='w', spine_color='grey', spine_width=0.5,
                tick_labelsize=10, tick_pad=[-5,-4,-1.5], tick_color='k',
                tick_inward_length=0, tick_outward_length=0.3, tick_linewidth=0.5, 
                grid_color='lightgray', grid_linewidth=0.5, grid_linestyle=':',):
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
    
    if title:
        ax.set_title(title, fontsize=title_size, pad=title_pad)

    if isinstance(labelsize, int) or isinstance(labelsize, float):
        labelsize = [labelsize]*3 
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=labelpad[0], fontsize=labelsize[0], rotation=label_rotation[0])
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=labelpad[1], fontsize=labelsize[1], rotation=label_rotation[1])
    if zlabel:
        ax.set_zlabel(zlabel, labelpad=labelpad[2], fontsize=labelsize[2], rotation=label_rotation[2])

    if axis_limits:
        ax.set_xlim(*axis_limits[:2])
        ax.set_ylim(*axis_limits[2:4])
        ax.set_zlim(*axis_limits[4:])
    if box_aspect:
        ax.set_box_aspect(box_aspect)


def plot_latex_render(flag=True):
    """
    Enable or disable LaTeX rendering in matplotlib plots.

    Parameters:
    ----------
    flag (bool): Whether to enable or disable LaTeX rendering.
                If True, enable LaTeX rendering and set the font to 'Computer Modern Roman',
                        which is the default font used in LaTeX.
                If False, disable LaTeX rendering and reset the font to default settings of matplotlib. 

    Example:
    ----------
    >>> plot_latex_render(True)
    """
    if flag:
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amssymb,amsmath,amsthm,bm,bbm}'
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Computer Modern Roman', 'Times New Roman'] + plt.rcParams['font.serif']
    else:
        plt.rc('text', usetex=False)
        plt.rcParams['font.family'] = 'sans-serif'


def colorbar_config(fig, ax, img, label=None, labelsize=10, 
                    shrink=0.7, aspect=20, pad=0, orientation='vertical', fraction=0.1,
                    tick_length=2, tick_width=0.5, 
                    outline_visible=True, 
                    outline_linewidth=0.5, outline_edgecolor='black', outline_linestyle='-'):
    """
    Add colorbar to the figure and configure the appearance and label of the colorbar.

    Parameters
    ----------
    fig : Figure object
        The figure object associated with the colorbar.
    ax : Axes object
        The axes object associated with the colorbar.
    img : ScalarMappable object
        The image object created by ax.imshow() or similar functions.
    label : str
        The label for the colorbar.
    labelsize : int, optional
        The font size of the label, default is 10.
    shrink : float, optional
        The shrink factor of the colorbar, default is 0.7.
    aspect : int, optional
        The aspect ratio of the colorbar, default is 20.
    pad : float, optional
        The padding between the colorbar and the axes, default is 0.
    orientation : str, optional
        The orientation of the colorbar, default is 'vertical'.
    fraction : float, optional
        The fraction of the axes that the colorbar occupies, default is 0.1.
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
    """
    
    # Add colorbar
    cbar = fig.colorbar(img, ax=ax, shrink=shrink, aspect=aspect, pad=pad, orientation=orientation, fraction=fraction)
    cbar.ax.tick_params(labelsize=labelsize, length=tick_length, width=tick_width)

    if label:
        cbar.set_label(label, fontsize=labelsize)

    if outline_visible:
        cbar.outline.set_visible(True)
        cbar.outline.set_linewidth(outline_linewidth)
        cbar.outline.set_edgecolor(outline_edgecolor)
        cbar.outline.set_linestyle(outline_linestyle)
    else:
        cbar.outline.set_visible(False)