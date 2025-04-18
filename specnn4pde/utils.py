"""
utils.py

This module provides various utility functions for system information, file handling, and timing.

Brief description:
    - pkg_system_info: Display package versions, system information, and GPU details.
    - convert2pdf: Convert images to PDF format and optionally merge them into a single PDF file.
    - func_timer: A decorator to measure the execution time of a function.
    - timer: A class for measuring elapsed time and recording lap times.
    - file_finder: List and sort files in a directory with a specific prefix.

For more information, see the documentation of each function or class.
"""

__all__ = ['pkg_system_info', 'convert2pdf', 'func_timer', 'timer', 'file_finder',
           ]

import os
import platform
import psutil
import pandas as pd
from datetime import datetime
from IPython.display import display, HTML
import GPUtil
import importlib
import subprocess
from PIL import Image
import PyPDF2

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


def convert2pdf(directory, inkscape_path=None, extension=(".pdf", ".png", ".jpg", ".jpeg"), 
                merge=False, output_path=None, output_name="merged.pdf"):
    """
    Convert images in the specified directory to PDF format. Optionally merge them into a single PDF file.

    This function is only tested on Windows and it requires 
    Inkscape to be installed on your system for SVG conversion.
    You can download Inkscape from https://inkscape.org/release/ and install it.

    Parameters:
    ----------
    directory (str): The path to the directory containing images.
    inkscape_path (str): The path to the Inkscape executable (if needed for SVG conversion).
                         If inkscape is in the system PATH, this parameter can be ignored.
    extension (tuple): A tuple of file extensions to include in the conversion.
    merge (bool): Whether to merge all images into a single PDF file.
    output_path (str): The path to save the output PDF file(s).
    output_name (str): The name of the merged PDF file (if merge is True).

    Example:
    ----------
    >>> convert2pdf(r'D:/path/to/your/directory')
    """

    if inkscape_path is None:
        inkscape_path = "inkscape"

    if output_path is None:
        output_path = directory

    if merge and (output_name in os.listdir(output_path)):
        raise ValueError("The output directory already contains a file named 'merged.pdf'. Please provide a different name for the output file.")

    temp_pdfs = []
    remove_pdfs = []
    for filename in os.listdir(directory):
        if filename.endswith(extension):
            image_file = os.path.join(directory, filename)
            pdf_file = os.path.join(output_path, os.path.splitext(filename)[0] + ".pdf")
            temp_pdfs.append(pdf_file)
            
            if filename.lower().endswith(".pdf"):
                pass
            elif filename.lower().endswith(".svg"):
                command = f'"{inkscape_path}" "{image_file}" --export-filename="{pdf_file}"'
                subprocess.run(command, shell=True)
                remove_pdfs.append(pdf_file)
            else:
                image = Image.open(image_file).convert('RGB')
                image.save(pdf_file)
                remove_pdfs.append(pdf_file)
        
    print(f"Conversion completed! {len(temp_pdfs)} PDF files have been created.")

    if merge:
        merger = PyPDF2.PdfMerger()
        for pdf in temp_pdfs:
            merger.append(pdf)

        merger.write(os.path.join(output_path, output_name))
        merger.close()

        for pdf in remove_pdfs:
            os.remove(pdf)
        
        print(f"PDF files have been merged into {output_name}.")


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
    
def file_finder(dir, ls = False, prefix = '', sort = 'time'):
    """
    This function lists files in a directory with a specific prefix and sorts them.
    It can also display the files in a numbered list.

    Parameters
    ----------
    dir : str
        The directory to search for files.
    ls : bool, optional
        If True, display the files in a numbered list. Default is False.
    prefix : str, optional
        The prefix to filter files. Default is ''.
    sort : str, optional
        The sorting method. Can be 'name' or 'time'. Default is 'time'.
    """
    files = os.listdir(dir)
    files = [f for f in files if f.startswith(prefix)]  # filter
    if sort == 'name':
        files.sort()
    elif sort == 'time':
        files.sort(key=lambda x: x[-16:-4])
    if ls:
        for i, item in enumerate(files):
            print(i, item)
    return files