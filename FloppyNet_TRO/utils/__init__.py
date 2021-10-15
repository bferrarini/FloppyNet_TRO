
import os
from pathlib import Path
from time import time
import os, shutil
import sys

'''
    Creates a directory trees is not exists
'''

def mkdirs(directory):
    
    created = False
    
    if not os.path.exists(directory):
        os.makedirs(directory)   
        created = True
        
    return created

'''
    Clears the content of a folder
'''

def clear_dir(folder):
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))    

'''
    Returns 'value' is it is not None, 'default_on_none' value otherwise
'''

def assign(value, default_on_none):
    return value if value is not None else default_on_none


'''
    Returns a file basename from  fullpath
'''

def basename_without_extension(filename): 
    return Path(filename).stem


def write_check(filename, to_override):
    
    if to_override or not os.path.exists(filename):
        return True
    else:
        return False

def countdown(t):
    while t:
        mins, secs = divmod(t, 60)
        timer = '{:02d}:{:02d}'.format(mins, secs)
        print(timer, end="\r")
        time.sleep(1)
        t -= 1
        

'''
    Returns a string with executing module name
'''

def get_current_module_name():
    module_name = sys.modules['__main__'].__file__
    module_name = os.path.basename(module_name)
    module_name = module_name.split(".")[0]
    
    return module_name

'''
    A stopwatch
'''

class Stopwatch(object):
    
    def __init__(self):
        self._start = 0
        self._stop = 0
        self._t = 0
        
    def __str__(self):
        s = "{:0.3f}".format(self._t)
        return s
        
    def start(self, reset = False):
        if reset:
            self.reset()
        self._start = time()
        
    def stop(self):
        self._stop = time()
        self._t += self._stop - self._start
        
    def reset(self):
        self._start = 0
        self._stop = 0
        self._t = 0
        
        
        
        
        
        
