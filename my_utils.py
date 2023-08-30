import os
import glob
import warnings

class helper(): 

    def reader_paths(folder, filetype, name = "", internal = True): 
        if internal:
            path = f"{os.path.dirname(__file__)}{folder}"
            reader_path = path+filetype
        else:
            cwd = os.path.abspath(os.getcwd())
            reader_path = cwd + folder + filetype
        files = glob.glob(reader_path)
        files = [i for i in files if name in i]
        if len(files) <= 0: 
            warnings.warn("WARNING: folder is empty")
            latest = ""
        else:
            latest = max(files, key = os.path.getctime)
        return files, latest
    
    def folder_path(folder): 
        path = f"{os.path.dirname(__file__)}/{folder}"
        return path
