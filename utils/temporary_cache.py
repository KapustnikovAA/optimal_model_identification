from os import remove, makedirs, rmdir, listdir
from os.path import exists, join

def remove_cache_folder (folder_name: str,
                         pid: int) -> None:
      folder_name = f"{folder_name}_{pid}"

      if exists(folder_name):
            for file_name in listdir(folder_name):
                  remove(join(folder_name, file_name))
            rmdir(folder_name)
      
def create_cache_folder (folder_name: str,
                         pid: int) -> None:
      makedirs(f"{folder_name}_{pid}")