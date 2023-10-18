import json
import matplotlib.pyplot as plt
from typing import Union, List

class Trajectory:
    
    @staticmethod
    def save(path: str, file_name: str, data_dict: dict) -> None:
        """
        Save a dictionary to a file at the specified path with the given name.
        
        :param path: The directory path where the file should be saved.
        :param file_name: The name of the file.
        :param data_dict: The dictionary to be saved.
        """
        with open(f"{path}/{file_name}", "w") as file:
            json.dump(data_dict, file)
    
    @staticmethod
    def load(path: str, file_name: str) -> dict:
        """
        Load a dictionary from a file at the specified path with the given name.
        
        :param path: The directory path where the file is located.
        :param file_name: The name of the file.
        :return: The loaded dictionary.
        """
        with open(f"{path}/{file_name}", "r") as file:
            return json.load(file)
        
    @staticmethod
    def plot_data(path: str, file_names: Union[str, List[str]], x_key: str, y_key: str) -> None:
        """
        Plot data using the specified X and Y keys from the loaded dictionary.
        
        :param path: The directory path where the file is located.
        :param file_name: The name of the file.
        :param x_key: The key for the X-axis data.
        :param y_key: The key for the Y-axis data.
        """
        if file_names is str:
            file_names = [file_names]
        
        data_dicts = {}
        
        plt.figure(figsize=(10, 6))
        for file_name in file_names:
            data_dicts[file_name] = Trajectory.load(path, file_name)
        
            # Check if the keys are in the dictionary
            if x_key not in data_dicts[file_name] or y_key not in data_dicts[file_name]:
                raise ValueError("Specified keys not found in the data dictionary")

            plt.plot(data_dicts[file_name][x_key], data_dicts[file_name][y_key], marker='o')
            
        plt.xlabel(x_key)
        plt.ylabel(y_key)
        plt.title(f"Plot of {y_key} vs {x_key}")
        plt.grid(True)
        plt.show()

# import json

# class Trajectory:

#     @staticmethod
#     def save(path: str, file_name: str, data_dict: dict, meta_dict: dict) -> None:
#         """
#         Save a dictionary and its metadata to a file at the specified path with the given name.
        
#         :param path: The directory path where the file should be saved.
#         :param file_name: The name of the file.
#         :param data_dict: The dictionary to be saved.
#         :param meta_dict: The metadata dictionary.
#         """
#         with open(f"{path}/{file_name}", "w") as file:
#             file.write("META_START\n")
#             json.dump(meta_dict, file)
#             file.write("\nMETA_STOP\n")
#             file.write("DATA_BEGIN\n")
#             json.dump(data_dict, file)
#             file.write("\nDATA_STOP\n")

#     @staticmethod
#     def load(path: str, file_name: str) -> (dict, dict):
#         """
#         Load a dictionary and its metadata from a file at the specified path with the given name.
        
#         :param path: The directory path where the file is located.
#         :param file_name: The name of the file.
#         :return: A tuple containing the loaded dictionary and its metadata.
#         """
#         with open(f"{path}/{file_name}", "r") as file:
#             content = file.read()

#             # Extract metadata and data from the content
#             meta_start = content.index("META_START") + len("META_START\n")
#             meta_stop = content.index("META_STOP")
#             data_start = content.index("DATA_BEGIN") + len("DATA_BEGIN\n")
#             data_stop = content.index("DATA_STOP")
            
#             meta_str = content[meta_start:meta_stop].strip()
#             data_str = content[data_start:data_stop].strip()

#             meta_dict = json.loads(meta_str)
#             data_dict = json.loads(data_str)

#             return data_dict, meta_dict
