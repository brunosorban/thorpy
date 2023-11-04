import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List


class DataHandler:
    """
    Class to handle saving and loading of trajectory data. Also contains a method to plot the data.
    """

    @staticmethod
    def save(path: str, file_name: str, data_dict: dict) -> None:
        """
        Save a dictionary to a json file at the specified path with the given name.

        Args:
            path: The directory path where the file should be saved.
            file_name: The name of the file.
            data_dict: The dictionary to be saved.
        """

        def serialize(obj):
            """Function to handle numpy objects during json.dump"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(
                obj,
                (
                    np.int_,
                    np.intc,
                    np.intp,
                    np.int8,
                    np.int16,
                    np.int32,
                    np.int64,
                    np.uint8,
                    np.uint16,
                    np.uint32,
                    np.uint64,
                ),
            ):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.complex_):
                return {"real": obj.real, "imag": obj.imag}
            else:
                raise TypeError(f"Unserializable object {obj} of type {type(obj)}")

        # Ensure the directory exists
        if not os.path.exists(path):
            os.makedirs(path)

        with open(f"{path}/{file_name}", "w") as file:
            json.dump(data_dict, file, default=serialize, indent=4)

    @staticmethod
    def load(path: str, file_name: str) -> dict:
        """
        Load a dictionary from a file at the specified path with the given name.

        Args:
            path: The directory path where the file is located.
            file_name: The name of the file.

        Returns:
            The loaded dictionary.
        """

        def deserialize(dct):
            """Function to handle deserialization of numpy objects from JSON"""
            if "real" in dct and "imag" in dct:
                return np.complex_(complex(dct["real"], dct["imag"]))
            # convert lists to numpy arrays
            for key, value in dct.items():
                if isinstance(value, list):
                    dct[key] = np.array(value)

            return dct

        with open(f"{path}/{file_name}", "r") as file:
            return json.loads(file.read(), object_hook=deserialize)

    @staticmethod
    def plot_data(path: str, file_names: Union[str, List[str]], x_key: str, y_keys: Union[str, List[str]]) -> None:
        """
        Plot data using the specified X and Y keys from the loaded dictionary.

        Args:
            path: The directory path where the file is located.
            file_names: The name(s) of the file(s).
            x_key: The key for the X-axis data.
            y_key: The key for the Y-axis data. One may give a list of keys to plot multiple lines on the same plot.
        """
        if isinstance(file_names, str):
            file_names = [file_names]

        if isinstance(y_keys, str):
            y_keys = [y_keys]

        data_dicts = {}

        plt.figure(figsize=(10, 6))
        for file_name in file_names:
            data_dicts[file_name] = DataHandler.load(path, file_name)

            for y_key in y_keys:
                # Check if the keys are in the dictionary
                if x_key not in data_dicts[file_name] or y_key not in data_dicts[file_name]:
                    raise ValueError("Specified keys not found in the data dictionary")

                x_data = np.array(data_dicts[file_name][x_key])
                y_data = np.array(data_dicts[file_name][y_key])
                plt.plot(x_data, y_data, label=y_key)

        y_str = ""
        for key in y_keys:
            if len(y_str) == 0:
                y_str += key
            else:
                y_str += f", {key}"

        plt.xlabel(x_key)
        plt.ylabel(y_str)
        plt.title(f"Plot of {y_key} vs {x_key}")
        plt.legend()
        plt.grid(True)
        plt.show()
