"""Input parameter management for FBPIC simulations.

This module provides a class-based system for managing and saving simulation
input parameters to INI configuration files.
"""

import configparser
from pathlib import Path
from typing import Any, Union


class InputParameters:
    """Manages simulation input parameters and saves them to INI files.
    
    This class uses a nested dictionary structure where the first level
    represents parameter groups (e.g., 'Laser', 'Gas', 'Simulation') and
    the second level contains parameter names and their values.
    
    Attributes:
        input_params (Dict): Nested dictionary storing all parameters
    """
    
    input_params: dict[str, dict[str, Any]] = {}
    
    @classmethod
    def add(cls, group: str, params: dict[str, Any]) -> None:
        """Add parameters to a specific group.
        
        Args:
            group: The parameter group name (e.g., 'Laser', 'Gas', 'Simulation')
            params: Dictionary of parameter names and values to add
        """
        if group not in cls.input_params:
            cls.input_params[group] = {}
        
        cls.input_params[group].update(params)
    
    @classmethod
    def clear(cls) -> None:
        """Clear all stored parameters."""
        cls.input_params = {}
    
    @classmethod
    def save_to_ini(cls, save_folder: str, filename: str = 'input.ini') -> None:
        """Save all parameters to an INI configuration file.
        
        Args:
            save_folder: Directory path where the INI file will be saved
            filename: Name of the INI file (default: 'input.ini')
        """
        config = configparser.ConfigParser()
        config.optionxform = str
        
        # Dynamically create sections from input_params
        for group_name, group_params in cls.input_params.items():
            config[group_name] = {}
            for param_name, param_value in group_params.items():
                config[group_name][param_name] = str(param_value)
        
        # Create directory if it doesn't exist
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        
        # Write to file
        config_path = Path(save_folder) / filename
        with open(config_path, 'w') as configfile:
            config.write(configfile)
        
        print(f"Input parameters saved to {config_path}")
    
    @classmethod
    def get_params(cls, group: str = None) -> Union[dict, dict[str, dict]]:
        """Get stored parameters.
        
        Args:
            group: Optional group name. If None, returns all parameters.
        
        Returns:
            Dictionary of parameters for the specified group, or all parameters
        """
        if group is None:
            return cls.input_params
        return cls.input_params.get(group, {})
    
    @classmethod
    def print_summary(cls) -> None:
        """Print a summary of all stored parameters."""
        print("\n=== Input Parameters Summary ===")
        for group_name, group_params in cls.input_params.items():
            print(f"\n[{group_name}]")
            for param_name, param_value in group_params.items():
                print(f"  {param_name}: {param_value}")
        print("\n================================\n")

