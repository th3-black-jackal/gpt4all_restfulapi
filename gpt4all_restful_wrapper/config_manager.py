import os
import configparser
from typing import List, Dict, Any, Optional
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QLabel, QTabWidget, 
                            QWidget, QGroupBox, QFormLayout, QLineEdit, QPushButton, 
                            QSpinBox, QDoubleSpinBox, QCheckBox, QHBoxLayout, 
                            QComboBox, QTextEdit, QMessageBox, QFileDialog)
from PyQt5.QtGui import QFont


class ConfigManager:
    """Manages application configuration"""
    
    def __init__(self, config_file: str = "config.ini"):
        self.config_file = config_file
        self.default_config = {
            'model': {
                'name': 'orca-mini-3b-gguf2-q4_0.gguf',
                'model_path': '',
                'max_tokens': '2000',
                'temperature': '0.1',
                'top_k': '40',
                'top_p': '0.9',
                'repeat_penalty': '1.1'
            },
            'analysis': {
                'default_summaries': 'true',
                'default_gap_analysis': 'true',
                'default_github_extraction': 'true',
                'default_export_format': 'markdown',
                'summary_length': '150',
                'embedding_chunk_size': '1000',
                'cluster_eps': '0.5',
                'cluster_min_samples': '2'
            },
            'ui': {
                'window_width': '1200',
                'window_height': '800',
                'auto_save_results': 'true',
                'results_directory': './results',
                'theme': 'light'
            },
            'advanced': {
                'enable_debug_logging': 'false',
                'max_concurrent_tasks': '2',
                'cache_embeddings': 'true',
                'embedding_cache_size': '100'
            }
        }
        self.config = configparser.ConfigParser()
        self.load_config()
    
    def load_config(self):
        """Load configuration from file or create default"""
        if os.path.exists(self.config_file):
            self.config.read(self.config_file)
            self._migrate_config()
        else:
            self.create_default_config()
    
    def _migrate_config(self):
        """Migrate old config files to new format"""
        needs_save = False
        
        # Ensure all sections exist
        for section, options in self.default_config.items():
            if not self.config.has_section(section):
                self.config[section] = options
                needs_save = True
        
        # Ensure all options exist
        for section, options in self.default_config.items():
            for option, default_value in options.items():
                if not self.config.has_option(section, option):
                    self.config.set(section, option, default_value)
                    needs_save = True
        
        if needs_save:
            self.save_config()
    
    def create_default_config(self):
        """Create default configuration file"""
        for section, options in self.default_config.items():
            self.config[section] = options
        self.save_config()
    
    def save_config(self):
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            self.config.write(f)
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return {
            'name': self.config.get('model', 'name', fallback=self.default_config['model']['name']),
            'model_path': self.config.get('model', 'model_path', fallback=self.default_config['model']['model_path']),
            'max_tokens': self.config.getint('model', 'max_tokens', fallback=int(self.default_config['model']['max_tokens'])),
            'temperature': self.config.getfloat('model', 'temperature', fallback=float(self.default_config['model']['temperature'])),
            'top_k': self.config.getint('model', 'top_k', fallback=int(self.default_config['model']['top_k'])),
            'top_p': self.config.getfloat('model', 'top_p', fallback=float(self.default_config['model']['top_p'])),
            'repeat_penalty': self.config.getfloat('model', 'repeat_penalty', fallback=float(self.default_config['model']['repeat_penalty']))
        }
    
    def get_analysis_config(self) -> Dict[str, Any]:
        """Get analysis configuration"""
        return {
            'default_summaries': self.config.getboolean('analysis', 'default_summaries', fallback=True),
            'default_gap_analysis': self.config.getboolean('analysis', 'default_gap_analysis', fallback=True),
            'default_github_extraction': self.config.getboolean('analysis', 'default_github_extraction', fallback=True),
            'default_export_format': self.config.get('analysis', 'default_export_format', fallback='markdown'),
            'summary_length': self.config.getint('analysis', 'summary_length', fallback=150),
            'embedding_chunk_size': self.config.getint('analysis', 'embedding_chunk_size', fallback=1000),
            'cluster_eps': self.config.getfloat('analysis', 'cluster_eps', fallback=0.5),
            'cluster_min_samples': self.config.getint('analysis', 'cluster_min_samples', fallback=2)
        }
    
    def get_ui_config(self) -> Dict[str, Any]:
        """Get UI configuration"""
        return {
            'window_width': self.config.getint('ui', 'window_width', fallback=1200),
            'window_height': self.config.getint('ui', 'window_height', fallback=800),
            'auto_save_results': self.config.getboolean('ui', 'auto_save_results', fallback=True),
            'results_directory': self.config.get('ui', 'results_directory', fallback='./results'),
            'theme': self.config.get('ui', 'theme', fallback='light')
        }
    
    def get_advanced_config(self) -> Dict[str, Any]:
        """Get advanced configuration"""
        return {
            'enable_debug_logging': self.config.getboolean('advanced', 'enable_debug_logging', fallback=False),
            'max_concurrent_tasks': self.config.getint('advanced', 'max_concurrent_tasks', fallback=2),
            'cache_embeddings': self.config.getboolean('advanced', 'cache_embeddings', fallback=True),
            'embedding_cache_size': self.config.getint('advanced', 'embedding_cache_size', fallback=100)
        }
    
    def save_model_config(self, model_config: Dict[str, Any]):
        """Save model configuration"""
        for key, value in model_config.items():
            self.config.set('model', key, str(value))
        self.save_config()
    
    def save_analysis_config(self, analysis_config: Dict[str, Any]):
        """Save analysis configuration"""
        for key, value in analysis_config.items():
            self.config.set('analysis', key, str(value).lower())
        self.save_config()
    
    def save_ui_config(self, ui_config: Dict[str, Any]):
        """Save UI configuration"""
        for key, value in ui_config.items():
            self.config.set('ui', key, str(value).lower())
        self.save_config()
    
    def save_advanced_config(self, advanced_config: Dict[str, Any]):
        """Save advanced configuration"""
        for key, value in advanced_config.items():
            self.config.set('advanced', key, str(value).lower())
        self.save_config()
    
    def get_all_config(self) -> Dict[str, Dict[str, Any]]:
        """Get all configuration as a dictionary"""
        return {
            'model': self.get_model_config(),
            'analysis': self.get_analysis_config(),
            'ui': self.get_ui_config(),
            'advanced': self.get_advanced_config()
        }
    
    def save_all_config(self, config_dict: Dict[str, Dict[str, Any]]):
        """Save all configuration from dictionary"""
        for section, options in config_dict.items():
            for key, value in options.items():
                self.config.set(section, key, str(value))
        self.save_config()
    
    def refresh_config(self):
        """Reload configuration from file"""
        self.load_config()