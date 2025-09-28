import os
import json
import pandas as pd
import zipfile
from pathlib import Path
import glob
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MuShroomDataLoader:
    """
    Professional data loader for Mu-SHROOM hallucination detection project.
    Handles multiple file formats and provides comprehensive dataset analysis.
    """
    
    def __init__(self, base_paths=None):
        """
        Initialize the data loader with specified base paths.
        
        Args:
            base_paths (dict): Dictionary mapping dataset names to their paths
        """
        if base_paths is None:
            self.base_paths = {
                'test_labeled': '/kaggle/input/test-labeled/v1/',
                'test_unlabeled': '/kaggle/input/test-unlabeled/v1/',
                'train_data': '/kaggle/input/train-data/train/'
            }
        else:
            self.base_paths = base_paths
            
        self.loaded_datasets = {}
        self.file_registry = {}
        
    def discover_files(self):
        """
        Discover all files in the specified data directories.
        
        Returns:
            dict: Dictionary mapping dataset names to lists of file information
        """
        logger.info("Starting file discovery process")
        
        discovered_files = {}
        
        for dataset_name, path in self.base_paths.items():
            logger.info(f"Exploring dataset: {dataset_name}")
            logger.info(f"Path: {path}")
            
            if not os.path.exists(path):
                logger.warning(f"Path does not exist: {path}")
                discovered_files[dataset_name] = []
                continue
            
            files = []
            
            try:
                for root, dirs, filenames in os.walk(path):
                    for filename in filenames:
                        file_path = os.path.join(root, filename)
                        file_size = os.path.getsize(file_path)
                        file_ext = os.path.splitext(filename)[1].lower()
                        
                        relative_path = os.path.relpath(file_path, path)
                        
                        file_info = {
                            'name': relative_path,
                            'full_path': file_path,
                            'size_bytes': file_size,
                            'extension': file_ext,
                            'dataset': dataset_name
                        }
                        
                        files.append(file_info)
                        logger.debug(f"Found file: {relative_path} ({file_size} bytes)")
                
                logger.info(f"Dataset {dataset_name}: Found {len(files)} files")
                discovered_files[dataset_name] = files
                
            except Exception as e:
                logger.error(f"Error exploring {dataset_name}: {str(e)}")
                discovered_files[dataset_name] = []
        
        self.file_registry = discovered_files
        return discovered_files
    
    def load_json_file(self, file_path, max_records=None):
        """
        Load a JSON file and return its contents.
        
        Args:
            file_path (str): Path to the JSON file
            max_records (int, optional): Maximum number of records to load
            
        Returns:
            dict or list: Loaded JSON data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list) and max_records is not None:
                data = data[:max_records]
            
            logger.info(f"Successfully loaded JSON file: {os.path.basename(file_path)}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {str(e)}")
            return None
    
    def load_jsonl_file(self, file_path, max_records=None):
        """
        Load a JSONL file and return its contents as a list.
        
        Args:
            file_path (str): Path to the JSONL file
            max_records (int, optional): Maximum number of records to load
            
        Returns:
            list: List of JSON objects
        """
        try:
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if max_records is not None and i >= max_records:
                        break
                    
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            
            logger.info(f"Successfully loaded JSONL file: {os.path.basename(file_path)} ({len(data)} records)")
            return data
            
        except Exception as e:
            logger.error(f"Error loading JSONL file {file_path}: {str(e)}")
            return None
    
    def load_csv_file(self, file_path, max_records=None):
        """
        Load a CSV file and return as pandas DataFrame.
        
        Args:
            file_path (str): Path to the CSV file
            max_records (int, optional): Maximum number of records to load
            
        Returns:
            pandas.DataFrame: Loaded CSV data
        """
        try:
            if max_records is not None:
                data = pd.read_csv(file_path, nrows=max_records)
            else:
                data = pd.read_csv(file_path)
            
            logger.info(f"Successfully loaded CSV file: {os.path.basename(file_path)} ({data.shape[0]} rows)")
            return data
            
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {str(e)}")
            return None
    
    def extract_and_load_zip(self, file_path, max_records=None):
        """
        Extract and load contents from a ZIP file.
        
        Args:
            file_path (str): Path to the ZIP file
            max_records (int, optional): Maximum number of records to load per file
            
        Returns:
            dict: Dictionary mapping filenames to their loaded data
        """
        try:
            zip_contents = {}
            
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                logger.info(f"ZIP file contains {len(file_list)} files")
                
                for zip_filename in file_list:
                    if zip_filename.endswith('/'):  # Skip directories
                        continue
                    
                    file_ext = os.path.splitext(zip_filename)[1].lower()
                    
                    try:
                        if file_ext == '.json':
                            with zip_ref.open(zip_filename) as f:
                                data = json.load(f)
                                if isinstance(data, list) and max_records is not None:
                                    data = data[:max_records]
                                zip_contents[zip_filename] = data
                                
                        elif file_ext == '.jsonl':
                            data = []
                            with zip_ref.open(zip_filename) as f:
                                for i, line in enumerate(f):
                                    if max_records is not None and i >= max_records:
                                        break
                                    line = line.decode('utf-8').strip()
                                    if line:
                                        data.append(json.loads(line))
                            zip_contents[zip_filename] = data
                            
                        logger.debug(f"Loaded from ZIP: {zip_filename}")
                        
                    except Exception as e:
                        logger.warning(f"Could not load {zip_filename} from ZIP: {str(e)}")
            
            logger.info(f"Successfully processed ZIP file: {os.path.basename(file_path)}")
            return zip_contents
            
        except Exception as e:
            logger.error(f"Error processing ZIP file {file_path}: {str(e)}")
            return None
    
    def load_file(self, file_info, max_records=1000):
        """
        Load a single file based on its extension.
        
        Args:
            file_info (dict): File information dictionary
            max_records (int): Maximum number of records to load
            
        Returns:
            tuple: (success, data) where success is boolean and data is the loaded content
        """
        file_path = file_info['full_path']
        extension = file_info['extension']
        
        # Skip very small files (likely metadata or empty)
        if file_info['size_bytes'] < 50:
            logger.debug(f"Skipping small file: {file_info['name']} ({file_info['size_bytes']} bytes)")
            return False, None
        
        if extension == '.json':
            data = self.load_json_file(file_path, max_records)
            return data is not None, data
            
        elif extension == '.jsonl':
            data = self.load_jsonl_file(file_path, max_records)
            return data is not None, data
            
        elif extension == '.csv':
            data = self.load_csv_file(file_path, max_records)
            return data is not None, data
            
        elif extension == '.zip':
            data = self.extract_and_load_zip(file_path, max_records)
            return data is not None, data
            
        else:
            logger.debug(f"Unsupported file type: {extension}")
            return False, None
    
    def load_all_datasets(self, max_records_per_file=1000):
        """
        Load all discovered datasets.
        
        Args:
            max_records_per_file (int): Maximum records to load per file
            
        Returns:
            dict: Dictionary mapping dataset names to their loaded data
        """
        logger.info("Starting dataset loading process")
        
        if not self.file_registry:
            self.discover_files()
        
        loaded_data = {}
        
        for dataset_name, files in self.file_registry.items():
            logger.info(f"Loading dataset: {dataset_name}")
            
            dataset_data = {}
            successful_loads = 0
            
            for file_info in files:
                success, data = self.load_file(file_info, max_records_per_file)
                
                if success:
                    dataset_data[file_info['name']] = {
                        'data': data,
                        'file_info': file_info,
                        'loaded_at': pd.Timestamp.now()
                    }
                    successful_loads += 1
            
            loaded_data[dataset_name] = dataset_data
            logger.info(f"Dataset {dataset_name}: Successfully loaded {successful_loads}/{len(files)} files")
        
        self.loaded_datasets = loaded_data
        return loaded_data
    
    def analyze_dataset_structure(self, dataset_name=None):
        """
        Analyze the structure of loaded datasets.
        
        Args:
            dataset_name (str, optional): Specific dataset to analyze. If None, analyzes all.
            
        Returns:
            dict: Analysis results
        """
        if not self.loaded_datasets:
            logger.warning("No datasets loaded. Call load_all_datasets() first.")
            return {}
        
        datasets_to_analyze = [dataset_name] if dataset_name else list(self.loaded_datasets.keys())
        analysis_results = {}
        
        for ds_name in datasets_to_analyze:
            if ds_name not in self.loaded_datasets:
                logger.warning(f"Dataset {ds_name} not found in loaded datasets")
                continue
            
            logger.info(f"Analyzing dataset structure: {ds_name}")
            
            dataset = self.loaded_datasets[ds_name]
            analysis = {
                'file_count': len(dataset),
                'total_records': 0,
                'record_types': {},
                'common_keys': None,
                'sample_records': [],
                'data_schema': {}
            }
            
            all_keys_sets = []
            
            for filename, file_data in dataset.items():
                data = file_data['data']
                
                if isinstance(data, list):
                    analysis['total_records'] += len(data)
                    analysis['record_types'][filename] = f"list ({len(data)} items)"
                    
                    if len(data) > 0 and isinstance(data[0], dict):
                        keys = set(data[0].keys())
                        all_keys_sets.append(keys)
                        
                        # Store sample records
                        analysis['sample_records'].extend(data[:2])
                        
                        # Analyze data types
                        for key, value in data[0].items():
                            if key not in analysis['data_schema']:
                                analysis['data_schema'][key] = set()
                            analysis['data_schema'][key].add(type(value).__name__)
                
                elif isinstance(data, dict):
                    # Handle ZIP contents or nested dictionaries
                    for sub_key, sub_data in data.items():
                        if isinstance(sub_data, list) and len(sub_data) > 0:
                            analysis['total_records'] += len(sub_data)
                            if isinstance(sub_data[0], dict):
                                keys = set(sub_data[0].keys())
                                all_keys_sets.append(keys)
                                analysis['sample_records'].extend(sub_data[:2])
                
                elif isinstance(data, pd.DataFrame):
                    analysis['total_records'] += len(data)
                    analysis['record_types'][filename] = f"dataframe ({data.shape[0]}x{data.shape[1]})"
                    keys = set(data.columns)
                    all_keys_sets.append(keys)
                    analysis['sample_records'].extend(data.head(2).to_dict('records'))
            
            # Find common keys across all files
            if all_keys_sets:
                analysis['common_keys'] = set.intersection(*all_keys_sets) if all_keys_sets else set()
                analysis['all_unique_keys'] = set.union(*all_keys_sets) if all_keys_sets else set()
            
            # Clean up data schema
            for key, types in analysis['data_schema'].items():
                analysis['data_schema'][key] = list(types)
            
            analysis_results[ds_name] = analysis
            
            # Log analysis summary
            logger.info(f"Dataset {ds_name} analysis:")
            logger.info(f"  - Files: {analysis['file_count']}")
            logger.info(f"  - Total records: {analysis['total_records']}")
            logger.info(f"  - Common keys: {len(analysis['common_keys'])} keys")
            logger.info(f"  - Unique keys across all files: {len(analysis.get('all_unique_keys', set()))} keys")
        
        return analysis_results
    
    def get_summary_report(self):
        """
        Generate a comprehensive summary report of all loaded data.
        
        Returns:
            dict: Summary report
        """
        if not self.loaded_datasets:
            return {"error": "No datasets loaded"}
        
        total_files = 0
        total_records = 0
        datasets_summary = {}
        
        analysis = self.analyze_dataset_structure()
        
        for dataset_name, dataset_analysis in analysis.items():
            total_files += dataset_analysis['file_count']
            total_records += dataset_analysis['total_records']
            
            datasets_summary[dataset_name] = {
                'files': dataset_analysis['file_count'],
                'records': dataset_analysis['total_records'],
                'common_keys': list(dataset_analysis['common_keys']),
                'sample_schema': dataset_analysis['data_schema']
            }
        
        summary = {
            'total_datasets': len(self.loaded_datasets),
            'total_files': total_files,
            'total_records': total_records,
            'datasets': datasets_summary,
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
        
        return summary

def main():
    """
    Main execution function for the data loader.
    """
    # Initialize the data loader
    loader = MuShroomDataLoader()
    
    # Discover all files
    logger.info("MU-SHROOM Data Loading Pipeline - Starting")
    discovered_files = loader.discover_files()
    
    # Load all datasets
    loaded_data = loader.load_all_datasets(max_records_per_file=1000)
    
    # Generate analysis
    analysis_results = loader.analyze_dataset_structure()
    
    # Generate summary report
    summary = loader.get_summary_report()
    
    # Print summary
    print("\nMU-SHROOM DATA LOADING SUMMARY")
    print("=" * 50)
    print(f"Datasets loaded: {summary['total_datasets']}")
    print(f"Total files: {summary['total_files']}")
    print(f"Total records: {summary['total_records']}")
    
    print("\nDATASET BREAKDOWN:")
    for dataset_name, info in summary['datasets'].items():
        print(f"  {dataset_name}:")
        print(f"    Files: {info['files']}")
        print(f"    Records: {info['records']}")
        print(f"    Common keys: {info['common_keys']}")
    
    return loader, loaded_data, analysis_results, summary

# Execute the pipeline
if __name__ == "__main__":
    loader, data, analysis, summary = main()
    print("\nData loading complete. Ready for preprocessing and feature engineering.")
