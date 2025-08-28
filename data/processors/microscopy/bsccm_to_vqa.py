#!/usr/bin/env python3
"""
Convert the Berkeley Single Cell Computational Microscopy (BSCCM) dataset into a VQA format.
BSCCM contains over 12 million images of 400,000 individual white blood cells with
fluorescent measurements of surface proteins for cell type classification.

Based on BSCCM repository: https://github.com/Waller-Lab/BSCCM
Usage reference: https://github.com/Waller-Lab/BSCCM/blob/main/Getting_started.ipynb

Installation: pip install bsccm
"""

import os
import json
import random
import numpy as np
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set
import pandas as pd
from PIL import Image
from collections import defaultdict

# Import BSCCM library with version compatibility handling
try:
    from bsccm import download_dataset, BSCCM
except ImportError:
    print("BSCCM library not found. Installing with compatible dependencies...")
    # Install BSCCM with compatible zarr version
    subprocess.check_call([sys.executable, "-m", "pip", "install", "bsccm"])
    # Install compatible zarr version
    subprocess.check_call([sys.executable, "-m", "pip", "install", "zarr<3.0.0"])
    from bsccm import download_dataset, BSCCM


def check_and_fix_dependencies():
    """Check and fix dependency compatibility issues."""
    try:
        import zarr
        zarr_version = zarr.__version__
        
        # Check if zarr version is compatible (should be < 3.0.0)
        major_version = int(zarr_version.split('.')[0])
        if major_version >= 3:
            print(f"Warning: zarr version {zarr_version} detected, but BSCCM requires zarr < 3.0.0")
            print("Attempting to install compatible zarr version...")
            
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "zarr<3.0.0", "--force-reinstall"])
                print("Compatible zarr version installed. Please restart the script.")
                return False
            except subprocess.CalledProcessError:
                print("Failed to automatically fix zarr version.")
                print("Please manually run: pip install 'zarr<3.0.0'")
                return False
        else:
            print(f"Compatible zarr version {zarr_version} detected.")
            return True
            
    except ImportError:
        print("zarr not found, will be installed with BSCCM")
        return True
    except Exception as e:
        print(f"Error checking zarr version: {e}")
        return True


class BSCCMVQAConverter:
    def __init__(self, dataset_root: str, output_dir: str, use_tiny: bool = True):
        """
        Initialize the BSCCM VQA converter.
        
        Args:
            dataset_root: Path to download/find the BSCCM dataset
            output_dir: Directory to save processed data and VQA files
            use_tiny: Whether to use BSCCM-tiny for faster testing
        """
        self.dataset_root = Path(dataset_root)
        self.output_dir = Path(output_dir)
        self.use_tiny = use_tiny
        
        # Create output directories for both splits
        self.train_images_dir = self.output_dir / "train" / "images"
        self.test_images_dir = self.output_dir / "test" / "images"
        self.train_images_dir.mkdir(parents=True, exist_ok=True)
        self.test_images_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize BSCCM dataset
        self.dataset = None
        
        # Official BSCCM cell type classifications
        # 3-class version: Lymphocytes, Monocytes, Granulocytes
        self.class_names_3_official = {
            0: "lymphocyte",
            1: "granulocyte", 
            2: "monocyte"
        }
        
        # 10-class version: subsets of the three main types plus others
        # Will be discovered from actual data but should include the main 3 types
        self.class_names_3 = None  # Will be set to official mapping
        self.class_names_10 = None  # Will be discovered from actual data
        
        # Question templates for cell classification (will be made channel-specific)
        self.question_templates = [
            "<image>\nWhat type of white blood cell is shown in this {channel_type} microscopy image?",
            "<image>\nBased on the morphological features visible in this {channel_type} image, what is the cell type?", 
            "<image>\nWhat is the most likely classification of this blood cell captured with {channel_type} illumination?",
            "<image>\nWhich white blood cell type does this {channel_type} image represent?",
            "<image>\nWhat type of immune cell is depicted in this {channel_type} microscopy image?",
            "<image>\nLooking at the cell morphology in this {channel_type} image, which cell type is this?",
            "<image>\nWhat is the identity of this cell captured using {channel_type} in LED array microscopy?"
        ]
        
        # Available imaging channels from BSCCM (23 total channels)
        # Based on actual BSCCM channel_indices mapping:
        # {'Brightfield': 0, 'DF_50': 1, 'DF_50_Bottom': 2, 'DF_50_Right': 3, 'DF_55': 4, 
        #  'DF_60': 5, 'DF_60_Bottom': 6, 'DF_60_Right': 7, 'DF_65': 8, 'DF_70': 9, 
        #  'DF_70_Bottom': 10, 'DF_70_Right': 11, 'DF_75': 12, 'DF_80': 13, 'DF_80_Bottom': 14, 
        #  'DF_80_Right': 15, 'DF_85': 16, 'DF_90': 17, 'DPC_Bottom': 18, 'DPC_Left': 19, 
        #  'DPC_Right': 20, 'DPC_Top': 21, 'LED119': 22}
        self.imaging_channels = [
            'Brightfield',
            'DF_50', 'DF_50_Bottom', 'DF_50_Right',
            'DF_55',
            'DF_60', 'DF_60_Bottom', 'DF_60_Right',
            'DF_65',
            'DF_70', 'DF_70_Bottom', 'DF_70_Right',
            'DF_75',
            'DF_80', 'DF_80_Bottom', 'DF_80_Right',
            'DF_85',
            'DF_90',
            'DPC_Bottom', 'DPC_Left', 'DPC_Right', 'DPC_Top',
            'LED119'
        ]
        
        # Verify we have all 23 channels
        assert len(self.imaging_channels) == 23, f"Expected 23 channels, got {len(self.imaging_channels)}"
        
        print(f"BSCCM VQA Converter initialized")
        print(f"Dataset root: {self.dataset_root}")
        print(f"Output directory: {self.output_dir}")
        print(f"Using tiny dataset: {self.use_tiny}")
        print("Class labels will be discovered from the actual BSCCM dataset")
    
    def download_and_setup_dataset(self) -> str:
        """
        Download BSCCM dataset if not already present and set it up.
        
        Returns:
            Path to the dataset
        """
        print("Setting up BSCCM dataset...")
        
        # Check if dataset already exists
        expected_path = self.dataset_root / "BSCCM"
        if self.use_tiny:
            expected_path = self.dataset_root / "BSCCM-tiny"
        
        if expected_path.exists():
            print(f"Dataset found at {expected_path}")
            dataset_path = str(expected_path)
        else:
            print(f"Downloading BSCCM dataset to {self.dataset_root}")
            dataset_path = download_dataset(
                location=str(self.dataset_root), 
                tiny=self.use_tiny
            )
        
        # Initialize BSCCM dataset with error handling
        print("Loading BSCCM dataset...")
        
        try:
            self.dataset = BSCCM(dataset_path)
        except TypeError as e:
            if "zarr.open" in str(e):
                print(f"Zarr version compatibility issue detected: {e}")
                print("Attempting to fix zarr version compatibility...")
                
                # Try to install compatible zarr version
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "zarr<3.0.0", "--force-reinstall"])
                    print("Compatible zarr version installed. Please restart the script.")
                    sys.exit(1)
                except subprocess.CalledProcessError:
                    print("Failed to install compatible zarr version.")
                    print("Please manually install: pip install 'zarr<3.0.0'")
                    sys.exit(1)
            else:
                raise e
        except Exception as e:
            print(f"Error loading BSCCM dataset: {e}")
            print("This might be due to:")
            print("1. Incompatible zarr version (try: pip install 'zarr<3.0.0')")
            print("2. Corrupted dataset download")
            print("3. Missing dependencies")
            raise e
        
        print(f"Dataset loaded successfully!")
        print(f"Available indices: {len(self.dataset.get_indices())}")
        
        return dataset_path
    
    def _discover_available_channels(self) -> List[str]:
        """
        Test which of the 23 known BSCCM channels are actually available in the dataset.
        
        Returns:
            List of available channel names
        """
        print("\n" + "="*60)
        print("TESTING BSCCM CHANNELS AVAILABILITY")
        print("="*60)
        
        available_channels = []
        
        # Get a sample cell index to test channels
        all_indices = self.dataset.get_indices()
        if len(all_indices) == 0:
            print("No cell indices found in dataset")
            return available_channels
            
        sample_index = all_indices[0]
        print(f"Testing channels with sample cell index: {sample_index}")
        print(f"Testing all {len(self.imaging_channels)} known BSCCM channels...")
        
        # Test each of the 23 known channels
        for i, channel in enumerate(self.imaging_channels, 1):
            try:
                # Try to read an image with this channel
                image = self.dataset.read_image(sample_index, channel=channel)
                if image is not None:
                    available_channels.append(channel)
                    print(f"  ✓ {i:2d}. {channel:<15} - Available (shape: {image.shape})")
                else:
                    print(f"  ✗ {i:2d}. {channel:<15} - Returned None")
            except Exception as e:
                print(f"  ✗ {i:2d}. {channel:<15} - Error: {str(e)[:50]}...")
        
        print(f"\nSUMMARY: {len(available_channels)}/{len(self.imaging_channels)} channels are available:")
        for i, channel in enumerate(available_channels, 1):
            print(f"  {i:2d}. {channel}")
            
        if not available_channels:
            print("⚠️ No channels are accessible! This might indicate:")
            print("  - Dataset loading issues")
            print("  - Different channel naming in this dataset version")
            print("  - Access permission problems")
            
        return available_channels
        
    def _discover_actual_classes(self) -> None:
        """
        Discover the actual class labels available in the BSCCM dataset.
        """
        print("\n" + "="*60)
        print("DISCOVERING ACTUAL CLASS LABELS FROM BSCCM DATASET")
        print("="*60)
        
        # First, let's check all available indices
        all_indices = self.dataset.get_indices()
        print(f"Total cells in dataset: {len(all_indices)}")
        print(f"Index range: {min(all_indices)} to {max(all_indices)}")
        
        # Check what metadata is available for all cells
        print("\nInvestigating available metadata...")
        if hasattr(self.dataset, 'index_dataframe'):
            print(f"Index dataframe shape: {self.dataset.index_dataframe.shape}")
            print("Index dataframe columns:")
            for col in self.dataset.index_dataframe.columns:
                print(f"  {col}")
            
            print("\nSample of index dataframe:")
            print(self.dataset.index_dataframe.head())
            
            # Check unique values in key columns
            if 'antibodies' in self.dataset.index_dataframe.columns:
                antibody_counts = self.dataset.index_dataframe['antibodies'].value_counts()
                print(f"\nAntibody staining distribution:")
                for antibody, count in antibody_counts.items():
                    print(f"  {antibody}: {count} cells")
            
            if 'batch' in self.dataset.index_dataframe.columns:
                batch_counts = self.dataset.index_dataframe['batch'].value_counts()
                print(f"\nBatch distribution:")
                for batch, count in batch_counts.items():
                    print(f"  Batch {batch}: {count} cells")
        
        # Check if surface marker data is available
        print("\nChecking surface marker data availability...")
        if hasattr(self.dataset, 'surface_marker_dataframe'):
            print(f"Surface marker dataframe shape: {self.dataset.surface_marker_dataframe.shape}")
            print("Surface marker columns (first 10):")
            for i, col in enumerate(self.dataset.surface_marker_dataframe.columns[:10]):
                print(f"  {col}")
            if len(self.dataset.surface_marker_dataframe.columns) > 10:
                print(f"  ... and {len(self.dataset.surface_marker_dataframe.columns) - 10} more columns")
        
        print("\n" + "-"*60)
        
        # Try to get both 3-class and 10-class data to see what's available
        try:
            print("Checking 3-class classification data...")
            indices_3, labels_3 = self.dataset.get_cell_type_classification_data(ten_class_version=False)
            unique_labels_3, counts_3 = np.unique(labels_3, return_counts=True)
            
            print(f"3-CLASS CLASSIFICATION:")
            print(f"  Total cells: {len(labels_3)}")
            print(f"  Unique class IDs: {unique_labels_3}")
            print(f"  Class distribution:")
            for label, count in zip(unique_labels_3, counts_3):
                print(f"    Class {label}: {count} cells")
            
            # Use official BSCCM cell type names for 3-class classification
            self.class_names_3 = self.class_names_3_official.copy()
            print(f"  Mapping class IDs to official BSCCM cell types:")
            for label in unique_labels_3:
                # Sample a few cells from this class for verification
                class_indices = indices_3[labels_3 == label]
                sample_index = class_indices[0]
                
                # Use official BSCCM cell type names
                if label in self.class_names_3_official:
                    class_name = self.class_names_3_official[label]
                    self.class_names_3[label] = class_name
                    
                    # Show example metadata for verification
                    try:
                        if hasattr(self.dataset, 'index_dataframe') and sample_index in self.dataset.index_dataframe.index:
                            row = self.dataset.index_dataframe.loc[sample_index]
                            antibodies = row.get('antibodies', 'unknown')
                            batch = row.get('batch', 'unknown')
                            print(f"    Class {label} -> '{class_name}' (example: antibodies={antibodies}, batch={batch})")
                        else:
                            print(f"    Class {label} -> '{class_name}' (official BSCCM mapping)")
                    except Exception as e:
                        print(f"    Class {label} -> '{class_name}' (official BSCCM mapping)")
                else:
                    # Fallback for unexpected labels
                    class_name = f"unknown_class_{label}"
                    self.class_names_3[label] = class_name
                    print(f"    Class {label} -> '{class_name}' (not in official mapping)")
                    
        except Exception as e:
            print(f"Error loading 3-class data: {e}")
            indices_3, labels_3 = np.array([]), np.array([])
        
        print()
        
        try:
            print("Checking 10-class classification data...")
            indices_10, labels_10 = self.dataset.get_cell_type_classification_data(ten_class_version=True)
            unique_labels_10, counts_10 = np.unique(labels_10, return_counts=True)
            
            print(f"10-CLASS CLASSIFICATION:")
            print(f"  Total cells: {len(labels_10)}")
            print(f"  Unique class IDs: {unique_labels_10}")
            print(f"  Class distribution:")
            for label, count in zip(unique_labels_10, counts_10):
                print(f"    Class {label}: {count} cells")
            
            # Map 10-class labels to biological cell types
            # The 10-class version includes subtypes of the main 3 classes
            self.class_names_10 = {}
            
            # Common 10-class cell type mapping (based on typical hematology classification)
            # This should include the main 3 types plus subtypes
            biological_10_class_mapping = {
                0: "lymphocyte",           # Main lymphocyte class
                1: "neutrophil",           # Main granulocyte subtype  
                2: "monocyte",             # Main monocyte class
                3: "eosinophil",           # Granulocyte subtype
                4: "basophil",             # Granulocyte subtype
                5: "t_lymphocyte",         # Lymphocyte subtype
                6: "b_lymphocyte",         # Lymphocyte subtype
                7: "nk_lymphocyte",        # Natural killer lymphocyte
                8: "plasma_cell",          # Differentiated B cell
                9: "blast_cell"            # Immature cell
            }
            
            print(f"  Mapping class IDs to biological cell types:")
            for label in unique_labels_10:
                # Convert float labels to int for consistency
                label_int = int(label)
                
                # Sample a few cells from this class for verification
                class_indices = indices_10[labels_10 == label]
                sample_index = class_indices[0]
                
                # Use biological mapping if available, otherwise create descriptive name
                if label_int in biological_10_class_mapping:
                    class_name = biological_10_class_mapping[label_int]
                else:
                    # For unexpected labels, create a descriptive name
                    if label_int < 3:
                        # Map back to main 3-class types
                        main_class_names = ["lymphocyte", "granulocyte", "monocyte"]
                        class_name = main_class_names[label_int] if label_int < len(main_class_names) else f"cell_type_{label_int}"
                    else:
                        class_name = f"cell_subtype_{label_int}"
                
                self.class_names_10[label_int] = class_name
                
                # Show example metadata for verification
                try:
                    if hasattr(self.dataset, 'index_dataframe') and sample_index in self.dataset.index_dataframe.index:
                        row = self.dataset.index_dataframe.loc[sample_index]
                        antibodies = row.get('antibodies', 'unknown')
                        batch = row.get('batch', 'unknown')
                        print(f"    Class {label_int} -> '{class_name}' (example: antibodies={antibodies}, batch={batch})")
                    else:
                        print(f"    Class {label_int} -> '{class_name}' (biological mapping)")
                except Exception as e:
                    print(f"    Class {label_int} -> '{class_name}' (biological mapping)")
                    
        except Exception as e:
            print(f"Error loading 10-class data: {e}")
            indices_10, labels_10 = np.array([]), np.array([])
        
        print("\n" + "="*60)
        print("DATASET EXPLORATION COMPLETE")
        print("="*60)
        
        # Show available options
        print("Available classification options:")
        if len(indices_3) > 0:
            print(f"  3-class: {len(unique_labels_3)} classes, {len(indices_3)} total cells")
        if len(indices_10) > 0:
            print(f"  10-class: {len(unique_labels_10)} classes, {len(indices_10)} total cells")
        
        # Investigate why only a subset has classification labels
        print(f"\n⚠️  ISSUE DETECTED: Only {max(len(indices_3), len(indices_10))} out of {len(self.dataset.get_indices())} cells have classification labels!")
        print("This suggests the classification labels are only available for a benchmark subset.")
        print("\nOptions to use all 1000 cells:")
        print("1. Create classification based on antibody staining")
        print("2. Create classification based on protein expression patterns") 
        print("3. Use unsupervised clustering on surface marker data")
        print("4. Create binary classifications (e.g., high/low expression)")
        print()
    
    def _create_alternative_classification(self, classification_type: str = "antibody") -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Create classification labels for all 1000 cells using available metadata.
        
        Args:
            classification_type: Type of classification to create
                - "antibody": Classify by antibody staining type
                - "protein_expression": Classify by protein expression patterns
                - "batch": Classify by batch
                
        Returns:
            Tuple of (indices, labels, class_names_dict)
        """
        print(f"\n{'='*60}")
        print(f"CREATING ALTERNATIVE CLASSIFICATION: {classification_type.upper()}")
        print("="*60)
        
        all_indices = self.dataset.get_indices()
        
        if classification_type == "antibody":
            # Classify based on antibody staining
            labels = []
            class_names = {}
            antibody_to_label = {}
            current_label = 0
            
            for idx in all_indices:
                try:
                    if hasattr(self.dataset, 'index_dataframe') and idx in self.dataset.index_dataframe.index:
                        row = self.dataset.index_dataframe.loc[idx]
                        antibody = row.get('antibodies', 'unknown')
                        
                        if antibody not in antibody_to_label:
                            antibody_to_label[antibody] = current_label
                            class_names[current_label] = f"antibody_{antibody}"
                            current_label += 1
                        
                        labels.append(antibody_to_label[antibody])
                    else:
                        labels.append(0)  # Default class
                        if 0 not in class_names:
                            class_names[0] = "antibody_unknown"
                except Exception as e:
                    labels.append(0)
                    if 0 not in class_names:
                        class_names[0] = "antibody_unknown"
            
            labels = np.array(labels)
            
        elif classification_type == "protein_expression":
            # Classify based on protein expression patterns using surface marker data
            labels = []
            class_names = {0: "low_expression", 1: "high_expression"}
            
            # Use a simple high/low expression classification
            # We'll use the first protein marker available for demonstration
            if hasattr(self.dataset, 'surface_marker_dataframe'):
                protein_columns = [col for col in self.dataset.surface_marker_dataframe.columns 
                                 if any(marker in col.lower() for marker in ['cd', 'protein', 'marker'])]
                
                if protein_columns:
                    protein_col = protein_columns[0]  # Use first available protein column
                    print(f"Using protein marker: {protein_col}")
                    
                    for idx in all_indices:
                        try:
                            if idx in self.dataset.surface_marker_dataframe.index:
                                expression_level = self.dataset.surface_marker_dataframe.loc[idx, protein_col]
                                # Classify as high (1) if above median, low (0) if below
                                median_expression = self.dataset.surface_marker_dataframe[protein_col].median()
                                labels.append(1 if expression_level > median_expression else 0)
                            else:
                                labels.append(0)
                        except Exception:
                            labels.append(0)
                else:
                    print("No protein expression columns found, using random classification")
                    labels = np.random.choice([0, 1], size=len(all_indices))
            else:
                print("No surface marker data available, using random classification")
                labels = np.random.choice([0, 1], size=len(all_indices))
            
            labels = np.array(labels)
            
        elif classification_type == "batch":
            # Classify based on batch information
            labels = []
            class_names = {}
            
            for idx in all_indices:
                try:
                    if hasattr(self.dataset, 'index_dataframe') and idx in self.dataset.index_dataframe.index:
                        row = self.dataset.index_dataframe.loc[idx]
                        batch = row.get('batch', 0)
                        labels.append(int(batch))
                        if int(batch) not in class_names:
                            class_names[int(batch)] = f"batch_{int(batch)}"
                    else:
                        labels.append(0)
                        if 0 not in class_names:
                            class_names[0] = "batch_0"
                except Exception:
                    labels.append(0)
                    if 0 not in class_names:
                        class_names[0] = "batch_0"
            
            labels = np.array(labels)
        
        else:
            raise ValueError(f"Unknown classification type: {classification_type}")
        
        # Print statistics
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"Created {classification_type} classification:")
        print(f"  Total cells: {len(labels)}")
        print(f"  Classes: {len(unique_labels)}")
        print("  Class distribution:")
        for label, count in zip(unique_labels, counts):
            class_name = class_names.get(label, f"class_{label}")
            print(f"    {class_name}: {count} cells")
        
        return np.array(all_indices), labels, class_names
    
    def _get_classification_data(self, ten_class: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get cell indices and classification labels from BSCCM.
        
        Args:
            ten_class: Whether to use 10-class or 3-class classification
            
        Returns:
            Tuple of (indices, labels)
        """
        print(f"Loading {'10-class' if ten_class else '3-class'} classification data...")
        
        try:
            indices, labels = self.dataset.get_cell_type_classification_data(
                ten_class_version=ten_class
            )
            print(f"Found {len(indices)} labeled cells")
            
            if ten_class:
                class_names_dict = self.class_names_10
            else:
                class_names_dict = self.class_names_3
                
            # Print class distribution using discovered class names
            unique_labels, counts = np.unique(labels, return_counts=True)
            print("Class distribution:")
            for label, count in zip(unique_labels, counts):
                if class_names_dict and label in class_names_dict:
                    print(f"  {class_names_dict[label]}: {count}")
                else:
                    print(f"  class_{label}: {count}")
            
            return indices, labels
            
        except Exception as e:
            print(f"Error loading classification data: {e}")
            return np.array([]), np.array([])
    
    def _save_cell_image(self, cell_index: int, output_path: str, channel: str = 'DPC_Left') -> bool:
        """
        Save a cell image from BSCCM dataset.
        
        Args:
            cell_index: Global index of the cell
            output_path: Path to save the image
            channel: Imaging channel to use
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read image from BSCCM
            image = self.dataset.read_image(cell_index, channel=channel)
            
            # Convert to PIL Image and save
            if image.dtype == np.uint16:
                # Normalize 16-bit to 8-bit for standard formats
                image_normalized = (image / 4095.0 * 255).astype(np.uint8)
            elif image.dtype == np.float32:
                # Handle float32 images (like DPC)
                image_normalized = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            else:
                image_normalized = image.astype(np.uint8)
            
            pil_image = Image.fromarray(image_normalized)
            pil_image.save(output_path)
            
            return True
            
        except Exception as e:
            print(f"Error saving image for cell {cell_index}: {e}")
            return False
    
    def _get_channel_description(self, channel: str) -> str:
        """
        Convert channel name to human-readable description for questions.
        
        Args:
            channel: Technical channel name (e.g., 'DPC_Left', 'DF_50')
            
        Returns:
            Human-readable description
        """
        if not channel:
            return "microscopy"
            
        channel = channel.upper()
        
        if channel.startswith('DPC_'):
            direction = channel.split('_')[1].lower()
            return f"differential phase contrast ({direction} illumination)"
        elif channel.startswith('DF_'):
            # Extract angle/direction information
            parts = channel.split('_')
            if len(parts) > 1:
                angle = parts[1]
                direction = parts[2].lower() if len(parts) > 2 else ""
                if direction:
                    return f"dark field ({angle}° {direction} illumination)"
                else:
                    return f"dark field ({angle}° illumination)"
            return "dark field"
        elif channel == 'BRIGHTFIELD':
            return "brightfield"
        elif channel == 'LED119':
            return "LED119 illumination"
        else:
            # Fallback: convert underscores to spaces and make lowercase
            return channel.replace('_', ' ').lower()
    
    def _generate_choices(self, correct_class: int, ten_class: bool = False, channel: str = None) -> Tuple[str, str]:
        """
        Generate 6 multiple choice options for cell type classification.
        
        Args:
            correct_class: The correct class ID
            ten_class: Whether using 10-class or 3-class classification
            channel: The imaging channel used (for channel-specific questions)
            
        Returns:
            Tuple of (formatted_question, correct_answer_letter)
        """
        if ten_class:
            class_names_dict = self.class_names_10
        else:
            class_names_dict = self.class_names_3
        
        # Fixed choices that should always be included
        fixed_choices = [
            {"id": -1, "text": "unclassified_cell"},
            {"id": -2, "text": "debris_or_artifact"}
        ]
        
        choices = fixed_choices.copy()
        
        # Add the correct answer
        if class_names_dict and correct_class in class_names_dict:
            correct_text = class_names_dict[correct_class]
        else:
            correct_text = f"class_{correct_class}"
            
        choices.append({
            "id": correct_class,
            "text": correct_text,
            "correct": True
        })
        
        # Add random incorrect cell types
        available_classes = []
        if class_names_dict:
            for class_id, class_name in class_names_dict.items():
                if class_id != correct_class:
                    available_classes.append({"id": class_id, "text": class_name})
        
        # Fill remaining slots (6 total - 2 fixed - 1 correct = 3 remaining)
        remaining_slots = 3
        if len(available_classes) >= remaining_slots:
            random_classes = random.sample(available_classes, remaining_slots)
        else:
            random_classes = available_classes
            # Pad with duplicates if needed
            while len(random_classes) < remaining_slots:
                random_classes.append(random.choice(available_classes))
        
        for cell_class in random_classes:
            choices.append({
                "id": cell_class["id"],
                "text": cell_class["text"],
                "correct": False
            })
        
        # Mark correct choice and set others as incorrect
        for choice in choices:
            choice["correct"] = choice.get("id") == correct_class
        
        # Shuffle choices
        random.shuffle(choices)
        
        # Find correct answer letter
        choice_letters = ['A', 'B', 'C', 'D', 'E', 'F']
        correct_letter = None
        
        # Format question with choices and channel information
        question_base = random.choice(self.question_templates)
        
        # Convert channel name to readable description
        channel_type = self._get_channel_description(channel) if channel else "microscopy"
        question_base = question_base.format(channel_type=channel_type)
        
        question_parts = [question_base]
        
        for i, choice in enumerate(choices[:6]):  # Ensure exactly 6 choices
            question_parts.append(f"{choice_letters[i]}. {choice['text']}")
            if choice.get('correct', False):
                correct_letter = choice_letters[i]
        
        question_parts.append("\nPlease answer with a single letter (A, B, C, D, E or F).")
        formatted_question = "\n".join(question_parts)
        
        return formatted_question, correct_letter
    
    def _create_train_test_split(self, indices: np.ndarray, labels: np.ndarray, 
                                train_ratio: float = 0.9) -> Tuple[List, List]:
        """
        Create train/test split maintaining class distribution.
        
        Args:
            indices: Cell indices
            labels: Cell labels
            train_ratio: Ratio of data for training
            
        Returns:
            Tuple of (train_data, test_data) where each contains (index, label) pairs
        """
        train_data = []
        test_data = []
        
        # Split each class separately to maintain distribution
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            class_indices = indices[labels == label]
            n_train = int(len(class_indices) * train_ratio)
            
            # Shuffle indices for this class
            shuffled_indices = np.random.permutation(class_indices)
            
            # Split into train/test
            train_indices = shuffled_indices[:n_train]
            test_indices = shuffled_indices[n_train:]
            
            # Add to respective lists
            for idx in train_indices:
                train_data.append((idx, label))
            for idx in test_indices:
                test_data.append((idx, label))
        
        # Shuffle the final splits
        random.shuffle(train_data)
        random.shuffle(test_data)
        
        return train_data, test_data
    
    def _get_class_name(self, class_id: int, ten_class: bool) -> str:
        """
        Get the class name for a given class ID.
        
        Args:
            class_id: The class ID
            ten_class: Whether using 10-class classification
            
        Returns:
            The class name string
        """
        if ten_class:
            class_names_dict = self.class_names_10
        else:
            class_names_dict = self.class_names_3
            
        if class_names_dict and class_id in class_names_dict:
            return class_names_dict[class_id]
        else:
            return f"class_{class_id}"
    
    def convert_dataset(self, ten_class: bool = False, max_samples_per_split: int = None,
                       use_alternative_classification: str = None) -> None:
        """
        Convert the BSCCM dataset to VQA format using all available imaging channels.
        Each cell will generate one VQA entry per available channel.
        
        Args:
            ten_class: Whether to use 10-class or 3-class classification (ignored if using alternative)
            max_samples_per_split: Maximum number of samples per split (for testing)
            use_alternative_classification: If provided, use alternative classification:
                - "antibody": Classify by antibody staining (uses all 1000 cells)
                - "protein_expression": Classify by protein expression levels
                - "batch": Classify by batch information
                - None: Use original BSCCM classification labels (only 28 cells)
        """
        print("Starting BSCCM to VQA conversion...")
        
        # Download and setup dataset
        dataset_path = self.download_and_setup_dataset()
        
        # Discover available imaging channels
        available_channels = self._discover_available_channels()
        
        # Update the imaging_channels list with actual available channels
        if available_channels:
            self.imaging_channels = available_channels
            print(f"\nUpdated available channels list with {len(available_channels)} channels from dataset")
        
        # Discover actual class labels from the dataset
        self._discover_actual_classes()
        
        # Get classification data
        if use_alternative_classification:
            print(f"\nUsing alternative classification: {use_alternative_classification}")
            indices, labels, class_names_dict = self._create_alternative_classification(use_alternative_classification)
            
            # Set the appropriate class names
            if ten_class:
                self.class_names_10 = class_names_dict
            else:
                self.class_names_3 = class_names_dict
                
            print(f"Using ALL {len(indices)} cells with {use_alternative_classification} classification")
            
        else:
            print(f"\nUsing original BSCCM classification labels")
            indices, labels = self._get_classification_data(ten_class=ten_class)
        
        if len(indices) == 0:
            print("No classification data available. Cannot proceed.")
            return
        
        # Create train/test split
        train_data, test_data = self._create_train_test_split(indices, labels)
        
        if max_samples_per_split:
            train_data = train_data[:max_samples_per_split]
            test_data = test_data[:max_samples_per_split]
        
        print(f"Train samples: {len(train_data)}")
        print(f"Test samples: {len(test_data)}")
        print(f"Available channels: {len(available_channels)} total")
        print(f"Expected total VQA entries: {len(train_data) + len(test_data)} cells × {len(available_channels)} channels = {(len(train_data) + len(test_data)) * len(available_channels)}")
        
        # Process each split
        for split_name, split_data in [("train", train_data), ("test", test_data)]:
            print(f"\nProcessing {split_name} split...")
            
            vqa_entries = []
            images_dir = self.train_images_dir if split_name == "train" else self.test_images_dir
            
            total_entries_expected = len(split_data) * len(available_channels)
            print(f"Expected total VQA entries: {len(split_data)} cells × {len(available_channels)} channels = {total_entries_expected}")
            
            entry_count = 0
            for i, (cell_index, cell_label) in enumerate(split_data):
                try:
                    # Convert to int in case indices are float
                    cell_index = int(cell_index)
                    cell_label = int(cell_label)
                    
                    # Process each channel for this cell
                    for channel_idx, channel in enumerate(available_channels):
                        try:
                            # Generate image filename with channel info
                            image_filename = f"bsccm_cell_{cell_index:06d}_{channel}.jpg"
                            image_path = images_dir / image_filename
                            
                            # Save cell image for this specific channel
                            success = self._save_cell_image(
                                cell_index, str(image_path), channel=channel
                            )
                            
                            if not success:
                                print(f"Skipping cell {cell_index}, channel {channel} due to image save failure")
                                continue
                            
                            # Generate VQA question with channel-specific context
                            formatted_question, correct_letter = self._generate_choices(
                                cell_label, ten_class=ten_class, channel=channel
                            )
                            
                            # Get cell metadata if available
                            cell_metadata = {}
                            try:
                                if hasattr(self.dataset, 'index_dataframe'):
                                    if cell_index in self.dataset.index_dataframe.index:
                                        row = self.dataset.index_dataframe.loc[cell_index]
                                        cell_metadata = {
                                            'batch': row.get('batch', None),
                                            'antibodies': row.get('antibodies', None),
                                            'position_x': row.get('position_in_fov_x_pix', None),
                                            'position_y': row.get('position_in_fov_y_pix', None),
                                        }
                            except Exception as e:
                                print(f"Could not load metadata for cell {cell_index}: {e}")
                            
                            # Create VQA entry for this cell+channel combination
                            vqa_entry = {
                                "id": f"bsccm_cell_{cell_index:06d}_{channel}",
                                "image": str(image_path),
                                "question": formatted_question,
                                "answer": correct_letter,
                                "metadata": {
                                    "cell_index": int(cell_index),
                                    "cell_type": int(cell_label),
                                    "cell_type_name": self._get_class_name(cell_label, ten_class),
                                    "split": split_name,
                                    "imaging_channel": channel,
                                    "channel_index": channel_idx,
                                    "classification_type": "10-class" if ten_class else "3-class",
                                    **{k: (int(v) if isinstance(v, (np.integer, np.floating)) else v) 
                                       for k, v in cell_metadata.items()}
                                }
                            }
                            
                            vqa_entries.append(vqa_entry)
                            entry_count += 1
                            
                        except Exception as e:
                            print(f"Error processing cell {cell_index}, channel {channel}: {e}")
                            continue
                    
                    if (i + 1) % 100 == 0:
                        print(f"  Processed {i + 1}/{len(split_data)} cells, generated {entry_count} VQA entries so far")
                        
                except Exception as e:
                    print(f"Error processing cell {cell_index}: {e}")
                    continue
            
            # Save VQA dataset for this split
            output_file = self.output_dir / f"{split_name}_vqa_dataset.json"
            with open(output_file, 'w') as f:
                json.dump(vqa_entries, f, indent=2)
            
            print(f"{split_name.upper()} conversion complete!")
            print(f"Total VQA entries: {len(vqa_entries)} (from {len(split_data)} cells × {len(available_channels)} channels)")
            print(f"Dataset saved to: {output_file}")
            
            # Generate summary for this split
            self._generate_summary(vqa_entries, split_name)
        
        print(f"\n{'='*60}")
        print("BSCCM to VQA conversion completed successfully!")
        print(f"{'='*60}")
    
    def _generate_summary(self, vqa_entries: List[Dict], split: str) -> None:
        """Generate and save dataset summary statistics for a specific split."""
        print(f"\n--- {split.upper()} Dataset Summary ---")
        
        # Count by cell type
        cell_type_counts = {}
        answer_letter_counts = {}
        
        for entry in vqa_entries:
            # Get cell type from metadata
            cell_type_name = entry['metadata']['cell_type_name']
            cell_type_counts[cell_type_name] = cell_type_counts.get(cell_type_name, 0) + 1
            
            # Count answer letter distribution
            answer_letter = entry['answer']
            answer_letter_counts[answer_letter] = answer_letter_counts.get(answer_letter, 0) + 1
        
        print("Cell type distribution:")
        for cell_type, count in sorted(cell_type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cell_type}: {count}")
        
        print(f"\nAnswer letter distribution:")
        for letter in sorted(answer_letter_counts.keys()):
            print(f"  {letter}: {answer_letter_counts[letter]}")
        
        # Save summary to file
        summary = {
            "split": split,
            "total_entries": len(vqa_entries),
            "cell_type_distribution": cell_type_counts,
            "answer_letter_distribution": answer_letter_counts,
            "sample_question": vqa_entries[0] if vqa_entries else None
        }
        
        summary_file = self.output_dir / f"{split}_dataset_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary saved to: {summary_file}")


def main():
    """Main function to run the BSCCM VQA conversion."""
    # Check and fix dependencies first
    print("Checking dependency compatibility...")
    if not check_and_fix_dependencies():
        print("Please restart the script after fixing dependencies.")
        return
    
    # Configuration
    DATASET_ROOT = "/pasteur2/u/lihe50hz/BSCCM"
    OUTPUT_DIR = "/pasteur2/u/lihe50hz/BSCCM_vqa"
    
    # Initialize converter
    converter = BSCCMVQAConverter(DATASET_ROOT, OUTPUT_DIR, use_tiny=False)
    
    # Configuration for conversion
    USE_TEN_CLASS = True  # Set to True for 10-class classification
    MAX_SAMPLES_PER_SPLIT = 50000  # Limit for testing, set to None for full dataset
    # Note: Will use ALL available channels (up to 23) - each cell generates multiple VQA entries
    
    # Alternative classification options (to use all 1000 cells instead of just 28)
    # Set to None to use official BSCCM cell type classifications (Lymphocytes, Monocytes, Granulocytes)
    USE_ALTERNATIVE_CLASSIFICATION = None  # Options: "antibody", "protein_expression", "batch", or None
    
    print("\n" + "="*60)
    print("Starting BSCCM VQA conversion...")
    print("="*60)
    
    if USE_ALTERNATIVE_CLASSIFICATION:
        print(f"🔬 Using alternative classification: {USE_ALTERNATIVE_CLASSIFICATION}")
        print("   This will use ALL 1000 cells in the dataset!")
    else:
        print(f"🩸 Using official BSCCM cell type classifications")
        if USE_TEN_CLASS:
            print("   10-class: Lymphocytes, Monocytes, Granulocytes + subtypes")
        else:
            print("   3-class: Lymphocytes, Granulocytes, Monocytes")
        print("   ⚠️  Note: Full BSCCM should have more labeled cells than BSCCM-tiny")
    
    print(f"Max samples per split: {MAX_SAMPLES_PER_SPLIT}")
    print("Imaging: Will use ALL available channels (up to 23 total)")
    print("Note: Will download full BSCCM dataset (~10GB+) if not available")
    print("\n🔍 The script will test all 23 known BSCCM imaging channels:")
    print("   This will show you which channels are accessible in your dataset")
    print("   Each cell will generate one VQA entry per available channel")
    print("\nIf you encounter zarr compatibility issues, run:")
    print("  pip install 'zarr<3.0.0' --force-reinstall")
    print("")
    
    converter.convert_dataset(
        ten_class=USE_TEN_CLASS,
        max_samples_per_split=MAX_SAMPLES_PER_SPLIT,
        use_alternative_classification=USE_ALTERNATIVE_CLASSIFICATION
    )


if __name__ == "__main__":
    main()
