import os
import torch
import argparse
import yaml
from typing import Union, Tuple, Dict, Any
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    LlavaForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor
)
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config_from_yaml(yaml_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        yaml_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML config file not found: {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {yaml_path}")
    return config

def is_huggingface_path(path: str) -> bool:
    """
    Check if the path is a HuggingFace model identifier or a local path.
    
    Args:
        path: Model path or identifier
        
    Returns:
        True if it's a HuggingFace path, False if it's a local path
    """
    # HuggingFace paths typically don't contain file separators and contain '/'
    # Local paths contain file separators or are relative/absolute paths
    return not os.path.exists(path) and '/' in path and not path.startswith('/')

def detect_model_type(checkpoint_path: str) -> str:
    """
    Detect the model type based on the config.json file or model identifier.
    
    Args:
        checkpoint_path: Path to the model checkpoint directory or HuggingFace model identifier
        
    Returns:
        Model type: 'qwen2_5_vl' or 'llava'
    """
    if is_huggingface_path(checkpoint_path):
        # For HuggingFace paths, we can infer from the model name
        if 'qwen2' in checkpoint_path.lower() and 'vl' in checkpoint_path.lower():
            return 'qwen2_5_vl'
        elif 'llava' in checkpoint_path.lower():
            return 'llava'
        else:
            # Try to load config from HuggingFace Hub
            try:
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
                
                model_type = getattr(config, 'model_type', '')
                architectures = getattr(config, 'architectures', [])
                
                if 'qwen2_5_vl' in model_type.lower() or any('Qwen2_5_VL' in str(arch) for arch in architectures):
                    return 'qwen2_5_vl'
                elif 'llava' in model_type.lower() or any('Llava' in str(arch) for arch in architectures):
                    return 'llava'
                else:
                    raise ValueError(f"Unsupported model type from HuggingFace: {model_type}. Architectures: {architectures}")
            except Exception as e:
                raise ValueError(f"Could not determine model type from HuggingFace path {checkpoint_path}: {e}")
    else:
        # Local path handling
        config_path = os.path.join(checkpoint_path, 'config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {checkpoint_path}")
        
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        model_type = config.get('model_type', '')
        architectures = config.get('architectures', [])
        
        if 'qwen2_5_vl' in model_type.lower() or any('Qwen2_5_VL' in arch for arch in architectures):
            return 'qwen2_5_vl'
        elif 'llava' in model_type.lower() or any('Llava' in arch for arch in architectures):
            return 'llava'
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Architectures: {architectures}")

def load_model(checkpoint_path: str, model_type: str, device: str = 'cpu') -> Union[Qwen2_5_VLForConditionalGeneration, LlavaForConditionalGeneration]:
    """
    Load a model from checkpoint or HuggingFace Hub based on the model type.
    
    Args:
        checkpoint_path: Path to the model checkpoint directory or HuggingFace model identifier
        model_type: Type of model ('qwen2_5_vl' or 'llava')
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    if is_huggingface_path(checkpoint_path):
        logger.info(f"Loading {model_type} model from HuggingFace Hub: {checkpoint_path}")
    else:
        logger.info(f"Loading {model_type} model from local path: {checkpoint_path}")
    
    if model_type == 'qwen2_5_vl':
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.float32,  # Use float32 for merging to avoid precision issues
            device_map=None,  # Load to CPU first for merging
            trust_remote_code=True
        )
    elif model_type == 'llava':
        model = LlavaForConditionalGeneration.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.float32,  # Use float32 for merging to avoid precision issues
            device_map=None  # Load to CPU first for merging
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model = model.to(device)
    logger.info(f"Successfully loaded {model_type} model")
    return model

def merge_model_weights(
    model1: Union[Qwen2_5_VLForConditionalGeneration, LlavaForConditionalGeneration],
    model2: Union[Qwen2_5_VLForConditionalGeneration, LlavaForConditionalGeneration],
    lambda_weight: float
) -> Union[Qwen2_5_VLForConditionalGeneration, LlavaForConditionalGeneration]:
    """
    Perform linear interpolation on model weights.
    
    merged_weight = lambda_weight * weight1 + (1 - lambda_weight) * weight2
    
    Args:
        model1: First model
        model2: Second model  
        lambda_weight: Interpolation weight (0.0 to 1.0)
        
    Returns:
        Model with merged weights
    """
    if not (0.0 <= lambda_weight <= 1.0):
        raise ValueError(f"lambda_weight must be between 0.0 and 1.0, got {lambda_weight}")
    
    logger.info(f"Merging models with lambda = {lambda_weight}")
    
    # Get state dictionaries
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    
    # Check if models have the same architecture
    if set(state_dict1.keys()) != set(state_dict2.keys()):
        missing_in_2 = set(state_dict1.keys()) - set(state_dict2.keys())
        missing_in_1 = set(state_dict2.keys()) - set(state_dict1.keys())
        
        error_msg = "Models have different architectures:\n"
        if missing_in_2:
            error_msg += f"  Parameters in model1 but not in model2: {list(missing_in_2)[:5]}...\n"
        if missing_in_1:
            error_msg += f"  Parameters in model2 but not in model1: {list(missing_in_1)[:5]}...\n"
        
        raise ValueError(error_msg)
    
    # Perform linear interpolation
    merged_state_dict = {}
    total_params = len(state_dict1)
    
    for i, (key, param1) in enumerate(state_dict1.items()):
        param2 = state_dict2[key]
        
        if param1.shape != param2.shape:
            raise ValueError(f"Parameter {key} has different shapes: {param1.shape} vs {param2.shape}")
        
        # Linear interpolation: λ * w1 + (1-λ) * w2
        merged_param = lambda_weight * param1 + (1 - lambda_weight) * param2
        merged_state_dict[key] = merged_param
        
        if (i + 1) % 100 == 0:
            logger.info(f"Merged {i + 1}/{total_params} parameters")
    
    # Load merged weights into model1 (we'll return this as the merged model)
    model1.load_state_dict(merged_state_dict)
    logger.info("Successfully merged all model weights")
    
    return model1

def linear_weight_merging(
    checkpoint1_path: str,
    checkpoint2_path: str,
    lambda_weight: float,
    output_path: str,
    device: str = 'cpu'
) -> Union[Qwen2_5_VLForConditionalGeneration, LlavaForConditionalGeneration]:
    """
    Main function to perform linear weight merging of two model checkpoints and save the result.
    
    Args:
        checkpoint1_path: Path to first model checkpoint
        checkpoint2_path: Path to second model checkpoint
        lambda_weight: Interpolation weight (0.0 to 1.0)
        output_path: Path to save the merged model
        device: Device to perform merging on
        
    Returns:
        Merged model
    """
    logger.info("Starting linear weight merging...")
    logger.info(f"Model 1: {checkpoint1_path}")
    logger.info(f"Model 2: {checkpoint2_path}")
    logger.info(f"Lambda: {lambda_weight}")
    logger.info(f"Output: {output_path}")
    
    # Validate inputs
    if not is_huggingface_path(checkpoint1_path) and not os.path.exists(checkpoint1_path):
        raise FileNotFoundError(f"Local checkpoint 1 not found: {checkpoint1_path}")
    if not is_huggingface_path(checkpoint2_path) and not os.path.exists(checkpoint2_path):
        raise FileNotFoundError(f"Local checkpoint 2 not found: {checkpoint2_path}")
    
    # Detect model types
    model_type1 = detect_model_type(checkpoint1_path)
    model_type2 = detect_model_type(checkpoint2_path)
    
    if model_type1 != model_type2:
        raise ValueError(f"Cannot merge different model types: {model_type1} vs {model_type2}")
    
    logger.info(f"Detected model type: {model_type1}")
    
    # Load models
    model1 = load_model(checkpoint1_path, model_type1, device)
    model2 = load_model(checkpoint2_path, model_type2, device)
    
    # Merge weights
    merged_model = merge_model_weights(model1, model2, lambda_weight)
    
    # Save merged model
    logger.info(f"Saving merged model to {output_path}")
    os.makedirs(output_path, exist_ok=True)
    
    # Save the merged model
    merged_model.save_pretrained(output_path)
    
    # Copy tokenizer and processor from the first checkpoint
    try:
        if is_huggingface_path(checkpoint1_path):
            logger.info("Loading tokenizer from HuggingFace Hub")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint1_path, trust_remote_code=True)
        tokenizer.save_pretrained(output_path)
        logger.info("Saved tokenizer")
    except Exception as e:
        logger.warning(f"Failed to copy tokenizer: {e}")
    
    try:
        if is_huggingface_path(checkpoint1_path):
            logger.info("Loading processor from HuggingFace Hub")
        processor = AutoProcessor.from_pretrained(checkpoint1_path, trust_remote_code=True)
        processor.save_pretrained(output_path)
        logger.info("Saved processor")
    except Exception as e:
        logger.warning(f"Failed to copy processor: {e}")
    
    logger.info(f"Successfully saved merged model to {output_path}")
    logger.info("Weight merging completed successfully!")
    return merged_model

def main():
    """Command line interface for weight merging."""
    parser = argparse.ArgumentParser(description="Linear weight merging for Qwen2.5-VL and LLaVA models")
    
    # Add YAML config option
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file (if provided, other arguments are optional)"
    )
    
    parser.add_argument(
        "--checkpoint1",
        type=str,
        help="Path to the first model checkpoint"
    )
    
    parser.add_argument(
        "--checkpoint2", 
        type=str,
        help="Path to the second model checkpoint"
    )
    
    parser.add_argument(
        "--lambda_weight",
        type=float,
        help="Interpolation weight (0.0 to 1.0). Formula: lambda * model1 + (1-lambda) * model2"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save the merged model"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to perform merging on"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        # Load from YAML file
        config = load_config_from_yaml(args.config)
        logger.info("Using YAML configuration")
    else:
        logger.info("Using command line arguments")
    
    # Override YAML config with command line arguments (if provided)
    final_config = {
        'checkpoint1': config.get('checkpoint1', args.checkpoint1),
        'checkpoint2': config.get('checkpoint2', args.checkpoint2),
        'lambda_weight': config.get('lambda_weight', args.lambda_weight),
        'output_path': config.get('output_path', args.output_path),
        'device': config.get('device', args.device)
    }
    
    # Validate required parameters
    required_params = ['checkpoint1', 'checkpoint2', 'lambda_weight', 'output_path']
    missing_params = [param for param in required_params if final_config[param] is None]
    
    if missing_params:
        parser.error(f"Missing required parameters: {missing_params}. "
                    f"Provide them via command line arguments or YAML config file.")
    
    try:
        merged_model = linear_weight_merging(
            checkpoint1_path=final_config['checkpoint1'],
            checkpoint2_path=final_config['checkpoint2'],
            lambda_weight=final_config['lambda_weight'],
            output_path=final_config['output_path'],
            device=final_config['device']
        )
        
        print(f"\n✅ Weight merging completed successfully!")
        print(f"📁 Merged model saved to: {final_config['output_path']}")
            
    except Exception as e:
        logger.error(f"Weight merging failed: {e}")
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
