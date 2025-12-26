import json
import re
from datasets import load_from_disk, DatasetDict, concatenate_datasets, load_dataset
from typing import Dict, Any, Callable, Optional, List
import logging
from tqdm import tqdm
import random
import gc
import psutil
import os

# Memory management constants
MAX_MEMORY_PERCENTAGE = 70  # Maximum percentage of RAM to use
BATCH_SIZE = 100  # Reduced batch size for memory efficiency
STREAMING_BATCH_SIZE = 50  # Even smaller batch size for streaming operations

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def normalize_arabic_text(text: str) -> str:
    """
    Normalize Arabic text by removing diacritics and normalizing certain characters.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[ًٌٍَُِّْٰٖٕٓٔ]', '', text)  # Remove diacritics
    text = re.sub(r'[إأآا]', 'ا', text)  # Normalize alifs
    text = re.sub(r'ى', 'ي', text)  # Normalize ya
    text = re.sub(r'ة', 'ه', text)  # Normalize ta marbuta
    return text.strip()

def truncate_text(text: str, max_length: int = 500) -> str:
    """
    Truncate text to a maximum length while keeping whole words.
    """
    if not isinstance(text, str):
        return ""
    if len(text) <= max_length:
        return text
    return ' '.join(text[:max_length+1].split()[:-1])

def process_bactrian(batch: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Process Bactrian dataset entries in batches.
    """
    instructions = [normalize_arabic_text(inst) for inst in batch.get("instruction", [])]
    inputs = [normalize_arabic_text(inp) for inp in batch.get("input", [])]
    outputs = [normalize_arabic_text(out) for out in batch.get("output", [])]
    return {
        "instruction": instructions,
        "input": inputs,
        "output": outputs
    }

def process_oasst(batch: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Process OpenAssistant dataset entries in batches.
    """
    instructions = [normalize_arabic_text(q) for q in batch.get("question", [])]
    inputs = [""] * len(instructions)
    outputs = [normalize_arabic_text(a) for a in batch.get("answer", [])]
    return {
        "instruction": instructions,
        "input": inputs,
        "output": outputs
    }

def process_wikipedia(batch: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Process Wikipedia dataset entries in batches.
    """
    titles = [normalize_arabic_text(t) for t in batch.get("title", [])]
    texts = [normalize_arabic_text(t) for t in batch.get("text", [])]

    instructions = []
    inputs = []
    outputs = []
    for title, text in zip(titles, texts):
        questions = [
            f"ما هو {title}؟",
            f"اشرح بالتفصيل عن {title}.",
            f"ما هي أهم المعلومات عن {title}؟"
        ]
        instructions.append(random.choice(questions))
        inputs.append("")
        outputs.append(truncate_text(text, 1000))

    return {
        "instruction": instructions,
        "input": inputs,
        "output": outputs
    }

def inspect_dataset(name: str, dataset: Any) -> None:
    """
    Inspect and log details about a dataset.
    """
    logger.info(f"\nInspecting {name} dataset:")
    logger.info(f"Number of examples: {len(dataset)}")
    logger.info(f"Column names: {dataset.column_names}")
    logger.info("First example:")
    logger.info(json.dumps(dataset[0], indent=2, ensure_ascii=False))

def get_memory_usage() -> float:
    """Get current memory usage as percentage of total RAM."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    total_memory = psutil.virtual_memory().total
    return (memory_info.rss / total_memory) * 100

def ensure_memory_available(required_percentage: float = MAX_MEMORY_PERCENTAGE) -> bool:
    """Ensure we have enough memory available."""
    current_usage = get_memory_usage()
    if current_usage > required_percentage:
        logger.warning(f"Memory usage at {current_usage:.1f}% exceeds limit of {required_percentage}%")
        gc.collect()
        current_usage = get_memory_usage()
        if current_usage > required_percentage:
            logger.error(f"Still high memory usage: {current_usage:.1f}%")
            return False
    return True

def load_and_process_dataset(name: str, path: str, process_func: Callable) -> Optional[str]:
    """
    Load, process, and save a dataset to disk with memory management.
    """
    try:
        # Check memory before loading
        if not ensure_memory_available():
            return None

        # Load the dataset from disk
        dataset = load_from_disk(path)

        # If the dataset has multiple splits, use the 'train' split
        if isinstance(dataset, DatasetDict):
            dataset = dataset['train']

        logger.info(f"Loaded {name} dataset. Size: {len(dataset)}")
        logger.info(f"Memory usage after loading: {get_memory_usage():.1f}%")

        # Inspect the dataset
        inspect_dataset(name, dataset)

        # Process the dataset in smaller batches to save memory
        processed = dataset.map(process_func, batched=True, batch_size=BATCH_SIZE, remove_columns=dataset.column_names)
        logger.info(f"Processed {len(processed)} entries from {path}")
        logger.info(f"Memory usage after processing: {get_memory_usage():.1f}%")

        # Save processed dataset to disk
        save_path = f"./processed_{name}_dataset"
        processed.save_to_disk(save_path)
        logger.info(f"Saved processed {name} dataset to {save_path}")

        # Free memory
        del dataset, processed
        gc.collect()
        logger.info(f"Memory usage after cleanup: {get_memory_usage():.1f}%")

        return save_path
    except Exception as e:
        logger.error(f"Error processing dataset {name}: {str(e)}")
        gc.collect()
        return None

def stream_and_combine_datasets(processed_paths: List[str], output_path: str = "./arabic_instruction_dataset"):
    """
    Stream and combine datasets with memory management.
    """
    logger.info("Starting to stream and combine datasets...")
    
    # Create a temporary directory for intermediate files
    temp_dir = "./temp_combined_dataset"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Process datasets one by one to avoid loading all at once
    combined_data = []
    total_entries = 0
    
    for i, path in enumerate(processed_paths):
        logger.info(f"Processing dataset {i+1}/{len(processed_paths)}: {path}")
        
        # Check memory before loading
        if not ensure_memory_available():
            logger.error("Cannot proceed due to memory constraints")
            return None
        
        # Load dataset
        dataset = load_from_disk(path)
        logger.info(f"Memory usage after loading dataset {i+1}: {get_memory_usage():.1f}%")
        
        # Filter out empty entries
        filtered_dataset = dataset.filter(lambda x: x["instruction"] and x["output"])
        logger.info(f"Dataset {i+1} size after filtering: {len(filtered_dataset)}")
        
        # Process in smaller batches
        batch_size = STREAMING_BATCH_SIZE
        for start_idx in range(0, len(filtered_dataset), batch_size):
            end_idx = min(start_idx + batch_size, len(filtered_dataset))
            batch = filtered_dataset[start_idx:end_idx]
            
            # Convert batch to list of dictionaries
            batch_list = []
            # Handle the batch as a whole, not by index
            if isinstance(batch, dict):
                # If batch is a dictionary, convert to list format
                for key in batch:
                    if key in ["instruction", "input", "output"]:
                        values = batch[key]
                        if isinstance(values, list):
                            for i, value in enumerate(values):
                                # Ensure we have enough entries in batch_list
                                while len(batch_list) <= i:
                                    batch_list.append({"instruction": "", "input": "", "output": ""})
                                batch_list[i][key] = value
            else:
                # If batch is already a list or other format, iterate through it
                for example in batch:
                    if isinstance(example, dict):
                        batch_list.append({
                            "instruction": example.get("instruction", ""),
                            "input": example.get("input", ""),
                            "output": example.get("output", "")
                        })
            
            # Add to combined data
            combined_data.extend(batch_list)
            total_entries += len(batch_list)
            
            # Save intermediate results periodically
            if len(combined_data) >= 1000:  # Save every 1000 entries
                temp_file = os.path.join(temp_dir, f"batch_{total_entries}.json")
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(combined_data, f, ensure_ascii=False, indent=2)
                combined_data = []  # Clear the list
                gc.collect()
                logger.info(f"Saved intermediate batch with {total_entries} total entries")
                logger.info(f"Memory usage: {get_memory_usage():.1f}%")
        
        # Clean up
        del dataset, filtered_dataset
        gc.collect()
    
    # Save remaining data
    if combined_data:
        temp_file = os.path.join(temp_dir, f"batch_{total_entries}.json")
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Total entries processed: {total_entries}")
    
    # Create final dataset from saved files
    final_data = []
    for filename in sorted(os.listdir(temp_dir)):
        if filename.endswith('.json'):
            filepath = os.path.join(temp_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
                # Ensure batch_data is a list of dictionaries
                if isinstance(batch_data, list):
                    final_data.extend(batch_data)
                else:
                    logger.warning(f"Unexpected data format in {filename}")
    
    # Convert to Dataset and save
    if final_data:
        from datasets import Dataset
        # Ensure all entries are dictionaries with the expected keys
        cleaned_data = []
        for item in final_data:
            if isinstance(item, dict):
                cleaned_data.append({
                    "instruction": item.get("instruction", ""),
                    "input": item.get("input", ""),
                    "output": item.get("output", "")
                })
        
        if cleaned_data:
            final_dataset = Dataset.from_list(cleaned_data)
            final_dataset.save_to_disk(output_path)
            logger.info(f"Final combined dataset saved to {output_path}")
            
            # Clean up temp directory
            import shutil
            shutil.rmtree(temp_dir)
            
            return output_path
        else:
            logger.error("No valid data to save after cleaning")
            return None
    else:
        logger.error("No data to save")
        return None

def main():
    # Define datasets
    datasets = {
        "bactrian": ("./bactrian_dataset", process_bactrian),
        "oasst": ("./oasst_dataset", process_oasst),
        "wikipedia": ("./wikipedia_dataset", process_wikipedia)
    }

    logger.info(f"Starting dataset processing with memory limit: {MAX_MEMORY_PERCENTAGE}%")
    logger.info(f"Initial memory usage: {get_memory_usage():.1f}%")

    processed_paths = []
    for name, (path, process_func) in datasets.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {name} dataset...")
        logger.info(f"{'='*50}")
        
        save_path = load_and_process_dataset(name, path, process_func)
        if save_path:
            processed_paths.append(save_path)
        
        # Force garbage collection between datasets
        gc.collect()
        logger.info(f"Memory usage after processing {name}: {get_memory_usage():.1f}%")

    if processed_paths:
        logger.info(f"\n{'='*50}")
        logger.info("Combining processed datasets...")
        logger.info(f"{'='*50}")
        
        # Use streaming approach to combine datasets
        output_path = stream_and_combine_datasets(processed_paths)
        
        if output_path:
            # Load a small sample for verification
            logger.info("Loading sample for verification...")
            sample_dataset = load_from_disk(output_path)
            logger.info(f"Final dataset size: {len(sample_dataset)}")
            
            # Print sample entries
            print("Sample entries:")
            for i in range(min(3, len(sample_dataset))):
                sample = sample_dataset[i]
                print(json.dumps(sample, ensure_ascii=False, indent=2))
            
            logger.info(f"Final memory usage: {get_memory_usage():.1f}%")
        else:
            logger.error("Failed to combine datasets")
    else:
        logger.error("No datasets were successfully processed")

if __name__ == "__main__":
    main()