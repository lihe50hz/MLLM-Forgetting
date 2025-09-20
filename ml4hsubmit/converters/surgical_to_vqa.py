#!/usr/bin/env python3
"""
Convert the PitVis 2023 surgical dataset into a VQA (Visual Question Answering) format.
Extracts frames from videos and creates multiple choice questions about surgical instruments.
"""

import os
import csv
import json
import random
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Set
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
import shutil


def extract_video_frames_batch(video_path: str, frame_requests: List[Dict]) -> List[Dict]:
    """
    Global function to extract multiple frames from a video for multiprocessing.
    
    Args:
        video_path: Path to the video file
        frame_requests: List of frame request dictionaries
        
    Returns:
        List of successful frame extractions
    """
    if not frame_requests:
        return []
    
    successful_frames = []
    video_id = frame_requests[0].get('video_id', 'unknown')
    
    print(f"Batch extracting {len(frame_requests)} frames from video {video_id}")
    
    # Filter out frames that already exist
    frames_to_extract = []
    for frame_req in frame_requests:
        if os.path.exists(frame_req['output_path']):
            successful_frames.append(frame_req)
        else:
            frames_to_extract.append(frame_req)
    
    if not frames_to_extract:
        print(f"All {len(frame_requests)} frames already exist for video {video_id}")
        return successful_frames
    
    print(f"Extracting {len(frames_to_extract)} new frames for video {video_id}")
    
    # Use optimized sequential extraction with better seeking
    for frame_req in frames_to_extract:
        try:
            time_seconds = frame_req['timestamp'] / 24.0
            
            # Use optimized ffmpeg command with input seeking
            cmd = [
                'ffmpeg',
                '-ss', str(time_seconds),  # Seek before input for faster seeking
                '-i', video_path,
                '-vframes', '1',
                '-q:v', '2',  # High quality
                '-y',
                frame_req['output_path']
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0 and os.path.exists(frame_req['output_path']):
                successful_frames.append(frame_req)
            else:
                print(f"Failed to extract frame {frame_req['timestamp']} from video {video_id}: {result.stderr}")
                
        except Exception as e:
            print(f"Error extracting frame {frame_req['timestamp']} from video {video_id}: {e}")
    
    print(f"Completed video {video_id}: extracted {len(successful_frames)} / {len(frame_requests)} frames")
    return successful_frames


class SurgicalVQAConverter:
    def __init__(self, dataset_root: str, output_dir: str):
        """
        Initialize the VQA converter.
        
        Args:
            dataset_root: Path to the surgical dataset root directory
            output_dir: Directory to save extracted frames and VQA data
        """
        self.dataset_root = Path(dataset_root)
        self.output_dir = Path(output_dir)
        self.videos_dir = self.dataset_root / "videos"
        self.annotations_dir = self.dataset_root / "annotations"
        self.misc_dir = self.dataset_root / "misc"
        
        # Define dataset splits based on the official recommendation
        self.train_videos = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23]
        self.test_videos = [1, 12, 21, 24, 25]  # Original validation set used as test
        
        # Create output directories for both splits
        self.train_images_dir = self.output_dir / "train" / "images"
        self.test_images_dir = self.output_dir / "test" / "images"
        self.train_images_dir.mkdir(parents=True, exist_ok=True)
        self.test_images_dir.mkdir(parents=True, exist_ok=True)
        
        # Load instrument mappings
        self.instrument_map = self._load_instrument_map()
        
        # Question template
        self.question_template = "<image>\nWhat is the major surgical instrument being used in this frame?"
        
        # Fixed choices that must be included
        self.fixed_choices = [
            {"id": -2, "text": "no_secondary_instrument"},
            {"id": -1, "text": "out_of_patient"},
            {"id": 0, "text": "no_visible_instrument/occluded_image_inside_patient"}
        ]
        
        # Available instrument choices (excluding fixed ones)
        self.available_instruments = [
            item for item in self.instrument_map 
            if item["id"] not in [-2, -1, 0]
        ]
        
        print(f"Loaded {len(self.instrument_map)} instruments")
        print(f"Available for random selection: {len(self.available_instruments)}")
        print(f"Train videos: {self.train_videos}")
        print(f"Test videos: {self.test_videos}")
    
    def _get_images_dir_for_split(self, split: str) -> Path:
        """Get the appropriate images directory for the given split."""
        if split == "train":
            return self.train_images_dir
        elif split == "test":
            return self.test_images_dir
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'test'")
    
    def _get_split_for_video(self, video_id: int) -> str:
        """Determine which split a video belongs to."""
        if video_id in self.train_videos:
            return "train"
        elif video_id in self.test_videos:
            return "test"
        else:
            raise ValueError(f"Video {video_id} not found in any split")
    
    def _load_instrument_map(self) -> List[Dict]:
        """Load instrument mappings from CSV file."""
        instrument_map = []
        map_file = self.misc_dir / "map_instruments.csv"
        
        with open(map_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Handle the case where both 0 values map to different descriptions
                if int(row['int_instrument']) == 0:
                    # Only add once, combining both descriptions
                    if not any(item['id'] == 0 for item in instrument_map):
                        instrument_map.append({
                            "id": 0,
                            "text": "no_visible_instrument/occluded_image_inside_patient"
                        })
                else:
                    instrument_map.append({
                        "id": int(row['int_instrument']),
                        "text": row['str_instrument']
                    })
        
        return instrument_map
    

    
    def _generate_choices(self, correct_instrument_id: int) -> Tuple[str, str]:
        """
        Generate 6 multiple choice options and format as A-F question.
        
        Args:
            correct_instrument_id: The correct instrument ID
            
        Returns:
            Tuple of (formatted_question, correct_answer_letter)
        """
        choices = []
        
        # Add fixed choices
        for fixed_choice in self.fixed_choices:
            choices.append({
                "id": fixed_choice["id"],
                "text": fixed_choice["text"],
                "correct": fixed_choice["id"] == correct_instrument_id
            })
        
        # Add correct answer if not already in fixed choices
        if correct_instrument_id not in [-2, -1, 0]:
            correct_instrument = next(
                (item for item in self.instrument_map if item["id"] == correct_instrument_id),
                None
            )
            if correct_instrument:
                choices.append({
                    "id": correct_instrument["id"],
                    "text": correct_instrument["text"],
                    "correct": True
                })
        
        # Fill remaining slots with random instruments
        remaining_slots = 6 - len(choices)
        
        # Get available instruments (excluding those already selected)
        selected_ids = {choice["id"] for choice in choices}
        available_for_random = [
            item for item in self.available_instruments
            if item["id"] not in selected_ids
        ]
        
        # Randomly select remaining instruments
        if len(available_for_random) >= remaining_slots:
            random_instruments = random.sample(available_for_random, remaining_slots)
            for instrument in random_instruments:
                choices.append({
                    "id": instrument["id"],
                    "text": instrument["text"],
                    "correct": False
                })
        else:
            # If not enough instruments, add what we have
            for instrument in available_for_random:
                choices.append({
                    "id": instrument["id"],
                    "text": instrument["text"],
                    "correct": False
                })
        
        # Pad with duplicates if needed to reach exactly 6 choices
        while len(choices) < 6:
            choices.append(choices[0])  # Duplicate first choice if needed
        
        # Shuffle choices to randomize order
        random.shuffle(choices)
        
        # Find the correct answer letter
        correct_letter = None
        choice_letters = ['A', 'B', 'C', 'D', 'E', 'F']
        
        # Format question with A-F choices
        question_parts = [self.question_template]
        for i, choice in enumerate(choices[:6]):  # Ensure exactly 6 choices
            question_parts.append(f"{choice_letters[i]}. {choice['text']}")
            if choice['correct']:
                correct_letter = choice_letters[i]
        
        question_parts.append("\nPlease answer with a single letter (A, B, C, D, E or F).")
        formatted_question = "\n".join(question_parts)
        
        return formatted_question, correct_letter
    
    def _collect_frame_info(self, video_id: int, sample_rate: int = 30) -> List[Dict]:
        """
        Collect frame information for a video without extracting frames.
        
        Args:
            video_id: Video number (1-25)
            sample_rate: Extract every Nth frame to reduce dataset size
            
        Returns:
            List of frame information dictionaries
        """
        video_file = self.videos_dir / f"video_{video_id:02d}.mp4"
        annotation_file = self.annotations_dir / f"annotations_{video_id:02d}.csv"
        
        if not video_file.exists() or not annotation_file.exists():
            print(f"Missing files for video {video_id}")
            return []
        
        # Determine which split this video belongs to
        split = self._get_split_for_video(video_id)
        images_dir = self._get_images_dir_for_split(split)
        
        frame_info = []
        
        # Read annotations
        with open(annotation_file, 'r') as f:
            reader = csv.DictReader(f)
            annotations = list(reader)
        
        # Sample annotations based on sample_rate
        sampled_annotations = annotations[::sample_rate]
        
        print(f"Collecting frame info for video {video_id} ({split}): {len(sampled_annotations)} frames")
        
        for row in sampled_annotations:
            timestamp = int(row['int_time'])
            instrument1_id = int(row['int_instrument1'])
            
            # Generate frame filename and path (using split-specific directory)
            frame_filename = f"video_{video_id:02d}_frame_{timestamp:06d}.jpg"
            frame_path = images_dir / frame_filename
            
            frame_info.append({
                "video_id": video_id,
                "video_file": str(video_file),
                "timestamp": timestamp,
                "frame_filename": frame_filename,
                "frame_path": str(frame_path),
                "instrument1_id": instrument1_id,
                "step": int(row['int_step']),
                "instrument2_id": int(row['int_instrument2']),
                "split": split
            })
        
        return frame_info
    
    def _extract_all_frames(self, all_frame_info: List[Dict]) -> List[Dict]:
        """
        Extract all frames efficiently using batch processing and parallel execution.
        
        Args:
            all_frame_info: List of frame information dictionaries
            
        Returns:
            List of successfully extracted frame info
        """
        print(f"Extracting {len(all_frame_info)} frames with parallel batch processing...")
        
        # Group frames by video for batch processing
        frames_by_video = defaultdict(list)
        for frame_info in all_frame_info:
            video_id = frame_info['video_id']
            frames_by_video[video_id].append({
                'timestamp': frame_info['timestamp'],
                'output_path': frame_info['frame_path'],
                'video_id': video_id,
                'frame_info': frame_info
            })
        
        print(f"Processing {len(frames_by_video)} videos in parallel...")
        
        all_successful_frames = []
        
        # Process videos in parallel using ProcessPoolExecutor
        max_workers = min(4, len(frames_by_video))  # Limit to 4 processes to avoid overwhelming
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit batch extraction jobs for each video
            future_to_video = {}
            for video_id, frame_requests in frames_by_video.items():
                video_path = frame_requests[0]['frame_info']['video_file']
                future = executor.submit(extract_video_frames_batch, video_path, frame_requests)
                future_to_video[future] = video_id
            
            # Collect results as they complete
            for future in as_completed(future_to_video):
                video_id = future_to_video[future]
                try:
                    successful_extractions = future.result()
                    
                    # Convert back to frame_info format
                    for extraction in successful_extractions:
                        all_successful_frames.append(extraction['frame_info'])
                    
                    print(f"Completed video {video_id}: {len(successful_extractions)} frames extracted")
                    
                except Exception as e:
                    print(f"Error processing video {video_id}: {e}")
        
        print(f"Successfully extracted {len(all_successful_frames)} frames total")
        return all_successful_frames
    
    def _generate_vqa_entries(self, frame_info_list: List[Dict]) -> List[Dict]:
        """
        Generate VQA entries from extracted frame information.
        
        Args:
            frame_info_list: List of frame information dictionaries
            
        Returns:
            List of VQA entries
        """
        print(f"Generating VQA entries for {len(frame_info_list)} frames...")
        vqa_entries = []
        
        for i, frame_info in enumerate(frame_info_list):
            print(f"Processing frame {i} of {len(frame_info_list)}")
            # Generate multiple choice question in A-F format
            formatted_question, correct_letter = self._generate_choices(frame_info['instrument1_id'])
            
            # Create VQA entry with new format
            vqa_entry = {
                "id": f"surgical_{frame_info['video_id']:02d}_{frame_info['timestamp']:06d}",
                "image": frame_info['frame_path'],  # Full path to image
                "question": formatted_question,
                "answer": correct_letter,
                "metadata": {
                    "video_id": frame_info['video_id'],
                    "timestamp": frame_info['timestamp'],
                    "step": frame_info['step'],
                    "instrument1": frame_info['instrument1_id'],
                    "instrument2": frame_info['instrument2_id'],
                    "frame_filename": frame_info['frame_filename']
                }
            }
            
            vqa_entries.append(vqa_entry)
            
            if (i + 1) % 500 == 0:
                print(f"  Generated {i + 1}/{len(frame_info_list)} VQA entries")
        
        return vqa_entries
    
    def convert_dataset(self, splits: List[str] = None, sample_rate: int = 30) -> None:
        """
        Convert the dataset to VQA format with train/test splits.
        Uses a two-phase approach: first extract all frames, then generate VQA entries.
        
        Args:
            splits: List of splits to process ['train', 'test']. If None, process both.
            sample_rate: Extract every Nth frame to reduce dataset size
        """
        if splits is None:
            splits = ['train', 'test']
        
        print(f"Converting dataset to VQA format for splits: {splits}")
        print(f"Sample rate: every {sample_rate} frames")
        print(f"Output directory: {self.output_dir}")
        
        for split in splits:
            print(f"\n{'='*60}")
            print(f"Processing {split.upper()} split")
            print(f"{'='*60}")
            
            # Get video IDs for this split
            if split == "train":
                video_ids = self.train_videos
            elif split == "test":
                video_ids = self.test_videos
            else:
                raise ValueError(f"Invalid split: {split}")
            
            print(f"Videos for {split}: {video_ids}")
            
            # Phase 1: Collect all frame information for this split
            print(f"\n=== Phase 1: Collecting frame information for {split} ===")
            all_frame_info = []
            
            for video_id in video_ids:
                try:
                    frame_info = self._collect_frame_info(video_id, sample_rate)
                    all_frame_info.extend(frame_info)
                except Exception as e:
                    print(f"Error collecting frame info for video {video_id}: {e}")
                    continue
            
            print(f"Collected information for {len(all_frame_info)} frames in {split}")
            
            # Phase 2: Extract all frames for this split
            print(f"\n=== Phase 2: Extracting frames for {split} ===")
            successful_frames = self._extract_all_frames(all_frame_info)
            
            # Phase 3: Generate VQA entries for this split
            print(f"\n=== Phase 3: Generating VQA entries for {split} ===")
            vqa_entries = self._generate_vqa_entries(successful_frames)
            
            # Save split-specific VQA dataset
            output_file = self.output_dir / f"{split}_vqa_dataset.json"
            with open(output_file, 'w') as f:
                json.dump(vqa_entries, f, indent=2)
            
            print(f"\n{split.upper()} conversion complete!")
            print(f"Total VQA entries: {len(vqa_entries)}")
            print(f"Dataset saved to: {output_file}")
            print(f"Images saved to: {self._get_images_dir_for_split(split)}")
            
            # Generate summary statistics for this split
            self._generate_summary(vqa_entries, split)
        
        print(f"\n{'='*60}")
        print("All splits processed successfully!")
        print(f"{'='*60}")
    
    def _generate_summary(self, vqa_entries: List[Dict], split: str) -> None:
        """Generate and save dataset summary statistics for a specific split."""
        print(f"\n--- {split.upper()} Dataset Summary ---")
        
        # Count by instrument using metadata
        instrument_counts = {}
        answer_letter_counts = {}
        
        for entry in vqa_entries:
            # Get instrument ID from metadata
            instrument_id = entry['metadata']['instrument1']
            instrument_text = next(
                (item['text'] for item in self.instrument_map if item['id'] == instrument_id),
                f"unknown_{instrument_id}"
            )
            instrument_counts[instrument_text] = instrument_counts.get(instrument_text, 0) + 1
            
            # Count answer letter distribution
            answer_letter = entry['answer']
            answer_letter_counts[answer_letter] = answer_letter_counts.get(answer_letter, 0) + 1
        
        print("Instrument distribution:")
        for instrument, count in sorted(instrument_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {instrument}: {count}")
        
        print(f"\nAnswer letter distribution:")
        for letter in sorted(answer_letter_counts.keys()):
            print(f"  {letter}: {answer_letter_counts[letter]}")
        
        # Count by video
        video_counts = {}
        for entry in vqa_entries:
            video_id = entry['metadata']['video_id']
            video_counts[video_id] = video_counts.get(video_id, 0) + 1
        
        print(f"\nFrames per video:")
        for video_id in sorted(video_counts.keys()):
            print(f"  Video {video_id:02d}: {video_counts[video_id]} frames")
        
        # Save summary to file
        summary = {
            "split": split,
            "total_entries": len(vqa_entries),
            "instrument_distribution": instrument_counts,
            "answer_letter_distribution": answer_letter_counts,
            "video_distribution": video_counts,
            "sample_question": vqa_entries[0] if vqa_entries else None
        }
        
        summary_file = self.output_dir / f"{split}_dataset_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary saved to: {summary_file}")


def main():
    """Main function to run the VQA conversion."""
    # Configuration
    DATASET_ROOT = "/path/to/your/surgical"
    OUTPUT_DIR = "/path/to/your/surgical_vqa"
    
    # Initialize converter
    converter = SurgicalVQAConverter(DATASET_ROOT, OUTPUT_DIR)
    
    # Configuration for processing
    sample_rate = 6  #
    
    # Process specific splits or both
    # Options: ['train'], ['test'], or ['train', 'test'] for both
    splits_to_process = ['train', 'test']  # Start with test split for faster testing
    
    print("Starting VQA conversion...")
    print(f"Processing splits: {splits_to_process}")
    print(f"Sample rate: every {sample_rate} frames")
    
    converter.convert_dataset(splits=splits_to_process, sample_rate=sample_rate)


if __name__ == "__main__":
    main() 