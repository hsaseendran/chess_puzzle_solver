#!/usr/bin/env python3

import pandas as pd
import argparse
import os
import logging

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def filter_puzzles_by_theme(input_csv, output_matein1_csv, output_matein2_csv):
    """Filter puzzles by mate-in-X themes and save to separate files"""
    # Load the dataset
    logging.info(f"Loading puzzles from {input_csv}")
    df = pd.read_csv(input_csv)
    logging.info(f"Loaded {len(df)} puzzles")
    
    # Filter by themes
    matein1_df = df[df['Themes'].str.contains('mateIn1', na=False)]
    matein2_df = df[df['Themes'].str.contains('mateIn2', na=False)]
    
    logging.info(f"Found {len(matein1_df)} mateIn1 puzzles")
    logging.info(f"Found {len(matein2_df)} mateIn2 puzzles")
    
    # Save to separate files
    matein1_df.to_csv(output_matein1_csv, index=False)
    matein2_df.to_csv(output_matein2_csv, index=False)
    
    logging.info(f"MateIn1 puzzles saved to {output_matein1_csv}")
    logging.info(f"MateIn2 puzzles saved to {output_matein2_csv}")
    
    # Analyze puzzle difficulty distribution
    if 'Rating' in df.columns:
        logging.info("\nRating Distribution:")
        logging.info(f"MateIn1 - Mean: {matein1_df['Rating'].mean():.0f}, Min: {matein1_df['Rating'].min()}, Max: {matein1_df['Rating'].max()}")
        logging.info(f"MateIn2 - Mean: {matein2_df['Rating'].mean():.0f}, Min: {matein2_df['Rating'].min()}, Max: {matein2_df['Rating'].max()}")

def main():
    parser = argparse.ArgumentParser(description='Filter chess puzzles by mate-in-X theme')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file with puzzles')
    parser.add_argument('--output-dir', type=str, default='filtered_puzzles', help='Output directory')
    parser.add_argument('--output1', type=str, help='Custom output filename for mateIn1 puzzles')
    parser.add_argument('--output2', type=str, help='Custom output filename for mateIn2 puzzles')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define output filenames
    output_matein1_csv = args.output1 if args.output1 else os.path.join(args.output_dir, 'matein1_puzzles.csv')
    output_matein2_csv = args.output2 if args.output2 else os.path.join(args.output_dir, 'matein2_puzzles.csv')
    
    # Filter and save puzzles
    filter_puzzles_by_theme(args.input, output_matein1_csv, output_matein2_csv)

if __name__ == "__main__":
    main()