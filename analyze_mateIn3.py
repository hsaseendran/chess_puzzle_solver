#!/usr/bin/env python3
"""
Analyze how many mateIn3 puzzles are in the dataset
"""

import pandas as pd
import argparse
from collections import Counter


def analyze_dataset(csv_path):
    """Analyze the dataset for mateIn3 puzzles"""
    print(f"Loading dataset from {csv_path}")
    
    df = pd.read_csv(csv_path)
    total_puzzles = len(df)
    
    print(f"Total puzzles in dataset: {total_puzzles}")
    
    # Filter for Themes column
    df_with_themes = df.dropna(subset=['Themes'])
    
    print(f"Puzzles with themes: {len(df_with_themes)}")
    
    # Analyze all themes
    all_themes = []
    mate_themes = []
    
    for themes_str in df_with_themes['Themes']:
        if pd.notna(themes_str):
            themes = themes_str.split()
            all_themes.extend(themes)
            
            # Check for mate themes
            for theme in themes:
                if 'mate' in theme.lower():
                    mate_themes.append(theme)
    
    # Count themes
    theme_counts = Counter(all_themes)
    mate_theme_counts = Counter(mate_themes)
    
    print(f"\nTotal unique themes: {len(theme_counts)}")
    print(f"Most common themes:")
    for theme, count in theme_counts.most_common(20):
        print(f"  {theme}: {count}")
    
    print(f"\nMate-related themes:")
    for theme, count in mate_theme_counts.most_common():
        print(f"  {theme}: {count}")
    
    # Filter for mateIn3
    matein3_puzzles = df_with_themes[df_with_themes['Themes'].str.contains('mateIn3', na=False)]
    
    print(f"\nMateIn3 puzzles found: {len(matein3_puzzles)}")
    print(f"Percentage of total: {len(matein3_puzzles) / total_puzzles * 100:.2f}%")
    
    # Analyze ratings
    if 'Rating' in matein3_puzzles.columns:
        ratings = matein3_puzzles['Rating'].dropna()
        if not ratings.empty:
            print(f"\nRating statistics for mateIn3 puzzles:")
            print(f"  Mean: {ratings.mean():.0f}")
            print(f"  Median: {ratings.median():.0f}")
            print(f"  Min: {ratings.min():.0f}")
            print(f"  Max: {ratings.max():.0f}")
    
    # Show some examples
    print("\nExample mateIn3 puzzles:")
    for i, (_, puzzle) in enumerate(matein3_puzzles.head(5).iterrows()):
        print(f"\nPuzzle {i+1}:")
        print(f"  FEN: {puzzle['FEN']}")
        print(f"  Moves: {puzzle['Moves']}")
        print(f"  Rating: {puzzle.get('Rating', 'N/A')}")
        print(f"  Themes: {puzzle['Themes']}")
    
    return matein3_puzzles


def main():
    parser = argparse.ArgumentParser(description="Analyze dataset for mateIn3 puzzles")
    parser.add_argument("--data", type=str, required=True, help="Path to puzzle CSV file")
    args = parser.parse_args()
    
    matein3_puzzles = analyze_dataset(args.data)


if __name__ == "__main__":
    main()