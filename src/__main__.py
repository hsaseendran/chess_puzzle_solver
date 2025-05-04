#!/usr/bin/env python3
"""
Main entry point for the chess puzzle solver package
"""

import sys
import os

def print_usage():
    print("Chess Puzzle Solver")
    print("Usage:")
    print("  python -m src train --data <puzzle_file> [options]")
    print("  python -m src evaluate --model <model_file> --data <puzzle_file> [options]")
    print("")
    print("Commands:")
    print("  train     Train the chess puzzle solver")
    print("  evaluate  Evaluate a trained model")
    print("")
    print("For more options, use: python -m src <command> --help")

def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    command = sys.argv[1]
    
    # Remove the command from argv so the scripts see the correct arguments
    sys.argv = sys.argv[1:]
    
    if command == "train":
        from scripts.train import main as train_main
        train_main()
    elif command == "evaluate":
        from scripts.evaluate import main as eval_main
        eval_main()
    else:
        print(f"Unknown command: {command}")
        print_usage()
        sys.exit(1)

if __name__ == "__main__":
    main()