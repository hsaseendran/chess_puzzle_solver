#!/usr/bin/env python3
"""
Run training script with proper path setup
"""

import os
import sys
import subprocess

# Get the project root directory
project_root = os.path.dirname(os.path.abspath(__file__))

# Set up the Python path
env = os.environ.copy()
if 'PYTHONPATH' in env:
    env['PYTHONPATH'] = f"{project_root}:{env['PYTHONPATH']}"
else:
    env['PYTHONPATH'] = project_root

# Run the training script
script_path = os.path.join(project_root, 'scripts', 'train.py')
cmd = [sys.executable, script_path] + sys.argv[1:]

# Execute with the updated environment
subprocess.run(cmd, env=env)