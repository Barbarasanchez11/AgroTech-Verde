import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the main app
from src.app import main

if __name__ == "__main__":
    main() 