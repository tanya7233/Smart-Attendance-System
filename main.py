"""
Smart Attendance System using AI Facial Recognition
Main entry point for the application

Author: AI Assistant
Date: 2025
"""

import os
import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('attendance_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories for the application"""
    directories = [
        'data/known_faces',
        'data/reports',
        'data/temp',
        'logs'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

def main():
    """Main function to run the attendance system"""
    try:
        logger.info("Starting Smart Attendance System...")
        
        # Setup required directories
        setup_directories()
        
        # Import and initialize database
        from database import DatabaseManager
        db_manager = DatabaseManager()
        db_manager.initialize_database()
        
        logger.info("Database initialized successfully")
        
        # Start the Streamlit dashboard
        logger.info("Starting Streamlit dashboard...")
        os.system("streamlit run dashboard.py")
        
    except Exception as e:
        logger.error(f"Error starting the application: {str(e)}")
        raise

if __name__ == "__main__":
    main()