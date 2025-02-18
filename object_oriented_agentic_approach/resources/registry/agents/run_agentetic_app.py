#!/usr/bin/env python3
"""
Agentic Application Orchestration Script

This script orchestrates two agents: 
1. FileAccessAgent – reads file content (e.g., traffic_accidents.csv) and provides context.
2. PythonExecAgent – uses dynamic tool calling to generate and execute Python code based on the user's question.

The generated Python code is executed in a secure, isolated Docker container.

Prerequisites:
- Docker installed and running.
- Python 3.10+ installed.
- OpenAI API key configured appropriately.
- The project structure must contain the agents under 'resources/registry/agents/'.
"""

import os
import sys
import logging

# Add the project root to sys.path so that the 'resources' module is found.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging for debugging and error handling.
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Import our agents.
try:
    from resources.registry.agents.file_access_agent import FileAccessAgent
    from resources.registry.agents.python_code_exec_agent import PythonExecAgent
except ImportError as e:
    logging.error("Failed to import agent modules. Verify that PYTHONPATH and project structure are set correctly.")
    logging.exception(e)
    sys.exit(1)


def main():
    try:
        # Define the prompt that describes the file contents and format.
        prompt = (
            "Use the file traffic_accidents.csv for your analysis. The column names are:\n\n"
            "Variable\tDescription\n"
            "accidents\tNumber of recorded accidents, as a positive integer\n"
            "traffic_fine_amount\tTraffic fine amount, expressed in thousands of USD\n"
            "traffic_density\tTraffic density index, scale from 0 (low) to 10 (high)\n"
            "traffic_lights\tProportion of traffic lights in the area (0 to 1)\n"
            "pavement_quality\tPavement quality, scale from 0 (very poor) to 5 (excellent)\n"
            "urban_area\tUrban area (1) or rural area (0), as an integer\n"
            "average_speed\tAverage speed of vehicles in km/h\n"
            "rain_intensity\tRain intensity, scale from 0 (no rain) to 3 (heavy rain)\n"
            "vehicle_count\tEstimated number of vehicles, in thousands, as an integer\n"
            "time_of_day\tTime of day in 24-hour format (0 to 24)"
        )

        print("Setup:")
        print(prompt)
        logging.info("Setting up the agents...")

        # Check if required files exist
        csv_path = os.path.join(project_root, "resources", "data", "traffic_accidents.csv")
        if not os.path.exists(csv_path):
            logging.error(f"Required file not found: {csv_path}")
            print("Error: Required data file not found. Please ensure traffic_accidents.csv exists in the resources/data directory.")
            sys.exit(1)

        # Instantiate agents with proper resource management
        file_ingestion_agent = None
        data_analysis_agent = None
        
        try:
            file_ingestion_agent = FileAccessAgent()
            data_analysis_agent = PythonExecAgent(model_name="o3-mini", reasoning_effort="high")

            print("Understanding the contents of the file...")
            file_ingestion_agent_output = file_ingestion_agent.task(prompt)

            # Add the prompt and file context to the PythonExecAgent
            data_analysis_agent.add_context(prompt)
            data_analysis_agent.add_context(file_ingestion_agent_output)

            # Main interaction loop
            while True:
                print("\nType your question related to the data in the file (or type 'exit' to exit):")
                try:
                    user_input = input("Your question: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nExiting the application.")
                    break

                if user_input.lower() == "exit":
                    print("Exiting the application.")
                    break

                print(f"User question: {user_input}")
                print("Generating dynamic tools and executing code interpreter...")

                try:
                    data_analysis_agent_output = data_analysis_agent.task(user_input)
                    print("Output:")
                    print(data_analysis_agent_output)
                except Exception as e:
                    logging.error("Error during dynamic code generation or execution.")
                    logging.exception(e)
                    print("An error occurred while processing your request. Please try again.")

        finally:
            # Cleanup resources
            if file_ingestion_agent:
                try:
                    file_ingestion_agent.cleanup()
                except Exception as e:
                    logging.error("Error cleaning up file ingestion agent")
                    logging.exception(e)
            
            if data_analysis_agent:
                try:
                    data_analysis_agent.cleanup()
                except Exception as e:
                    logging.error("Error cleaning up data analysis agent")
                    logging.exception(e)

    except KeyboardInterrupt:
        print("\nApplication interrupted by user. Exiting gracefully.")
    except Exception as e:
        logging.error("Unexpected error in the main execution loop.")
        logging.exception(e)
        sys.exit(1)


if __name__ == "__main__":
    main()