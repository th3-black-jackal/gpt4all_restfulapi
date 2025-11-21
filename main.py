import sys
import json
import requests
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QLineEdit, QTextEdit, QPushButton, 
    QLabel, QSpinBox, QDoubleSpinBox, QMessageBox
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
import json

# --- Configuration ---
API_URL = "BACKEND_SERVER/query" # Ensure this is the correct URL

# --- Worker Thread for API Calls ---

class ApiWorker(QThread):
    """
    A QThread worker to handle the synchronous API call without blocking 
    the main UI thread.
    """
    # Signals for communication: (result_text) or (error_message)
    result_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, data: dict, parent=None):
        super().__init__(parent)
        self.data = data

    def run(self):
        """Execute the API call."""
        try:
            # Send the POST request
            response = requests.post(
                API_URL, 
                json=self.data, 
                headers={"Content-Type": "application/json"},
                timeout=300  # Increased timeout to 5 minutes (300 seconds)
            )
            
            # Check for HTTP errors (4xx or 5xx)
            response.raise_for_status()
            
            # Process successful response
            response_json = response.json()
            
            # Extract the generated text from the JSON response structure
            if 'response' in response_json:
                generated_text = response_json['response']
                self.result_ready.emit(generated_text)
            else:
                self.error_occurred.emit(f"API Response Error: 'response' key missing. Full response: {response.text}")

        except requests.exceptions.RequestException as e:
            # Handle network errors (connection, timeout, DNS, etc.)
            error_message = f"Network Error: Could not connect to FastAPI server at {API_URL}. \n\nPlease ensure the server is running."
            self.error_occurred.emit(error_message)
        except json.JSONDecodeError:
            self.error_occurred.emit(f"JSON Decode Error: Failed to parse response from server: {response.text}")
        except Exception as e:
            self.error_occurred.emit(f"An unexpected error occurred: {e}")


# --- Main Application Window ---

class LLMClientApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GPT4All FastAPI Client")
        self.setGeometry(100, 100, 800, 600)

        self.worker_thread = None
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 1. Prompt Input Area
        main_layout.addWidget(QLabel("Prompt:"))
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("Enter your query here...")
        self.prompt_input.setFixedHeight(120)
        main_layout.addWidget(self.prompt_input)

        # 2. Parameters (Max Tokens & Temperature)
        params_layout = QHBoxLayout()
        
        # Max Tokens
        params_layout.addWidget(QLabel("Max Tokens:"))
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(1, 4096)
        self.max_tokens_spin.setValue(256)
        self.max_tokens_spin.setToolTip("Maximum number of tokens to generate.")
        params_layout.addWidget(self.max_tokens_spin)
        
        # Temperature
        params_layout.addWidget(QLabel("Temperature:"))
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setValue(0.7)
        self.temp_spin.setToolTip("Sampling temperature for creativity (0.0=deterministic, 2.0=highly creative).")
        params_layout.addWidget(self.temp_spin)
        
        params_layout.addStretch() # Push widgets to the left
        main_layout.addLayout(params_layout)

        # 3. Control Button
        self.generate_button = QPushButton("Generate Response")
        self.generate_button.clicked.connect(self.send_query)
        main_layout.addWidget(self.generate_button)

        # 4. Response Output Area
        main_layout.addWidget(QLabel("Model Response:"))
        self.response_output = QTextEdit()
        self.response_output.setReadOnly(True)
        self.response_output.setText("Waiting for model query...")
        main_layout.addWidget(self.response_output)

    def send_query(self):
        """Gathers input and initiates the API call in a separate thread."""
        description = self.prompt_input.toPlainText().strip()
        if not description:
            QMessageBox.warning(self, "Input Error", "Please enter a description before generating.")
            return

        # Disable UI and show loading state
        self.generate_button.setEnabled(False)
        self.generate_button.setText("Generating... (Please wait)")
        self.response_output.setText("Sending request to FastAPI server...")
        QApplication.processEvents() # Force UI update immediately
        
        # Construct the data payload
        prompt = (f"Given the cybersecurity scenario description: '{description}', identify and list the key terms, "
              "techniques, or technologies relevant to MITRE ATT&CK. Extract TTPs from the scenario. "
              "If the description is too basic, expand upon it with additional details, applicable campaign, "
              "or attack types based on dataset knowledge. Then, extract the TTPs from the revised description.")
        data = {
            "prompt": prompt,
            "max_tokens": self.max_tokens_spin.value(),
            "temp": self.temp_spin.value()
        }

        # Create and start the worker thread
        self.worker_thread = ApiWorker(data)
        self.worker_thread.result_ready.connect(self.handle_result)
        self.worker_thread.error_occurred.connect(self.handle_error)
        self.worker_thread.finished.connect(self.reset_ui) # Always reset UI when thread is done
        self.worker_thread.start()

    def handle_result(self, result_text: str):
        """Receives successful result from the worker and updates the UI."""
        self.response_output.setText(result_text)

    def handle_error(self, error_message: str):
        """Receives error message from the worker and updates the UI/shows a message box."""
        self.response_output.setText(f"ERROR:\n{error_message}")
        QMessageBox.critical(self, "API Error", error_message)

    def reset_ui(self):
        """Resets the UI elements after the thread finishes."""
        self.generate_button.setEnabled(True)
        self.generate_button.setText("Generate Response")
        self.worker_thread = None


if __name__ == '__main__':
    # Add dependency check reminder
    print("Dependencies required: PyQt6, requests")
    print("Install with: pip install PyQt6 requests")
    print("----------------------------------------------------------------------")
    print(f"Client is configured to connect to: {API_URL}")
    print("PLEASE ENSURE YOUR FASTAPI SERVER (gpt4all_api.py) IS RUNNING FIRST.")
    print("----------------------------------------------------------------------")

    app = QApplication(sys.argv)
    window = LLMClientApp()
    window.show()
    sys.exit(app.exec())
