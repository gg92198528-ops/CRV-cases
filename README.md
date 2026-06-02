CRE-cases
Code and data for Collision Risk Envelopes：A Novel Framework for Accurate Pedestrian-Autonomous Vehicle Interaction Risk Evaluation

# Risk Model Analyzer
An interactive desktop application for the analysis and visualization of vehicle-pedestrian interaction data. This tool calculates, compares, and dynamically visualizes three key traffic safety risk models: Time-to-Collision (TTC), Collision Risk Envelopes (CRE), and Driving Safety Field (DSF).

## Overview
The primary goal of this project is to provide traffic safety researchers with an intuitive tool to analyze complex interaction scenarios. By loading trajectory data, the application automatically generates a synchronized animation of the event alongside real-time plots of the risk metrics, offering deep insights into the temporal and spatial evolution of risk.


### Prerequisites
Python 3.7+   - Python 3.7
pip (Python package installer)- pip （Python包安装程序）
FFmpeg (Optional, but required for saving animations as video). Download from [ffmpeg.org](https://ffmpeg.org/download.html) and ensure it is added to your system's PATH.

### Steps
1.  Clone the repository:
2.  
    Bash  
    git clone https://github.com/your-username/your-repository-name.

3.  CREate a "requirements.txt" file:
    CREate a file named `requirements.txt` in the project root and add the following lines:
    
    pandas
    matplotlib
    numpy
    scipy
    
5.  Install dependencies:

    Bash
    pip install -r requirements.txt
    
4.  Run the application:
    Save the provided Python script as app.py (or any other name) in the project directory and run it from your terminal:

    Bash
    python app.py


#### How to Use

1.  Launch the application.
2.  Click the "Load CSV File" button.
3.  Select a CSV file that adheres to the specified 11-column headerless format.
4.  The analysis and animation will begin automatically.
5.  Use the control buttons to Replay, Pause/Resume, or Save the output.

Technology Stack

  - GUI: [Tkinter](https://docs.python.org/3/library/tkinter.html) (via `tkinter.ttk`)
  - Data Manipulation: [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/)
  - Plotting & Animation: [Matplotlib](https://matplotlib.org/)
  - Data Smoothing: [SciPy](https://scipy.org/)

##### License

This project is licensed under the MIT License. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
