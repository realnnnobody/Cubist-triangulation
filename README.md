# Cubist-triangulation

A Python image-processing project that transforms an input image into a cubist, triangulated-style result.

## Requirements

- Python 3.12.3
- `llvmlite==0.46.0`
- `numba==0.64.0`
- `numpy==2.4.3`
- `opencv-python==4.13.0.92`
- `scipy==1.17.1`
- `setuptools==82.0.1`

## Installation

Clone the repository:

```bash
git clone https://github.com/realnnnobody/Cubist-triangulation.git
cd Cubist-triangulation
```

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the program with an input image and save the result to an output file:
```bash
python main.py input.png -o output.png
```