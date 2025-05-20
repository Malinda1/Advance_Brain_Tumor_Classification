# Brain Tumor Classification System

A deep learning-based web application for detecting brain tumors from MRI scans using a Convolutional Neural Network (CNN) model.

## Overview

This project implements a brain tumor detection system that uses deep learning to analyze MRI scans and classify them as either containing a brain tumor or not. The system provides a user-friendly web interface for uploading MRI images and receiving predictions with confidence scores.

## Features

- Real-time brain tumor detection from MRI scans
- User-friendly web interface
- Confidence score visualization
- Risk stratification analysis
- Support for multiple image formats (PNG, JPG, JPEG, WEBP)
- Drag-and-drop image upload functionality
- Responsive design for various screen sizes

## Technical Stack

- **Backend:**
  - Python 3.x
  - Flask (Web Framework)
  - TensorFlow/Keras (Deep Learning)
  - NumPy (Numerical Computing)

- **Frontend:**
  - HTML5
  - CSS3
  - JavaScript
  - Chart.js (Visualization)

## Project Structure

```
├── brain_tumor_app/           # Main application directory
│   ├── app.py                # Flask application
│   ├── templates/            # HTML templates
│   │   └── index_2.html     # Main web interface
│   └── static/              # Static files (CSS, JS, uploads)
├── data/                     # Dataset directory
│   └── tumorous_and_nontumorous/
│       ├── train/           # Training data
│       ├── test/            # Test data
│       └── val/             # Validation data
├── model/                    # Model architecture files
├── saved_models/            # Saved model checkpoints
├── logs/                    # Training logs
├── best_model.h5           # Best performing model
└── requirements.txt        # Project dependencies
```

## Detailed Component Explanations

### 1. Training Notebooks

#### Brain_Tumor_Classification.ipynb
This notebook contains the main model training pipeline:
- Data preprocessing and augmentation
- Model architecture definition
- Training loop implementation
- Model evaluation metrics
- Visualization of training results
- Key features:
  - Image resizing to 224x224 pixels
  - Data augmentation (rotation, flip, zoom)
  - Batch normalization
  - Dropout layers for regularization
  - Learning rate scheduling
  - Early stopping implementation

#### fine_tuning_model.ipynb
This notebook focuses on model optimization:
- Transfer learning implementation
- Hyperparameter tuning
- Model architecture modifications
- Performance optimization
- Key features:
  - Pre-trained model adaptation
  - Layer freezing/unfreezing
  - Learning rate fine-tuning
  - Model checkpointing
  - Performance metrics tracking

### 2. Flask Server (app.py)

The Flask server implements the backend API with the following features:
- Image upload handling
- Model inference
- Error handling
- CORS support
- File validation
- Key endpoints:
  ```python
  @app.route('/')              # Main web interface
  @app.route('/upload')        # Image upload endpoint
  @app.route('/', methods=['POST'])  # Prediction endpoint
  ```
- Security features:
  - File type validation
  - File size limits
  - Secure filename handling
  - Error logging

### 3. Web Interface (index_2.html)

The frontend interface provides:
- Modern, responsive design
- Interactive features:
  - Drag-and-drop file upload
  - Real-time image preview
  - Progress indicators
  - Error handling
- Visualization components:
  - Confidence score display
  - Risk stratification chart
  - Image preview
- User experience features:
  - Clear feedback messages
  - Loading states
  - Error notifications
  - Mobile responsiveness

### 4. Model Architecture

The CNN model architecture includes:
- Input Layer: 224x224x3 (RGB images)
- Convolutional Layers:
  - Multiple Conv2D layers with ReLU activation
  - MaxPooling2D for dimensionality reduction
  - BatchNormalization for training stability
- Dense Layers:
  - Fully connected layers with dropout
  - Final sigmoid activation for binary classification
- Training Features:
  - Binary cross-entropy loss
  - Adam optimizer
  - Learning rate scheduling
  - Early stopping
  - Model checkpointing

### 5. Dataset Information

#### Dataset Structure
The dataset is organized into three main directories:
```
data/tumorous_and_nontumorous/
├── train/           # Training set
│   ├── yes/        # Images with tumors
│   └── no/         # Images without tumors
├── test/           # Test set
│   ├── yes/        # Images with tumors
│   └── no/         # Images without tumors
└── val/            # Validation set
    ├── yes/        # Images with tumors
    └── no/         # Images without tumors
```

#### Dataset Characteristics
- **Image Format**: MRI scans in JPG/PNG format
- **Image Size**: Original images are resized to 224x224 pixels
- **Class Distribution**:
  - Tumor (Yes): MRI scans showing brain tumors
  - No Tumor (No): Normal brain MRI scans
- **Data Split**:
  - Training Set: 70% of total images
  - Validation Set: 15% of total images
  - Test Set: 15% of total images

#### Data Preprocessing
1. **Image Resizing**:
   - All images are resized to 224x224 pixels
   - Maintains aspect ratio with padding
   - RGB color space

2. **Data Augmentation**:
   - Random rotation (±20 degrees)
   - Horizontal and vertical flips
   - Zoom range (0.8-1.2)
   - Brightness adjustment (±20%)
   - Contrast adjustment (±20%)

3. **Normalization**:
   - Pixel values scaled to [0,1] range
   - Mean subtraction
   - Standard deviation normalization

#### Dataset Statistics
- Total number of images: 8,378
- Training set size: 4,189 (2,100 tumor, 2,089 no tumor)
- Validation set size: ~1,047 images
- Test set size: ~1,047 images
- Class balance: Approximately 1:1 (balanced dataset)

#### Data Quality
- Invalid images are moved to `invalid_images/` directory
- Quality checks include:
  - Image format validation
  - Resolution verification
  - Artifact detection
  - Contrast assessment

#### Usage Guidelines
1. **Training**:
   - Use training set for model training
   - Apply data augmentation
   - Monitor class balance

2. **Validation**:
   - Use validation set for hyperparameter tuning
   - Monitor model performance
   - Prevent overfitting

3. **Testing**:
   - Use test set for final evaluation
   - No data augmentation
   - Report final metrics

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd brain-tumor-classification
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask server:
```bash
cd brain_tumor_app
python app.py
```

2. Access the web interface:
   - Open your web browser
   - Navigate to http://localhost:5001

3. Using the application:
   - Click "Upload" or drag and drop an MRI image
   - Click "Analyze Image" to process
   - View the prediction results and confidence score

## Model Details

The system uses a Convolutional Neural Network (CNN) trained on a dataset of brain MRI scans. The model:
- Processes images at 224x224 resolution
- Uses transfer learning for improved accuracy
- Provides confidence scores for predictions
- Supports real-time inference

## Training

The model training process is documented in:
- `Brain_Tumor_Classification.ipynb`: Main training notebook
- `fine_tuning_model.ipynb`: Fine-tuning and optimization

## API Endpoints

- `GET /`: Main web interface
- `POST /`: Image upload and prediction endpoint
- `POST /upload`: Alternative upload endpoint

## Performance

The model provides:
- Binary classification (Tumor/No Tumor)
- Confidence scores
- Risk stratification (Low/Medium/High)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

## Acknowledgments

- Dataset providers
- Open-source community
- Research papers and resources used

## Contact

For any questions or support, please open an issue in the repository. 