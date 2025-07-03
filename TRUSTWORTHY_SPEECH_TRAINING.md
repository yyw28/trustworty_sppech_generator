# Trustworthy Speech Training with HuBERT

This guide explains how to train a HuBERT model on your trustworthy speech dataset for binary classification of trustworthy vs. non-trustworthy speech.

## Dataset Structure

Your dataset should have the following structure:
```
trustworty_sppech_generator/
├── collected_ratings/
│   └── processed_df_rec_hum.csv  # CSV with filename and trustworthy columns
└── Audio/
    └── recommendation_humor_audio/  # Directory containing .wav files
        ├── q1_F_saved_audio_files_wav/
        ├── q1_M_saved_audio_files_wav/
        └── ...
```

## CSV Format

The CSV file should contain at least these columns:
- `filename`: The name of the audio file (without .wav extension)
- `trustworthy`: The trustworthy score (will be converted to binary: >0.5 = trustworthy)

## Setup

1. **Install Dependencies**
   ```bash
   pip install torch torchaudio transformers lightning pandas scikit-learn
   ```

2. **Test Data Loading**
   ```bash
   python test_data_loading.py
   ```
   This will verify that your data can be loaded correctly.

## Training

### Method 1: Using the Training Script (Recommended)

```bash
python train_trustworthy_speech.py fit \
    --config config/trustworthy_speech_finetune.json \
    --data.csv_file collected_ratings/processed_df_rec_hum.csv \
    --data.audio_dir Audio/recommendation_humor_audio
```

### Method 2: Using the Original Script

```bash
python src/tspeech/hubert.py fit \
    --config config/trustworthy_speech_finetune.json
```

## Configuration

The training configuration is in `config/trustworthy_speech_finetune.json`. Key parameters:

- **Model**: Uses `facebook/hubert-base-ls960` with 10 trainable layers
- **Data**: Batch size of 4, 5 workers
- **Training**: 50 epochs max, early stopping with patience of 5
- **Optimizer**: AdamW with learning rate 1e-6

## Customization

### Using Different Datasets

To use a different CSV file or audio directory, modify the configuration:

```json
{
    "data": {
        "class_path": "tspeech.data.TrustworthySpeechDataModule",
        "init_args": {
            "csv_file": "path/to/your/data.csv",
            "audio_dir": "path/to/your/audio/files",
            "batch_size": 4,
            "num_workers": 5
        }
    }
}
```

### Adjusting the Trustworthy Threshold

In `src/tspeech/data/trustworthy_speech_dataset.py`, you can modify the binary classification threshold:

```python
# Change this line to adjust the threshold
trustworthy_binary = torch.tensor(
    [[trustworthy_score > 0.5]], dtype=torch.float  # Change 0.5 to your threshold
)
```

### Using Different HuBERT Models

You can use different pre-trained HuBERT models by changing the `hubert_model_name` in the config:

- `facebook/hubert-base-ls960` (default)
- `facebook/hubert-large-ls960-ft`
- `facebook/hubert-xlarge-ls960-ft`

## Training Output

The training will create:
- **Checkpoints**: Saved in the current directory with names like `trustworthy_speech_checkpoint-epoch-{epoch}-{validation_loss:.5f}.ckpt`
- **Logs**: TensorBoard logs in `lightning_logs/trustworthy_speech_finetune/`

## Monitoring Training

To monitor training with TensorBoard:
```bash
tensorboard --logdir lightning_logs/trustworthy_speech_finetune
```

## Inference

After training, you can load the model for inference:

```python
import torch
from tspeech.model.trustworthiness import TrustworthinessClassifier

# Load the trained model
model = TrustworthinessClassifier.load_from_checkpoint("path/to/checkpoint.ckpt")
model.eval()

# Load and preprocess audio
# ... (implement audio loading similar to the dataset)

# Make prediction
with torch.no_grad():
    prediction = model(wav, mask)
    probability = torch.sigmoid(prediction)
    is_trustworthy = probability > 0.5
```

## Troubleshooting

### Common Issues

1. **Audio files not found**: Make sure the filenames in your CSV match the actual .wav files
2. **Memory issues**: Reduce batch size in the configuration
3. **CUDA out of memory**: Use `--trainer.precision 16-mixed` or reduce batch size

### Data Validation

Run the test script to validate your data:
```bash
python test_data_loading.py
```

This will check:
- CSV file exists and can be loaded
- Audio directory exists
- Audio files can be found and loaded
- Data shapes are correct

## Performance Tips

1. **Use GPU**: The configuration automatically detects and uses available GPUs
2. **Mixed Precision**: Already enabled in the config for faster training
3. **Data Loading**: Adjust `num_workers` based on your system (more workers = faster loading)
4. **Batch Size**: Increase if you have more GPU memory available

## Model Architecture

The model uses:
- **HuBERT Encoder**: Pre-trained speech representation model
- **Pooling**: Mean pooling over the sequence dimension
- **Classifier**: Linear layer for binary classification
- **Loss**: Binary cross-entropy with logits

The model fine-tunes the last 10 layers of HuBERT by default, which provides a good balance between adaptation and preventing overfitting. 