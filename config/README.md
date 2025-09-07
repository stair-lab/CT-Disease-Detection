# Configuration Directory

This directory contains configuration files for the enhanced multi-task comorbidity detection system. The configuration system provides flexibility in defining experiments and biomarkers without modifying code.

## 📁 Files Overview

### Core Configuration Files

- **`experiment_config.py`** - Python module for experiment configuration management
- **`biomarker_config.py`** - Python module for biomarker configuration management
- **`__init__.py`** - Python package initialization file

### Biomarker Configuration Files

- **`biomarker_config_default.yaml`** - Default biomarker configuration (YAML format)
- **`biomarker_config_default.json`** - Default biomarker configuration (JSON format)
- **`biomarker_config_example.yaml`** - Example custom biomarker configuration (YAML format)
- **`biomarker_config_example.json`** - Example custom biomarker configuration (JSON format)

## 🧬 Biomarker Configuration

### Purpose
The biomarker configuration system allows you to flexibly specify which biomarkers to include in your training and how they should be handled. This eliminates the need to modify code when changing biomarkers.

### Supported Biomarker Types

1. **Binary Biomarkers** - Two-class classification (e.g., PRESENT/ABSENT)
2. **Multiclass Biomarkers** - Multiple discrete classes (e.g., ABSENT/LOW/MEDIUM/HIGH)
3. **Continuous Biomarkers** - Regression tasks (e.g., age, BMI)

### Configuration Format

#### YAML Format (Recommended)
```yaml
binary_biomarkers:
- name: HCC18
  positive_class: PRESENT
  negative_class: ABSENT
  class_weight: 1.0

multiclass_biomarkers:
- name: CalciumScoring_AbdominalAgatston
  classes: [ABSENT, LOW, MEDIUM, HIGH]
  class_weights:
    ABSENT: 1.0
    LOW: 1.0
    MEDIUM: 1.0
    HIGH: 1.0

continuous_biomarkers:
- name: AGE
  normalization_factor: 101.0
  min_value: null
  max_value: null
```

#### JSON Format
```json
{
  "binary_biomarkers": [
    {
      "name": "HCC18",
      "positive_class": "PRESENT",
      "negative_class": "ABSENT",
      "class_weight": 1.0
    }
  ],
  "multiclass_biomarkers": [
    {
      "name": "CalciumScoring_AbdominalAgatston",
      "classes": ["ABSENT", "LOW", "MEDIUM", "HIGH"],
      "class_weights": {
        "ABSENT": 1.0,
        "LOW": 1.0,
        "MEDIUM": 1.0,
        "HIGH": 1.0
      }
    }
  ],
  "continuous_biomarkers": [
    {
      "name": "AGE",
      "normalization_factor": 101.0,
      "min_value": null,
      "max_value": null
    }
  ]
}
```

### Configuration Parameters

#### Binary Biomarkers
- **`name`** (required) - Column name in your CSV data
- **`positive_class`** (default: "PRESENT") - Value representing positive class
- **`negative_class`** (default: "ABSENT") - Value representing negative class
- **`class_weight`** (default: 1.0) - Weight for class balancing

#### Multiclass Biomarkers
- **`name`** (required) - Column name in your CSV data
- **`classes`** (required) - List of all possible class values
- **`class_weights`** (optional) - Dictionary mapping class names to weights

#### Continuous Biomarkers
- **`name`** (required) - Column name in your CSV data
- **`normalization_factor`** (default: 1.0) - Factor to normalize values (value/factor)
- **`min_value`** (optional) - Minimum expected value for validation
- **`max_value`** (optional) - Maximum expected value for validation

## 🚀 Usage Examples

### Using Default Configuration

The system automatically uses `biomarker_config_default.yaml` if no configuration is specified:

```bash
python train_enhanced.py \
    --config_csv experimentation_plan_simplified.csv \
    --data_dir data \
    --model_name "ResNet-18" \
    --epochs 50
```

### Using Custom Configuration

Specify a custom biomarker configuration file:

```bash
python train_enhanced.py \
    --config_csv experimentation_plan_simplified.csv \
    --data_dir data \
    --model_name "ResNet-18" \
    --biomarker_config config/my_custom_biomarkers.yaml \
    --epochs 50
```

### Creating Custom Configurations

#### Method 1: Edit Existing Files
Copy and modify one of the existing configuration files:
```bash
cp config/biomarker_config_default.yaml config/my_biomarkers.yaml
# Edit my_biomarkers.yaml with your preferred editor
```

#### Method 2: Generate Programmatically
```python
from config.biomarker_config import BiomarkerConfig, BinaryBiomarker, MultiClassBiomarker, ContinuousBiomarker

# Create custom configuration
custom_config = BiomarkerConfig(
    binary_biomarkers=[
        BinaryBiomarker(name='Diabetes', positive_class='YES', negative_class='NO'),
        BinaryBiomarker(name='Hypertension', class_weight=2.0)
    ],
    multiclass_biomarkers=[
        MultiClassBiomarker(
            name='Disease_Stage', 
            classes=['I', 'II', 'III', 'IV'],
            class_weights={'I': 1.0, 'II': 1.5, 'III': 2.0, 'IV': 2.5}
        )
    ],
    continuous_biomarkers=[
        ContinuousBiomarker(name='BMI', normalization_factor=50.0, min_value=10, max_value=60),
        ContinuousBiomarker(name='BloodPressure', normalization_factor=200.0)
    ]
)

# Save to file
custom_config.save_to_yaml('config/my_custom_biomarkers.yaml')
custom_config.save_to_json('config/my_custom_biomarkers.json')
```

## 📊 Default Configuration Details

The default configuration (`biomarker_config_default.yaml`) matches the original hardcoded system:

- **Binary Biomarkers**: HCC18, HCC22, HCC85, HCC96, HCC108, HCC111 (6 total)
- **Multiclass Biomarkers**: CalciumScoring_AbdominalAgatston with 4 classes (ABSENT, LOW, MEDIUM, HIGH)
- **Continuous Biomarkers**: AGE (normalized by 101.0), RAF (normalized by 50.0)
- **Total Output Size**: 12 (6 binary + 4 multiclass + 2 continuous)

## 🔧 Advanced Usage

### Validation and Error Handling

The system automatically validates configurations:
- Checks for duplicate biomarker names
- Validates class mappings for multiclass biomarkers
- Handles missing values gracefully
- Provides informative error messages

### Configuration Testing

Test your configuration before training:

```python
from config.biomarker_config import BiomarkerConfig

# Load and validate configuration
config = BiomarkerConfig.load_from_yaml('config/my_biomarkers.yaml')

print(f"Total output size: {config.total_output_size}")
print(f"Biomarkers: {config.all_biomarker_names}")
print(f"Tensor layout: {config.get_tensor_layout()}")
```

### Integration with Dataset

The dataset automatically adapts to your biomarker configuration:

```python
from dataset import ClassifierDataset
from config.biomarker_config import BiomarkerConfig

# Load configuration
biomarker_config = BiomarkerConfig.load_from_yaml('config/my_biomarkers.yaml')

# Dataset automatically uses the configuration
dataset = ClassifierDataset('data', biomarker_config, transforms=transform, train=True)
```

## 🎯 Best Practices

1. **Use YAML Format**: More readable and easier to edit than JSON
2. **Descriptive Names**: Use clear, descriptive biomarker names
3. **Consistent Naming**: Match column names exactly as they appear in your CSV
4. **Backup Configurations**: Keep copies of working configurations
5. **Test First**: Validate configurations with small datasets before full training
6. **Document Changes**: Add comments in YAML files to explain custom configurations

## 🔍 Troubleshooting

### Common Issues

1. **Missing Column Error**: Ensure biomarker names match CSV column names exactly
2. **Unknown Class Value**: Check that all possible values are listed in multiclass `classes`
3. **Normalization Issues**: Verify normalization factors for continuous biomarkers
4. **File Format Error**: Ensure YAML/JSON syntax is correct

### Debug Commands

```python
# Test configuration loading
from config.biomarker_config import BiomarkerConfig
config = BiomarkerConfig.load_from_yaml('config/my_biomarkers.yaml')

# Test with dataset
from dataset import ClassifierDataset
dataset = ClassifierDataset('data', config, train=True)
image, labels = dataset[0]
print(f"Labels shape: {labels.shape}, Values: {labels}")
```

## 📚 Additional Resources

- See `biomarker_config_example.yaml` for advanced configuration examples
- Check the main training script for command-line usage examples
- Refer to the dataset.py file for implementation details

---

For questions or issues with configuration files, please refer to the main project documentation or create an issue in the project repository.
