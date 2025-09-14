# Enhanced Multi-Task Comorbidity Detection with Flexible Configuration System

This enhanced training system supports comprehensive experimentation with 25+ different deep learning architectures for multi-task comorbidity detection from CT scans, featuring a **flexible biomarker configuration system** that eliminates hardcoded assumptions.

## 🎯 Overview

The system now features a **completely flexible biomarker configuration system** that dynamically adapts to any task structure without requiring code changes. This allows for easy experimentation with different combinations of binary classification, multiclass classification, and regression tasks.

## 🚀 Key Features

### ✅ **Flexible Multi-Task Learning System**
- **Dynamic Task Configuration**: No more hardcoded assumptions about number of tasks
- **YAML/JSON Configuration**: Define tasks and their properties in configuration files
- **Automatic Tensor Layout**: System automatically generates appropriate tensor layouts
- **Dataset Compatibility Validation**: Automatic checking of dataset compatibility
- **Backward Compatibility**: Existing experiments continue to work

### ✅ **Architecture Support**
- **25+ Model Architectures**: CNNs, Vision Transformers, Vision-Language Models, Diffusion Models
- **Flexible Multi-Task Head**: Adapts to any number and combination of tasks
- **Comprehensive Logging**: TensorBoard integration with loss curves, metrics per biomarker
- **Advanced Checkpointing**: Best model selection based on average AUROC across biomarkers
- **Class Balancing**: Inverse frequency weighting and balanced batch sampling
- **Memory Management**: GPU memory requirement checking and optimization

### ✅ **Your Current Configuration**
- **12 Binary Tasks**: GENDER, MORTALITY, HCC codes (HCC12, HCC18, HCC19, HCC22, HCC48, HCC85, HCC96, HCC108, HCC111), CALCIUMSCORING_ABDOMINALAGATSTON_BINARY
- **1 Regression Task**: AGE (normalized to [0,1] from range 18-102)
- **0 Multiclass Tasks**: None for current experiment
- **Total Output Size**: 13

### 🔧 Architecture Support Status
- ✅ **Fully Implemented**: ResNet (18/34/50), DenseNet-121, EfficientNet (B0/B4), ConvNeXt-Base
- ✅ **Implemented with TIMM**: ViT variants (DINOv2, MAE), Swin Transformer, MaxViT
- 🔄 **Placeholder Implementation**: CLIP, BLIP-2, Medical VLMs, Diffusion Models
- 📋 **Future Enhancement**: Full CLIP/VLM integration with proper text encoders

## 📁 Project Structure

```
CT-Disease-Detection/
├── model/
│   ├── model_factory.py           # Model factory with flexible head integration
│   ├── flexible_multitask_head.py # Flexible multi-task components
│   ├── resnet34.py               # Original ResNet-34 implementation
│   └── cc_resnet.py              # Coordinate convolution ResNet
├── config/
│   ├── biomarker_config.py       # Flexible configuration system
│   ├── biomarker_config_comorbidities.yaml # Your specific biomarker config
│   ├── experiment_config.py       # Experiment configuration system
│   └── __init__.py
├── train.py                       # Flexible training pipeline
├── run_experiments.py             # Flexible experiment runner
├── test_setup.py                 # Setup verification script
├── experimentation_plan_simplified.csv # Experiment configurations
├── requirements_enhanced.txt      # Enhanced dependencies
└── README.md                     # This file
```

## 🛠 Setup

### 1. Environment Setup

```bash
# Activate your conda environment
conda activate mahmedc_env

# Install enhanced dependencies
pip install -r requirements_enhanced.txt
```

### 2. Verify Setup

```bash
# Test that everything is working
python test_setup.py
```

### 3. Prepare Data

Ensure your data directory contains:
- `train.csv` - Training data labels
- `val.csv` - Validation data labels  
- `test.csv` - Test data labels
- `data/` - Directory with CT scan images (PNG format)

## 🏃 Running Experiments

### Quick Start - Single Model

```bash
# Run a specific model with flexible configuration
python train.py \
    --config_csv experimentation_plan_simplified.csv \
    --biomarker_config config/biomarker_config_default.yaml \
    --data_dir /lfs/turing1/0/mahmedc/Comorbidities-Detection/datasets/full_data \
    --model_name "ResNet-18" \
    --epochs 100
```

### Batch Experiments - Must Include Models

```bash
# Run all must-include experiments with flexible configuration
python run_experiments.py \
    --config_csv experimentation_plan_simplified.csv \
    --biomarker_config config/biomarker_config_default.yaml \
    --data_dir /lfs/turing1/0/mahmedc/Comorbidities-Detection/datasets/full_data \
    --output_base_dir /lfs/turing1/0/mahmedc/Comorbidities-Detection/models \
    --epochs 100 \
    --must_include_only \
    --check_memory
```

### Advanced Usage

```bash
# Dry run to see what would be executed
python run_experiments.py \
    --config_csv experimentation_plan_simplified.csv \
    --biomarker_config config/biomarker_config_default.yaml \
    --data_dir /lfs/turing1/0/mahmedc/Comorbidities-Detection/datasets/full_data \
    --dry_run \
    --check_memory

# Run specific model
python run_experiments.py \
    --config_csv experimentation_plan_simplified.csv \
    --biomarker_config config/biomarker_config_default.yaml \
    --data_dir /lfs/turing1/0/mahmedc/Comorbidities-Detection/datasets/full_data \
    --model_name "ResNet-18" \
    --epochs 50

# Enable learning rate hyperparameter search
python run_experiments.py \
    --config_csv experimentation_plan_simplified.csv \
    --biomarker_config config/biomarker_config_default.yaml \
    --data_dir /lfs/turing1/0/mahmedc/Comorbidities-Detection/datasets/full_data \
    --model_name "ResNet-18" \
    --enable_lr_search \
    --epochs 50

# Run all must-include experiments with LR search (expands to ~51 experiments)
python run_experiments.py \
    --config_csv experimentation_plan_simplified.csv \
    --biomarker_config config/biomarker_config_default.yaml \
    --data_dir /lfs/turing1/0/mahmedc/Comorbidities-Detection/datasets/full_data \
    --must_include_only \
    --enable_lr_search \
    --epochs 100

# Run only Turing1-compatible experiments (21 experiments, fits RTX 2080 Ti 11GB)
python run_experiments.py \
    --config_csv experimentation_plan_simplified.csv \
    --biomarker_config config/biomarker_config_default.yaml \
    --data_dir /lfs/turing1/0/mahmedc/Comorbidities-Detection/datasets/full_data \
    --must_include_only \
    --turing1_only \
    --enable_lr_search \
    --epochs 100
```

## 📊 Flexible Biomarker Configuration

### Current Setup (biomarker_config_default.yaml)

```yaml
experiment_name: "comorbidities_detection"
description: "Multi-task learning for comorbidity detection from CT scans"

binary_biomarkers:
  - GENDER (male=1, female=0)
  - MORTALITY (PRESENT/ABSENT)
  - HCC12, HCC18, HCC19, HCC22, HCC48, HCC85, HCC96, HCC108, HCC111 (PRESENT/ABSENT)
  - CALCIUMSCORING_ABDOMINALAGATSTON_BINARY (PRESENT/ABSENT)

continuous_biomarkers:
  - AGE (18-102 years, normalized to [0,1])

multiclass_biomarkers: []
```

### Tensor Layout (Automatically Generated)

```
Index  Biomarker                                Type
0      GENDER                                   Binary
1      MORTALITY                                Binary  
2      HCC12                                    Binary
3      HCC18                                    Binary
4      HCC19                                    Binary
5      HCC22                                    Binary
6      HCC48                                    Binary
7      HCC85                                    Binary
8      HCC96                                    Binary
9      HCC108                                   Binary
10     HCC111                                   Binary
11     CALCIUMSCORING_ABDOMINALAGATSTON_BINARY  Binary
12     AGE                                      Continuous
```

## 🔄 Migration from Hardcoded System

### Old Way (Hardcoded)
```python
# Fixed assumptions in train.py
NUM_BINARY_TASKS = 7
NUM_REGRESSION_TASKS = 2  
CALCIUM_CLASSES = 4
CONDITIONS = ['GENDER', 'HCC18', ...]  # Fixed list

# Fixed multi-task head
model.fc = MultiTaskHead(feature_dim, num_binary_tasks=7, ...)
```

### New Way (Flexible)
```python
# Dynamic configuration
from config.biomarker_config import FlexibleBiomarkerConfig
biomarker_config = FlexibleBiomarkerConfig('config/biomarker_config_default.yaml')

# Adaptive multi-task head
model.fc = FlexibleMultiTaskHead(feature_dim, biomarker_config)

# Flexible loss and metrics
criterion = FlexibleMultiTaskLoss(biomarker_config)
metrics_calc = FlexibleMetricsCalculator(biomarker_config)
```

## 📊 Experiment Configuration

The system uses `experimentation_plan_simplified.csv` to configure experiments. Key parameters:

- **Model**: Architecture name (must match ModelFactory names)
- **Must Include**: Whether to include in batch runs
- **Learning Rate**: Single value or list for hyperparameter search (e.g., `"[1e-5, 1e-4, 1e-3]"`)
- **Batch Size**: Training batch size
- **Optimizer**: AdamW, Adam, or SGD
- **Scheduler**: CosineAnnealing, ReduceLROnPlateau, StepLR, ExponentialLR, etc.
- **Expected_GPU_Memory**: For memory checking
- **Class_Weighting**: inverse_frequency for balanced training
- **Sampling_Strategy**: balanced_batch for balanced sampling

### 🔍 **Hyperparameter Search**

The system supports **automatic learning rate hyperparameter search**:

- **CSV Format**: Specify multiple learning rates as `"[1e-5, 1e-4, 1e-3]"`
- **Automatic Expansion**: Each base experiment becomes multiple experiments (one per LR)
- **Directory Naming**: Each experiment gets a unique directory with LR in the name
  - Example: `ResNet-18_lr1e-05_bs16_20250913_113026/`
  - Example: `ResNet-18_lr1e-04_bs16_20250913_113026/`
- **Enable with Flag**: Use `--enable_lr_search` to activate hyperparameter search
- **Scaling**: Must-include experiments expand from ~15 to ~45 total experiments

### 🖥️ **GPU Compatibility (Turing1)**

The system includes **automatic GPU compatibility filtering** for RTX 2080 Ti GPUs (11GB VRAM):

- **Hardware Analysis**: 10x RTX 2080 Ti GPUs, ~10.5GB usable per GPU
- **Turing1 Column**: Added to CSV to mark compatible experiments
- **Smart Filtering**: `--turing1_only` flag filters out memory-intensive models
- **Optimization**: Reduces 51 experiments to 21 compatible experiments

**Compatible Models** (7 architectures × 3 learning rates = 21 experiments):
- ✅ **ResNet-18** (4-6GB) - Lightweight CNN
- ✅ **ResNet-34** (6-8GB) - Medium CNN  
- ✅ **DenseNet-121** (8-10GB) - Dense connections
- ✅ **EfficientNet-B0** (6-8GB) - Efficient scaling
- ✅ **ViT-Small (DINOv2)** (8-10GB) - Self-supervised ViT
- ✅ **Stable Diffusion VAE Encoder (frozen)** (8-10GB) - Generative features
- ✅ **ResNet-50 (RadImageNet)** (6-8GB) - Medical pre-training

**Filtered Out** (too big for 11GB):
- ❌ EfficientNet-B4, ConvNeXt-Base, ViT-Base/Large, Swin Transformer
- ❌ MaxViT, Full Diffusion models, MAE ViT-Base

## 📈 Monitoring and Results

### TensorBoard Logging

```bash
# View training progress
tensorboard --logdir /lfs/turing1/0/mahmedc/Comorbidities-Detection/models
```

Logged metrics include:
- Training/validation loss (total and per-task)
- AUROC per biomarker
- Average AUROC for model selection
- Learning rate schedules
- F1 scores per biomarker

### Output Structure

Each experiment creates:
```
models/
└── {experiment_name}/
    ├── tensorboard/           # TensorBoard logs
    ├── best_checkpoint.pth    # Best model (highest avg AUROC)
    ├── latest_checkpoint.pth  # Most recent model
    ├── config.json           # Experiment configuration
    └── biomarker_config.json # Biomarker configuration used
```

## 🎯 Multi-Task Learning Details

### Flexible Loss Function
Combined loss automatically adapts based on configuration:
- **Binary Tasks**: Weighted BCE with inverse frequency weighting
- **Multiclass Tasks**: Cross-entropy loss
- **Regression Tasks**: MSE loss

### Model Selection
Best model selected based on **average AUROC** across all binary classification and multiclass tasks.

### Comprehensive Metrics
- **Binary Tasks**: AUROC, Accuracy, Sensitivity, Specificity, F1-score
- **Multiclass Tasks**: AUROC (macro), Accuracy, Per-class metrics
- **Regression Tasks**: MSE, MAE, R² score

## ⚡ Key Benefits

### 1. **No More Hardcoding**
- Change tasks by editing YAML file only
- No code changes needed for different experiments
- Easy to add/remove biomarkers

### 2. **Automatic Validation**
- Dataset compatibility checking
- Missing column detection
- Tensor layout validation

### 3. **Future-Proof Design**
- Easy to add new task types
- Supports any combination of tasks
- Backward compatible with existing code

## 🔧 Customization

### Adding New Biomarkers

Simply edit the YAML configuration file:

```yaml
# Add new binary biomarker
binary_biomarkers:
  - name: "NEW_CONDITION"
    description: "New medical condition"
    positive_class: "PRESENT"

# Add new multiclass biomarker  
multiclass_biomarkers:
  - name: "SEVERITY_SCORE"
    description: "Disease severity"
    classes: ["MILD", "MODERATE", "SEVERE"]

# Add new continuous biomarker
continuous_biomarkers:
  - name: "BIOMARKER_VALUE"
    description: "Continuous biomarker"
    min_value: 0.0
    max_value: 100.0
    normalization: "min_max"
```

### Adding New Models

1. Add model creation method to `ModelFactory` in `model/model_factory.py`
2. Add entry to CSV configuration file
3. Update memory requirements in `get_model_memory_requirement()`

## 🚦 Getting Started

### 1. Verify Data Compatibility

```bash
python -c "
from config.biomarker_config import FlexibleBiomarkerConfig
import pandas as pd

   config = FlexibleBiomarkerConfig('config/biomarker_config_default.yaml')
df = pd.read_csv('/lfs/turing1/0/mahmedc/Comorbidities-Detection/datasets/full_data/train.csv')
compatible, missing = config.validate_dataset_compatibility(df)
print(f'Compatible: {compatible}')
if missing: print(f'Missing: {missing}')
"
```

### 2. Run Test Experiment

```bash
python run_experiments.py \
  --model_name "ResNet-18" \
  --biomarker_config config/biomarker_config_default.yaml \
  --data_dir /lfs/turing1/0/mahmedc/Comorbidities-Detection/datasets/full_data \
  --epochs 5 \
  --dry_run
```

### 3. Run Full Experiments

```bash
python run_experiments.py \
  --biomarker_config config/biomarker_config_default.yaml \
  --data_dir /lfs/turing1/0/mahmedc/Comorbidities-Detection/datasets/full_data \
  --epochs 100 \
  --must_include_only \
  --check_memory
```

## 📋 Experiment Tracking

The system automatically tracks:
- All hyperparameters and configurations
- Training/validation metrics per epoch
- Best model checkpoints
- Experiment success/failure status
- GPU memory usage and assignments
- Biomarker configuration used

Results are saved in:
- Individual experiment directories
- `experiment_results.csv` summary file
- TensorBoard logs for visualization

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Use `--check_memory` flag to verify GPU requirements
   - Reduce batch size in CSV configuration
   - Use models with lower memory requirements

2. **Missing Dependencies**
   - Run `pip install -r requirements_enhanced.txt`
   - For TIMM models: `pip install timm>=0.9.0`

3. **Import Errors**
   - Run `python test_setup.py` to verify setup
   - Check that all paths are correct

4. **Data Loading Issues**
   - Verify CSV files have correct column names
   - Check that image files exist in `data/` directory
   - Ensure normalization values match your data

5. **Biomarker Configuration Issues**
   - Use the validation function to check dataset compatibility
   - Ensure all required biomarker columns exist in your data
   - Check YAML syntax is correct

### Performance Tips

1. **GPU Memory Optimization**
   - Use gradient checkpointing for large models
   - Enable mixed precision training
   - Use smaller batch sizes for memory-intensive models

2. **Training Speed**
   - Use multiple GPUs with DataParallel
   - Increase `num_workers` in DataLoader
   - Use SSD storage for faster I/O

## 📈 Expected Results

With your 12 binary + 1 regression task configuration:
- **Model Output Size**: 13 
- **Loss Components**: 12 binary BCE losses + 1 MSE loss
- **Metrics**: AUROC/Accuracy/F1 for each binary task + MSE/MAE/R² for age
- **Model Selection**: Based on average AUROC across binary tasks

## 🔮 Future Enhancements

- **Full CLIP Integration**: Proper vision-language model support
- **Medical VLM Support**: Integration with MedCLIP, BiomedCLIP
- **Hyperparameter Optimization**: Automated hyperparameter tuning
- **Distributed Training**: Multi-GPU and multi-node support
- **Advanced Metrics**: ROC curves, confusion matrices, per-class analysis
- **Model Interpretability**: Attention visualization, GradCAM
- **Additional Task Types**: Support for ordinal regression, multi-label classification

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Run `python test_setup.py` to verify setup
3. Review TensorBoard logs for training issues
4. Check experiment output directories for detailed logs
5. Validate your biomarker configuration with the built-in validation tools

Questions: ayis@ayis.org

## 📄 License

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License

By exercising the Licensed Rights (defined below), You accept and agree to be bound by the terms and conditions of this Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License ("Public License"). To the extent this Public License may be interpreted as a contract, You are granted the Licensed Rights in consideration of Your acceptance of these terms and conditions, and the Licensor grants You such rights in consideration of benefits the Licensor receives from making the Licensed Material available under these terms and conditions.

Section 1 – Definitions.

Adapted Material means material subject to Copyright and Similar Rights that is derived from or based upon the Licensed Material and in which the Licensed Material is translated, altered, arranged, transformed, or otherwise modified in a manner requiring permission under the Copyright and Similar Rights held by the Licensor. For purposes of this Public License, where the Licensed Material is a musical work, performance, or sound recording, Adapted Material is always produced where the Licensed Material is synched in timed relation with a moving image.
Adapter's License means the license You apply to Your Copyright and Similar Rights in Your contributions to Adapted Material in accordance with the terms and conditions of this Public License.
BY-NC-SA Compatible License means a license listed at creativecommons.org/compatiblelicenses, approved by Creative Commons as essentially the equivalent of this Public License.
Copyright and Similar Rights means copyright and/or similar rights closely related to copyright including, without limitation, performance, broadcast, sound recording, and Sui Generis Database Rights, without regard to how the rights are labeled or categorized. For purposes of this Public License, the rights specified in Section 2(b)(1)-(2) are not Copyright and Similar Rights.
Effective Technological Measures means those measures that, in the absence of proper authority, may not be circumvented under laws fulfilling obligations under Article 11 of the WIPO Copyright Treaty adopted on December 20, 1996, and/or similar international agreements.
Exceptions and Limitations means fair use, fair dealing, and/or any other exception or limitation to Copyright and Similar Rights that applies to Your use of the Licensed Material.
License Elements means the license attributes listed in the name of a Creative Commons Public License. The License Elements of this Public License are Attribution, NonCommercial, and ShareAlike.
Licensed Material means the artistic or literary work, database, or other material to which the Licensor applied this Public License.
Licensed Rights means the rights granted to You subject to the terms and conditions of this Public License, which are limited to all Copyright and Similar Rights that apply to Your use of the Licensed Material and that the Licensor has authority to license.
Licensor means the individual(s) or entity(ies) granting rights under this Public License.
NonCommercial means not primarily intended for or directed towards commercial advantage or monetary compensation. For purposes of this Public License, the exchange of the Licensed Material for other material subject to Copyright and Similar Rights by digital file-sharing or similar means is NonCommercial provided there is no payment of monetary compensation in connection with the exchange.
Share means to provide material to the public by any means or process that requires permission under the Licensed Rights, such as reproduction, public display, public performance, distribution, dissemination, communication, or importation, and to make material available to the public including in ways that members of the public may access the material from a place and at a time individually chosen by them.
Sui Generis Database Rights means rights other than copyright resulting from Directive 96/9/EC of the European Parliament and of the Council of 11 March 1996 on the legal protection of databases, as amended and/or succeeded, as well as other essentially equivalent rights anywhere in the world.
You means the individual or entity exercising the Licensed Rights under this Public License. Your has a corresponding meaning.
Section 2 – Scope.

License grant.
Subject to the terms and conditions of this Public License, the Licensor hereby grants You a worldwide, royalty-free, non-sublicensable, non-exclusive, irrevocable license to exercise the Licensed Rights in the Licensed Material to:
reproduce and Share the Licensed Material, in whole or in part, for NonCommercial purposes only; and
produce, reproduce, and Share Adapted Material for NonCommercial purposes only.
Exceptions and Limitations. For the avoidance of doubt, where Exceptions and Limitations apply to Your use, this Public License does not apply, and You do not need to comply with its terms and conditions.
Term. The term of this Public License is specified in Section 6(a).
Media and formats; technical modifications allowed. The Licensor authorizes You to exercise the Licensed Rights in all media and formats whether now known or hereafter created, and to make technical modifications necessary to do so. The Licensor waives and/or agrees not to assert any right or authority to forbid You from making technical modifications necessary to exercise the Licensed Rights, including technical modifications necessary to circumvent Effective Technological Measures. For purposes of this Public License, simply making modifications authorized by this Section 2(a)(4) never produces Adapted Material.
Downstream recipients.
Offer from the Licensor – Licensed Material. Every recipient of the Licensed Material automatically receives an offer from the Licensor to exercise the Licensed Rights under the terms and conditions of this Public License.
Additional offer from the Licensor – Adapted Material. Every recipient of Adapted Material from You automatically receives an offer from the Licensor to exercise the Licensed Rights in the Adapted Material under the conditions of the Adapter's License You apply.
No downstream restrictions. You may not offer or impose any additional or different terms or conditions on, or apply any Effective Technological Measures to, the Licensed Material if doing so restricts exercise of the Licensed Rights by any recipient of the Licensed Material.
No endorsement. Nothing in this Public License constitutes or may be construed as permission to assert or imply that You are, or that Your use of the Licensed Material is, connected with, or sponsored, endorsed, or granted official status by, the Licensor or others designated to receive attribution as provided in Section 3(a)(1)(A)(i).
Other rights.

Moral rights, such as the right of integrity, are not licensed under this Public License, nor are publicity, privacy, and/or other similar personality rights; however, to the extent possible, the Licensor waives and/or agrees not to assert any such rights held by the Licensor to the limited extent necessary to allow You to exercise the Licensed Rights, but not otherwise.
Patent and trademark rights are not licensed under this Public License.
To the extent possible, the Licensor waives any right to collect royalties from You for the exercise of the Licensed Rights, whether directly or through a collecting society under any voluntary or waivable statutory or compulsory licensing scheme. In all other cases the Licensor expressly reserves any right to collect such royalties, including when the Licensed Material is used other than for NonCommercial purposes.
Section 3 – License Conditions.

Your exercise of the Licensed Rights is expressly made subject to the following conditions.

Attribution.

If You Share the Licensed Material (including in modified form), You must:

retain the following if it is supplied by the Licensor with the Licensed Material:
identification of the creator(s) of the Licensed Material and any others designated to receive attribution, in any reasonable manner requested by the Licensor (including by pseudonym if designated);
a copyright notice;
a notice that refers to this Public License;
a notice that refers to the disclaimer of warranties;
a URI or hyperlink to the Licensed Material to the extent reasonably practicable;
indicate if You modified the Licensed Material and retain an indication of any previous modifications; and
indicate the Licensed Material is licensed under this Public License, and include the text of, or the URI or hyperlink to, this Public License.
You may satisfy the conditions in Section 3(a)(1) in any reasonable manner based on the medium, means, and context in which You Share the Licensed Material. For example, it may be reasonable to satisfy the conditions by providing a URI or hyperlink to a resource that includes the required information.
If requested by the Licensor, You must remove any of the information required by Section 3(a)(1)(A) to the extent reasonably practicable.
ShareAlike.
In addition to the conditions in Section 3(a), if You Share Adapted Material You produce, the following conditions also apply.

The Adapter's License You apply must be a Creative Commons license with the same License Elements, this version or later, or a BY-NC-SA Compatible License.
You must include the text of, or the URI or hyperlink to, the Adapter's License You apply. You may satisfy this condition in any reasonable manner based on the medium, means, and context in which You Share Adapted Material.
You may not offer or impose any additional or different terms or conditions on, or apply any Effective Technological Measures to, Adapted Material that restrict exercise of the rights granted under the Adapter's License You apply.
Section 4 – Sui Generis Database Rights.

Where the Licensed Rights include Sui Generis Database Rights that apply to Your use of the Licensed Material:

for the avoidance of doubt, Section 2(a)(1) grants You the right to extract, reuse, reproduce, and Share all or a substantial portion of the contents of the database for NonCommercial purposes only;
if You include all or a substantial portion of the database contents in a database in which You have Sui Generis Database Rights, then the database in which You have Sui Generis Database Rights (but not its individual contents) is Adapted Material, including for purposes of Section 3(b); and
You must comply with the conditions in Section 3(a) if You Share all or a substantial portion of the contents of the database.
For the avoidance of doubt, this Section 4 supplements and does not replace Your obligations under this Public License where the Licensed Rights include other Copyright and Similar Rights.
Section 5 – Disclaimer of Warranties and Limitation of Liability.

Unless otherwise separately undertaken by the Licensor, to the extent possible, the Licensor offers the Licensed Material as-is and as-available, and makes no representations or warranties of any kind concerning the Licensed Material, whether express, implied, statutory, or other. This includes, without limitation, warranties of title, merchantability, fitness for a particular purpose, non-infringement, absence of latent or other defects, accuracy, or the presence or absence of errors, whether or not known or discoverable. Where disclaimers of warranties are not allowed in full or in part, this disclaimer may not apply to You.
To the extent possible, in no event will the Licensor be liable to You on any legal theory (including, without limitation, negligence) or otherwise for any direct, special, indirect, incidental, consequential, punitive, exemplary, or other losses, costs, expenses, or damages arising out of this Public License or use of the Licensed Material, even if the Licensor has been advised of the possibility of such losses, costs, expenses, or damages. Where a limitation of liability is not allowed in full or in part, this limitation may not apply to You.
The disclaimer of warranties and limitation of liability provided above shall be interpreted in a manner that, to the extent possible, most closely approximates an absolute disclaimer and waiver of all liability.
Section 6 – Term and Termination.

This Public License applies for the term of the Copyright and Similar Rights licensed here. However, if You fail to comply with this Public License, then Your rights under this Public License terminate automatically.
Where Your right to use the Licensed Material has terminated under Section 6(a), it reinstates:

automatically as of the date the violation is cured, provided it is cured within 30 days of Your discovery of the violation; or
upon express reinstatement by the Licensor.
For the avoidance of doubt, this Section 6(b) does not affect any right the Licensor may have to seek remedies for Your violations of this Public License.
For the avoidance of doubt, the Licensor may also offer the Licensed Material under separate terms or conditions or stop distributing the Licensed Material at any time; however, doing so will not terminate this Public License.
Sections 1, 5, 6, 7, and 8 survive termination of this Public License.
Section 7 – Other Terms and Conditions.

The Licensor shall not be bound by any additional or different terms or conditions communicated by You unless expressly agreed.
Any arrangements, understandings, or agreements regarding the Licensed Material not stated herein are separate from and independent of the terms and conditions of this Public License.
Section 8 – Interpretation.

For the avoidance of doubt, this Public License does not, and shall not be interpreted to, reduce, limit, restrict, or impose conditions on any use of the Licensed Material that could lawfully be made without permission under this Public License.
To the extent possible, if any provision of this Public License is deemed unenforceable, it shall be automatically reformed to the minimum extent necessary to make it enforceable. If the provision cannot be reformed, it shall be severed from this Public License without affecting the enforceability of the remaining terms and conditions.
No term or condition of this Public License will be waived and no failure to comply consented to unless expressly agreed to by the Licensor.
Nothing in this Public License constitutes or may be interpreted as a limitation upon, or waiver of, any privileges and immunities that apply to the Licensor or You, including from the legal processes of any jurisdiction or authority.