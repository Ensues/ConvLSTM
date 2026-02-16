# Prototype 3 - ConvLSTM Without Batch Normalization

## Overview
Prototype 3 is a controlled variant of Prototype 2, created specifically to investigate the impact of Batch Normalization on model performance. This version includes **7 out of 8** features from Prototype 2, with **Batch Normalization intentionally removed** for comparative analysis.

---

## Purpose

During testing of Prototype 2 (which included Batch Normalization), **unexpected accuracy degradation was observed**. To isolate whether Batch Normalization was the cause of this performance decrease, Prototype 3 was created as an ablation study variant.

### Research Question
**Does Batch Normalization improve or harm ConvLSTM performance for this specific Motor Vehicle Orientation (MVO) prediction task?**

---

## Key Difference from Prototype 2

### ❌ **REMOVED: Batch Normalization**

**In Prototype 2 (WITH Batch Normalization):**
```python
class ConvLSTM(nn.Module):
    def __init__(self, ...):
        # ...
        cell_list = []
        batch_norm_list = []  # Created batch norm layers
        for i in range(0, self.num_layers):
            cell_list.append(ConvLSTMCell(...))
            batch_norm_list.append(nn.BatchNorm2d(self.hidden_dim[i]))  # Added BN
        
        self.cell_list = nn.ModuleList(cell_list)
        self.batch_norm_list = nn.ModuleList(batch_norm_list)  # Stored BN layers
    
    def forward(self, input_tensor, hidden_state=None):
        # ...
        layer_output = torch.stack(output_inner, dim=1)
        
        # Reshape and apply Batch Normalization
        b_size, t_size, c_size, h_size, w_size = layer_output.size()
        layer_output = layer_output.view(b_size * t_size, c_size, h_size, w_size)
        layer_output = self.batch_norm_list[layer_idx](layer_output)  # Applied BN
        layer_output = layer_output.view(b_size, t_size, c_size, h_size, w_size)
        # ...
```

**In Prototype 3 (WITHOUT Batch Normalization):**
```python
class ConvLSTM(nn.Module):
    def __init__(self, ...):
        # ...
        cell_list = []
        for i in range(0, self.num_layers):
            cell_list.append(ConvLSTMCell(...))
        
        self.cell_list = nn.ModuleList(cell_list)
        # No batch normalization layers created
    
    def forward(self, input_tensor, hidden_state=None):
        # ...
        layer_output = torch.stack(output_inner, dim=1)
        # Directly pass through without normalization
        cur_layer_input = layer_output
        # ...
```

---

## Observations & Current Status

### Why This Version Exists

When Prototype 2 was tested with all 8 features (including Batch Normalization), the model showed **lower accuracy than expected**. The exact cause is still under investigation, but Batch Normalization is a prime suspect because:

1. **Temporal Data Sensitivity:** Batch Normalization normalizes across the batch dimension, which can disrupt temporal correlations in sequential data processed by ConvLSTMs.

2. **Small Batch Size:** With effective batch size of 20 (5 physical × 4 accumulation steps), batch statistics may be noisy and unreliable.

3. **Video Data Characteristics:** The spatial and temporal structure of video data may not benefit from normalization in the same way as image classification tasks.

4. **Training Instability:** BN introduces additional noise through running mean/variance updates, which could conflict with gradient accumulation.

### Research Hypothesis

**Hypothesis:** Removing Batch Normalization will restore or improve model accuracy compared to Prototype 2.

**Testing Plan:**
- Train Prototype 3 with identical hyperparameters as Prototype 2
- Compare validation and test accuracy between both versions
- Analyze training curves and convergence behavior
- Determine if BN helps, harms, or has neutral effect

---

## Experimental Setup

To ensure fair comparison between Prototype 2 and Prototype 3:

### Identical Configuration
```python
BATCH = 5
NUM_EPOCHS = 1  # Adjust as needed for full training
LEARNING_RATE = 1e-4
ACCUMULATION_STEPS = 4
EARLY_STOP_PATIENCE = 2
MIN_DELTA = 0.01
SEED = 8  # Same random seed for reproducible data splits

PARAMS = {
    'input_dim': 3,
    'hidden_dim': [64, 32],
    'kernel_size': (3, 3),
    'num_layers': 2,
    'height': 128,
    'width': 128,
    'num_classes': 3
}
```

### Controlled Variables
- ✅ Same dataset and train/val/test split (SEED=8)
- ✅ Same optimizer (Adam)
- ✅ Same learning rate schedule (ReduceLROnPlateau)
- ✅ Same model architecture (ConvLSTM with [64, 32] hidden dims)
- ✅ Same regularization (dropout=0.5)
- ✅ Same gradient handling (clipping + accumulation)

### Independent Variable
- **ONLY DIFFERENCE:** Presence/absence of Batch Normalization layers in ConvLSTM

---

## Expected Outcomes

### Scenario 1: Batch Normalization Was Harmful
**If Prototype 3 accuracy > Prototype 2 accuracy:**
- ✅ Confirms BN degrades performance for this task
- ✅ Use Prototype 3 as the production model
- 📝 Document why BN is unsuitable for ConvLSTM on video data

### Scenario 2: Batch Normalization Was Beneficial
**If Prototype 2 accuracy > Prototype 3 accuracy:**
- ❌ BN removal made things worse
- 🔍 Look for other causes of Prototype 2's poor performance
- 📝 Investigate BN implementation or hyperparameter tuning

### Scenario 3: No Significant Difference
**If accuracies are similar (within 1-2%):**
- ↔️ BN has minimal impact on this specific task
- ⚡ Consider keeping Prototype 3 (simpler, fewer parameters)
- 📝 Document null result for future reference

---

## Performance Comparison Template

| Metric | Prototype 2 (With BN) | Prototype 3 (Without BN) | Difference |
|--------|----------------------|-------------------------|------------|
| **Train Accuracy** | __%  | __%  | __% |
| **Validation Accuracy** | __%  | __%  | __% |
| **Test Accuracy** | __%  | __%  | __% |
| **Training Time** | __ epochs | __ epochs | __ |
| **Avg Inference Latency** | __ ms | __ ms | __ ms |
| **Model Parameters** | __ M | __ M | __ |
| **Convergence Speed** | Epoch __ | Epoch __ | __ |

*(To be filled after training both prototypes)*

---

## Implementation Notes

### Code Changes Required

**Minimal modifications** - only the `ConvLSTM` class differs:

1. **Removed in `__init__`:**
   - `batch_norm_list = []` list initialization
   - `batch_norm_list.append(nn.BatchNorm2d(...))` creation
   - `self.batch_norm_list = nn.ModuleList(batch_norm_list)` storage

2. **Removed in `forward`:**
   - Reshaping from `[B, T, C, H, W]` to `[B*T, C, H, W]`
   - `self.batch_norm_list[layer_idx](layer_output)` application
   - Reshaping back from `[B*T, C, H, W]` to `[B, T, C, H, W]`

All other code remains **identical** to Prototype 2.

---

## Why I'm Not Sure What Happened

Batch Normalization is typically beneficial for deep learning, but in this case it unexpectedly lowered accuracy. Several possible explanations:

### Potential Causes (To Be Investigated)

1. **Temporal Dynamics:** BN may be disrupting the recurrent temporal patterns that ConvLSTM learns across frames.

2. **Small Batch Statistics:** With small batches, BN's running mean/variance estimates may be too noisy.

3. **Gradient Accumulation Interaction:** Accumulating gradients over multiple steps may conflict with BN's per-batch normalization.

4. **Dataset Characteristics:** The specific distribution of our MVO video data may not benefit from normalization.

5. **Implementation Details:** Applying BN after stacking temporal outputs may introduce unexpected artifacts.

6. **Hyperparameter Sensitivity:** BN momentum or epsilon settings may need tuning for this task.

### Next Steps for Investigation

- ✅ Train Prototype 3 and compare results
- 📊 Visualize activation distributions with/without BN
- 📈 Compare training loss curves
- 🔬 Analyze per-class performance differences
- 📝 Document findings for thesis

---

## Conclusion

Prototype 3 serves as a **controlled experiment** to isolate the effect of Batch Normalization on ConvLSTM performance for MVO prediction. By maintaining all other features and hyperparameters constant, we can definitively determine whether BN helps or harms this specific task.

**This is a common practice in machine learning research** - when a feature unexpectedly degrades performance, create an ablation variant to isolate and study the effect. The results will guide future model development decisions.

---

## Files & Related Documentation

- **Main Notebook:** [prototype_3.ipynb](prototype_3.ipynb)
- **Comparison:** [prototype_2.ipynb](prototype_2.ipynb) (with Batch Normalization)
- **Prototype 2 Documentation:** [PROTOTYPE_2_README.md](PROTOTYPE_2_README.md)
- **Training Results:** _(To be generated after training)_
- **Test Results:** `test_results.csv` _(To be generated after testing)_

---

*This prototype was created for ablation study purposes to investigate unexpected accuracy degradation.*  
*Last Updated: February 16, 2026*
