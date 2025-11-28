# Natural Gradient Tests for LightGBMLSS

This directory contains comprehensive tests for the Natural Gradient functionality in LightGBMLSS with Gaussian distribution.

## Test Coverage

### `test_natural_gradient_gaussian.py`

Tests the natural gradient implementation matching the syntax used in `docs/examples/Gaussian_natural.ipynb`.

#### Test Cases

1. **`test_natural_gradient_initialization`**
   - Verifies Gaussian distribution can be initialized with `natural_gradient=True/False`
   - Ensures flag is properly stored

2. **`test_lightgbmlss_with_natural_gradient`**
   - Tests full training pipeline with natural gradients
   - Verifies the notebook syntax works correctly:
     ```python
     gauss_nat = Gaussian(stabilization="MAD", response_fn="softplus", 
                         loss_fn="nll", natural_gradient=True)
     lgblss_nat = LightGBMLSS(gauss_nat)
     lgblss_nat.start_values = np.array([np.array(0.5) for _ in range(lgblss_nat.dist.n_dist_param)])
     ```
   - Validates predictions have correct structure

3. **`test_natural_vs_standard_gradient_comparison`**
   - Compares natural gradient vs standard gradient training
   - Ensures they produce different results (validates implementation)
   - Calculates NLL for both methods

4. **`test_natural_gradient_convergence`**
   - Verifies training loss decreases over iterations
   - Checks that at least 70% of iterations show improvement
   - Validates convergence behavior

5. **`test_natural_gradient_parameter_estimation`**
   - Tests correlation between predicted and true parameters
   - Validates mean (loc) and std (scale) parameter accuracy
   - Ensures MAE is within reasonable bounds

6. **`test_start_values_setting`**
   - Tests the notebook syntax for setting initial values:
     ```python
     lgblss.start_values = np.array([np.array(0.5) for _ in range(lgblss.dist.n_dist_param)])
     ```
   - Verifies training works with custom start values

7. **`test_dataset_creation_syntax`**
   - Validates LightGBM dataset creation:
     ```python
     dtrain = lgb.Dataset(X_train, label=y_train)
     dvalid = lgb.Dataset(X_test, label=y_test, reference=dtrain)
     ```

8. **`test_prediction_output_format`**
   - Ensures predictions return DataFrame with 'loc' and 'scale' columns
   - Validates DataFrame shape and data types
   - Verifies predictions are finite and valid

## Running the Tests

### Run all natural gradient tests:
```bash
cd /Users/evgenygenov/Yandex.Disk.localized/GitHub/LSSboost_test/LightGBMLSS
pytest tests/test_natural_gradient/ -v -s
```

### Run specific test:
```bash
pytest tests/test_natural_gradient/test_natural_gradient_gaussian.py::TestNaturalGradientGaussian::test_natural_gradient_initialization -v -s
```

### Run with coverage:
```bash
pytest tests/test_natural_gradient/ --cov=lightgbmlss.distributions.Gaussian --cov-report=html
```

## Test Data

Tests use **synthetic heteroscedastic Gaussian data** matching the notebook example:

- **Mean function**: `μ = 2.0 + 0.5*X₀ - 0.3*X₁ + 0.2*X₂ + 0.15*X₃*X₄ + 0.1*sin(X₅)`
- **Std function**: `σ = exp(0.5 + 0.3*|X₆| + 0.2*X₇ + 0.1*X₈²)`
- **Sample**: `y ~ N(μ, σ²)`

This ensures:
1. **Heteroscedasticity**: Variance depends on features
2. **Non-linearity**: Both mean and variance have complex functional forms
3. **Realistic**: Tests challenging scenario for gradient-based methods

## Expected Results

When all tests pass, you should see output like:

```
    Natural gradient initialization test passed
    LightGBMLSS with natural gradient training test passed
    Natural vs Standard gradient comparison test passed
    Natural Gradient NLL: 1.4523
    Standard Gradient NLL: 1.4687
    Mean prediction difference: 0.0234
    Natural gradient convergence test passed
    Initial train loss: 1.8234
    Final train loss: 1.4123
    Improvement rate: 87.3%
    Natural gradient parameter estimation test passed
    Mean correlation: 0.8234, MAE: 0.3456
    Std correlation: 0.6789, MAE: 0.2345
    Start values setting test passed
    Dataset creation syntax test passed
    Prediction output format test passed
```

## Implementation Details

### Natural Gradient Formula

For Gaussian distribution with parameters `[μ, log(σ)]`, the Fisher Information Matrix is:

```
F = [1/σ²    0  ]
    [  0     2  ]
```

Natural gradient: `g_nat = F⁻¹ * g_standard`

This provides:
- **Better conditioning** for gradient updates
- **Invariance** to parameter transformations
- **Potentially faster convergence** for distributional parameters

### Key Differences from Standard Gradient

1. **Gradient Scaling**: Each parameter gradient is scaled by inverse FIM
2. **Parameter-specific**: μ gradient scaled by σ², log(σ) gradient scaled by 0.5
3. **Computational Cost**: Negligible overhead (diagonal FIM → element-wise division)

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the LightGBMLSS directory
2. **Path Issues**: The test uses `sys.path.insert` to find the module
3. **Data Type Errors**: Ensure labels are `np.float64` (matching notebook)
4. **Start Values**: Must be set before training (matching notebook syntax)

### Debug Mode

Run with verbose output to see detailed information:
```bash
pytest tests/test_natural_gradient/test_natural_gradient_gaussian.py -v -s --tb=short
```

## Future Enhancements

Potential additions to the test suite:

1. Tests for other distributions (StudentT, Gamma, etc.)
2. Performance benchmarks (natural vs standard gradient speed)
3. Tests with CRPS loss (should warn about incompatibility)
4. Tests with different stabilization methods
5. Tests with extreme values (numerical stability)
6. Comparison with NGBoost natural gradient implementation

## References

- Notebook: `docs/examples/Gaussian_natural.ipynb`
- Distribution: `lightgbmlss/distributions/Gaussian.py`
- Model: `lightgbmlss/model.py`
- Natural Gradient Theory: [Amari, 1998] "Natural Gradient Works Efficiently in Learning"
