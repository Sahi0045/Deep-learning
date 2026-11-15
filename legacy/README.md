# legacy/

Original CPSC 8810 (Clemson, 2019) artifacts and early-experiment scripts, kept
for provenance. **None of these are used by the current pipeline**, and the TF
1.x code will not run on TensorFlow 2.x.

| File | What it was |
|------|-------------|
| `training.py` | TF 1.x training script (Sessions, `set_random_seed`, `initialize_all_variables`) |
| `reading_data.py` | TF 1.x image loader used by `training.py` |
| `test.py` | TF 1.x single-image inference against `trained-model` |
| `identify_role_bullying.py` | Unrelated text/role-classification experiment |
| `download_remaining.py` | Bing image scraper for the original 10-class "cyberbullying" labels |
| `trained-model`, `trained-model.meta` | TF 1.x graph + checkpoint from the 2019 project |
| `best_model.keras` | An earlier 10-class `CyberbullyingCNN` checkpoint (1.44M params, 10-output) — incompatible with the current 3-class pipeline |
| `README` | The original 2019 project README |

The current project (TF 2.x / Keras 3, three-class AIRTLab violence detection,
four model families) lives at the repository root.
