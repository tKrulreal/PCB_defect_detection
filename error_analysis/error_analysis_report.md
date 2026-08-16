    # Error Analysis Report — resnet18

    ## Overview

    | Metric | Value |
    | --- | --- |
    | Total GT boxes | 2158 |
    | Total predictions | 1689 |
    | System accuracy | 0.7484 |
    | Missed detections | 505 |
    | False positives | 36 |
    | Misclassifications | 38 |

    ## Errors by Defect Class

    | Class | GT Count | Missed | Miss Rate | False Positive | Misclassified | Total Errors |
    | --- | --- | --- | --- | --- | --- | --- |
    | missing_hole | 379 | 96 | 25.3% | 6 | 0 | 102 |
| mouse_bite | 332 | 70 | 21.1% | 5 | 2 | 77 |
| open_circuit | 345 | 82 | 23.8% | 6 | 1 | 89 |
| short | 366 | 93 | 25.4% | 6 | 1 | 100 |
| spur | 348 | 70 | 20.1% | 8 | 2 | 80 |
| spurious_copper | 388 | 94 | 24.2% | 5 | 32 | 131 |

    ## Misclassification Pairs (GT -> Predicted)

    | True Label | Predicted Label | Count |
    | --- | --- | --- |
    | spurious_copper | spur | 32 |
| mouse_bite | spurious_copper | 2 |
| spur | spurious_copper | 1 |
| spur | short | 1 |
| open_circuit | mouse_bite | 1 |
| short | spur | 1 |

    ## Key Findings

    1. **Bottleneck**: Stage 1 (YOLO) missed 505 out of 2158 GT boxes (23.4%), which is the primary factor limiting system accuracy.
    2. **Stage 2 performs well**: Only 38 misclassifications out of 1653 detected boxes (97.70% accuracy).
    3. **Most confused pair**: spurious_copper -> spur (32 times)

    ## Generated Visualizations

    - `error_by_class.png` — Stacked bar: errors by class and type
    - `error_type_pie.png` — Pie chart of error distribution
    - `miss_rate_by_class.png` — YOLO miss rate per class
    - `misclassification_heatmap.png` — GT vs Predicted confusion
    - `gt_distribution.png` — GT class distribution
    - `examples_*.png` — Sample images for each error type
