    # Error Analysis Report — efficientnet_b2

    ## Overview

    | Metric | Value |
    | --- | --- |
    | Total GT boxes | 2158 |
    | Total predictions | 1689 |
    | System accuracy | 0.7567 |
    | Missed detections | 505 |
    | False positives | 36 |
    | Misclassifications | 20 |

    ## Errors by Defect Class

    | Class | GT Count | Missed | Miss Rate | False Positive | Misclassified | Total Errors |
    | --- | --- | --- | --- | --- | --- | --- |
    | missing_hole | 379 | 96 | 25.3% | 6 | 0 | 102 |
| mouse_bite | 332 | 70 | 21.1% | 6 | 2 | 78 |
| open_circuit | 345 | 82 | 23.8% | 5 | 4 | 91 |
| short | 366 | 93 | 25.4% | 5 | 2 | 100 |
| spur | 348 | 70 | 20.1% | 6 | 3 | 79 |
| spurious_copper | 388 | 94 | 24.2% | 8 | 9 | 111 |

    ## Misclassification Pairs (GT -> Predicted)

    | True Label | Predicted Label | Count |
    | --- | --- | --- |
    | spurious_copper | spur | 8 |
| open_circuit | mouse_bite | 4 |
| spur | spurious_copper | 3 |
| short | mouse_bite | 2 |
| mouse_bite | spurious_copper | 2 |
| spurious_copper | mouse_bite | 1 |

    ## Key Findings

    1. **Bottleneck**: Stage 1 (YOLO) missed 505 out of 2158 GT boxes (23.4%), which is the primary factor limiting system accuracy.
    2. **Stage 2 performs well**: Only 20 misclassifications out of 1653 detected boxes (98.79% accuracy).
    3. **Most confused pair**: spurious_copper -> spur (8 times)

    ## Generated Visualizations

    - `error_by_class.png` — Stacked bar: errors by class and type
    - `error_type_pie.png` — Pie chart of error distribution
    - `miss_rate_by_class.png` — YOLO miss rate per class
    - `misclassification_heatmap.png` — GT vs Predicted confusion
    - `gt_distribution.png` — GT class distribution
    - `examples_*.png` — Sample images for each error type
