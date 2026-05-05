# YOLO + CNN System Evaluation

- Split: `test`
- YOLO checkpoint: `runs\detect\v8m_768_adamw_aug\weights\best.pt`
- CNN checkpoint: `.\runs\stage2\efficientnet_b2\best.pt`

## Detection

- mAP50: 0.9920
- Precision: 0.9888
- Recall: 0.9930
- Precision @ current conf: 0.9787
- Recall @ current conf: 0.7660
- TP / FP / FN @ current conf: 1653 / 36 / 505
- Localization-only TP / FP / FN: 1653 / 36 / 505

## Classification

- Accuracy on detected boxes: 0.9879
- Macro F1 on detected boxes: 0.9880
- Number of detected boxes evaluated by CNN: 1653

## System

- Overall end-to-end accuracy: 0.7567
- Correct end-to-end predictions: 1633/2158
- Average latency: 51.38 ms/image
- Pipeline errors: {"false_positive_detection": 36, "misclassification": 20, "missed_detection": 505}

## Classification Report

```text
                 precision    recall  f1-score   support

     mouse_bite     0.9738    0.9924    0.9830       262
           spur     0.9717    0.9892    0.9804       278
   missing_hole     1.0000    1.0000    1.0000       283
          short     1.0000    0.9927    0.9963       273
   open_circuit     1.0000    0.9848    0.9923       263
spurious_copper     0.9828    0.9694    0.9760       294

       accuracy                         0.9879      1653
      macro avg     0.9880    0.9881    0.9880      1653
   weighted avg     0.9880    0.9879    0.9879      1653

```
