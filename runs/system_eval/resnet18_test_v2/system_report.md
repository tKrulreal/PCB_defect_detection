# YOLO + CNN System Evaluation

- Split: `test`
- YOLO checkpoint: `runs\detect\v8m_768_adamw_aug\weights\best.pt`
- CNN checkpoint: `runs/stage2/resnet18/best.pt`

## Detection

- mAP50: 0.9920
- Precision: 0.9888
- Recall: 0.9930
- Precision @ current conf: 0.9787
- Recall @ current conf: 0.7660
- TP / FP / FN @ current conf: 1653 / 36 / 505
- Localization-only TP / FP / FN: 1653 / 36 / 505

## Classification

- Accuracy on detected boxes: 0.9770
- Macro F1 on detected boxes: 0.9777
- Number of detected boxes evaluated by CNN: 1653

## System

- Overall end-to-end accuracy: 0.7484
- Correct end-to-end predictions: 1615/2158
- Average latency: 32.22 ms/image
- Pipeline errors: {"false_positive_detection": 36, "misclassification": 38, "missed_detection": 505}

## Classification Report

```text
                 precision    recall  f1-score   support

     mouse_bite     0.9962    0.9924    0.9943       262
           spur     0.8932    0.9928    0.9404       278
   missing_hole     1.0000    1.0000    1.0000       283
          short     0.9963    0.9963    0.9963       273
   open_circuit     1.0000    0.9962    0.9981       263
spurious_copper     0.9887    0.8912    0.9374       294

       accuracy                         0.9770      1653
      macro avg     0.9791    0.9781    0.9777      1653
   weighted avg     0.9788    0.9770    0.9770      1653

```
