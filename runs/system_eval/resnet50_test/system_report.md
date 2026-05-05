# YOLO + CNN System Evaluation

- Split: `test`
- YOLO checkpoint: `runs\detect\v8m_768_adamw_aug\weights\best.pt`
- CNN checkpoint: `.\runs\stage2\resnet50\best.pt`

## Detection

- mAP50: 0.9920
- Precision: 0.9888
- Recall: 0.9930
- Precision @ current conf: 0.9787
- Recall @ current conf: 0.7660
- TP / FP / FN @ current conf: 1653 / 36 / 505
- Localization-only TP / FP / FN: 1653 / 36 / 505

## Classification

- Accuracy on detected boxes: 0.9661
- Macro F1 on detected boxes: 0.9662
- Number of detected boxes evaluated by CNN: 1653

## System

- Overall end-to-end accuracy: 0.7400
- Correct end-to-end predictions: 1597/2158
- Average latency: 37.54 ms/image
- Pipeline errors: {"false_positive_detection": 36, "misclassification": 56, "missed_detection": 505}

## Classification Report

```text
                 precision    recall  f1-score   support

     mouse_bite     0.9921    0.9618    0.9767       262
           spur     0.9231    0.9928    0.9567       278
   missing_hole     0.9930    1.0000    0.9965       283
          short     0.9963    0.9963    0.9963       273
   open_circuit     0.9129    0.9962    0.9527       263
spurious_copper     0.9882    0.8571    0.9180       294

       accuracy                         0.9661      1653
      macro avg     0.9676    0.9674    0.9662      1653
   weighted avg     0.9681    0.9661    0.9657      1653

```
