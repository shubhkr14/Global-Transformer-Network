# GT-NET (Brain Tumor Classification)

## Short summary
This project trains a CNN (DenseNet121 backbone) to classify 4 classes: glioma, meningioma, notumor, pituitary.  
Training used images in `data/train` and validation in `data/val`.

## Files
- `dataloader.py` : loads & preprocesses (224x224, normalize, augment) images
- `model.py` : builds model (DenseNet121 + GAP + Dense + Dropout + Softmax)
- `train.py` : trains model, saves best weights and final .keras model
- `predict.py` : run single-image prediction (rebuilds model + loads weights)
- `predict_all.py` : batch predict on validation set → `predictions.csv`
- `eval_preds.py` : computes confusion matrix, per-class precision/recall/F1
- `visualize_confusion.py` : creates `confusion_matrix.png`
- `show_misclassified.py` : creates `misclassified_grid.png`

## Quick commands (run in project root)
1. Train (example short run): python train.py --epochs 2 --batch 8 --lr 0.0002

2. Single image prediction: python predict.py data/val/meningioma/Tr-me_0010.jpg

3. Predict all and evaluate:

python predict_all.py --val_dir data/val --weights ./checkpoints/gt_net.weights.h5 --out predictions.csv --batch 32
python eval_preds.py
python visualize_confusion.py
python show_misclassified.py

## Key results (validation)
- Overall accuracy: 95.53%
- See `metrics_summary.txt` and `confusion_matrix.png` for details.

## Notes / Next steps
- Improve class balance, add more training images for glioma/meningioma.
- Try fine-tuning more layers of backbone or using ensembling.
