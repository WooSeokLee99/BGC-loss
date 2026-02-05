Boundary Gradient Consistency Loss

Paper : Improvement of a Segmentation Network for Character Stroke Extraction 
        from Metal Movable Type Printed Documents
DOI   : https://doi.org/10.5573/ieie.2023.60.12.31

This loss function is designed to encourage smooth and consistent boundaries of the foreground
 (especially text regions) in segmentation tasks.
Since it only takes logits (predictions) as input, it should be used together with
 other loss functions during training.

In our experiments, the best performance was achieved when this loss was combined with
 Dice coefficient loss on old printed book datasets scanned at 600 dpi.

When the input resolution is too high, this loss tends to over-focus on details.
To mitigate this, max pooling is applied before computing the boundary gradients.
Therefore, the pooling scale should be adjusted according to the resolution of the input images.

Although average pooling or Gaussian blur can also be used, additional preprocessing steps
 (e.g., quantization or contour extraction) are required to compute boundary gradients properly.
