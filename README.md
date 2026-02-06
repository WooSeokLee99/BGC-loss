## Boundary Gradient Consistency Loss

Paper : Improvement of a Segmentation Network for Character Stroke Extraction from Metal Movable Type Printed Documents<br>
<br>
DOI   : https://doi.org/10.5573/ieie.2023.60.12.31<br>
<br>
This loss function is designed to encourage smooth and consistent boundaries of the foreground (especially text regions) in segmentation tasks. Since it only takes logits (predictions) as input, it should be used together with
 other loss functions during training.<br>
<br>
In our experiments, the best performance was achieved when this loss was combined with Dice coefficient loss on old printed book datasets scanned at 600 dpi.<br>
<br>
When the input resolution is too high, this loss tends to over-focus on details. To mitigate this, max pooling is applied before computing the boundary gradients. Therefore, the pooling scale should be adjusted according to the resolution of the input images.<br>
<br>
Although average pooling or Gaussian blur can also be used, additional preprocessing steps (e.g., quantization or contour extraction) are required to compute boundary gradients properly.<br>


## Dataset

<img src="assets/dataset.png" width="100%" />
(a) Input image<br>
(a1) Outer line<br>
(a2) Saperating line(Vertical)<br>
(b) Combo loss (Dice coefficient loss + Cross entropy loss)<br>
<br>
The dataset we used is scanned Korean old printed book (16-17th century).<br>
Previous model with combo loss cannot erase (a1) and (a2) well.


## Loss function

<img src="assets/equation.png" width="50%" />

<img src="assets/illust.png" width="50%" />
Orange pixel: anchor pixel<br>
<br>
1. Search grey neighbor pixels like N(I,j), and get their gradient.<br>
2. Find position of maximum gradient magnitude with argmax.<br>
3. Compare between anchor gradient and the others.<br>
4. If, anchor pixel gradient is similar to others it means smooth stroke, than loss value goes down.<br>
<br>
This method lead stroke smoother.


## Results by pooling parameter

<img src="assets/resert-param.png" width="50%" />
Results with various loss functions<br>
(a) Input<br>
(b) Dice + BGC(1)<br>
(c) Dice + BGC(2)<br>
(d) Dice + BGC(4)<br>
(e) Dice + BGC(8)<br>
(f) Dice + BGC(16)<br>
<br>
<img src="assets/resert-all.png" width="100%" />
(a) Cross Entropy<br>
(b) Dice<br>
(c) Dice + Total Variation<br>
(d) Dice + Cross Entropy<br>
(e) Cross Entropy + Focal<br>
(f) Dice + Focal<br>
(g) Dice + Gradient Difference<br>
(h) Dice + Cross Entropy + Total Variation<br>
(i) Dice + Cross Entropy + Total Variation + Focal<br>
(j) Dice + BGC(16) (proposed)
