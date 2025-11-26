Challenges Encountered During CNN Training and How They Were Resolved

During the initial training of the CNN model, the loss values for both training and validation remained close to 0.69, and the validation accuracy was stuck around 0.55. The loss curve did not decrease, and the model failed to learn any meaningful patterns from the data. This behavior signaled that the CNN was unable to perform effective gradient updates.

Upon investigation, the main issue was identified as the scale of the input data. The temperature values in the LST patches were in the range of 250–330 K, which is significantly larger than typical image intensity values (usually 0–1 or 0–255). Feeding these unnormalized values into a CNN caused gradient instability, preventing the model from optimizing its parameters properly.

To address this, we applied the following preprocessing steps:

NaN Handling – Patches with missing values were cleaned by replacing NaNs with the mean temperature of the patch.

Z-score Normalization – All patches were standardized using the training set mean and standard deviation.

𝑋
norm
=
𝑋
−
𝜇
𝜎
X
norm
	​

=
σ
X−μ
	​


This ensured that the input distribution had stable magnitude and variance, which helped the CNN learn effectively.

After applying Z-score normalization, the CNN began training successfully:

Training and validation losses decreased steadily

Validation accuracy improved dramatically (≈0.99)

The model demonstrated strong generalization on the validation set

This demonstrates that proper data cleaning and normalization were essential steps in enabling stable and effective CNN training for UHI classification.