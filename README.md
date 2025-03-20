# Music Autoencoder and Latent Space Analysis 🎵🚀

## Overview
This project presents the design, implementation, and evaluation of a music autoencoder based on neural networks. The goal is to compress audio signals into compact latent vectors and then decode them to reconstruct the original sound. In addition, the project explores latent space analysis through both supervised and unsupervised techniques to understand relationships among encoded music signals. This work was developed as part of the "TD VI: Artificial Intelligence" course (2nd Semester, 2024) by Ezequiel Grinblat, Luca Mazzarello, and Camila Migdal.

## Project Structure
The project is organized into the following sections:

1. **Introduction**  
   - Motivation and objectives of the project.
   - Overview of the challenge: designing an autoencoder for music and obtaining a representative latent vector for each song.

2. **Autoencoder Design: CELAutoencoder**  
   - Description of the two initial versions of the autoencoder:
     - **CELAutoencoderLeakyReLU**: Uses LeakyReLU activation.
     - **CELAutoencoderSiLU**: Uses SiLU activation.
   - Explanation of the encoder and decoder architectures:
     - **Encoder**: Compresses the input audio signal (1×110250 samples) using convolutional layers, batch normalization, and activation functions, reducing the dimensionality.
     - **Decoder**: Reconstructs the original audio signal from the compressed representation, including an upsampling layer to recover the original length.

3. **Encoding Songs into a Latent Vector**  
   - Implementation of the encoder to compress audio into latent vectors.
   - Experiments with different latent vector sizes:
     - The smallest possible vector (baseline model).
     - A smaller vector (half the size).
     - A larger vector (double the size).
   - Analysis of reconstruction quality versus latent space dimensionality (with training and validation losses as performance indicators).

4. **Exploratory Analysis of Latent Vectors**  
   - Unsupervised analysis using clustering (K-means) on the latent representations.
   - Dimensionality reduction with PCA for visualization (typically projecting to 2D).
   - Evaluation of clustering quality with methods such as the elbow method and metrics like homogeneity, completeness, and V-measure.
   - Comparison of clustering results across the three latent space sizes.

5. **Encoding New Music and Generation**  
   - Testing the autoencoder with new (unseen) music to evaluate its generalization capability.
   - Exploration of latent space modifications:
     - Amplification/attenuation, addition of Gaussian noise, and smoothing.
   - Discussion on whether these manipulations allow for generating new or altered music.

6. **Additional Experiments (Optional)**  
   - Training a generative model for music creation based on a chosen architecture.
   - Discussion of the results and challenges encountered.

## Tools and Libraries Used
- **Python** and **Jupyter Notebook** for experiment development.
- **PyTorch** for building and training the autoencoder.
- Libraries for audio processing and visualization (e.g., librosa, matplotlib, scikit-learn).
- Additional utilities for clustering evaluation and GPU memory management.

## How to Run the Analysis
1. **Set up the Environment:**  
   - Install the necessary Python packages (e.g., PyTorch, librosa, scikit-learn, matplotlib).
2. **Execute the Notebook:**  
   - Open the provided Jupyter Notebook (e.g., `TP4.ipynb`) to run the autoencoder training, latent space exploration, and audio reconstruction experiments.
3. **Evaluate the Results:**  
   - Review generated plots, loss metrics, clustering analysis, and listen to reconstructed audio to assess model performance.

## Key Insights
- **Autoencoder Performance:**  
  - The CELAutoencoderSiLU variant demonstrated superior reconstruction quality (lower train/validation losses) compared to the LeakyReLU version.
- **Latent Space Dimensionality:**  
  - A balance must be struck between compression and reconstruction quality. Extremely small latent spaces result in loss of fine details, while larger spaces retain more information.
- **Clustering Analysis:**  
  - Both unsupervised (clustering via K-means) and supervised (genre classification) approaches reveal that latent vectors capture meaningful audio features, although challenges remain in classifying genres with less distinct patterns.
- **Generalization and Generation:**  
  - The autoencoder is capable of encoding and reconstructing unseen music, and slight modifications in the latent space can lead to interesting variations in the generated audio.

## Acknowledgements 🙏
Special thanks to the course instructors, peers, and mentors for their guidance and support throughout the development of this project.

Happy coding and music exploring! 🎶
