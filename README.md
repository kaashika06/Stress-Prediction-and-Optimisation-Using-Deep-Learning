# Stress Prediction and Optimisation Using Deep Learning

Comparing two deep learning approaches for predicting maximum von Mises stress in a lattice-structured cantilever beam, given its strut-thickness distribution. The beam is built from centered rectangular unit cells with strut thickness varying by node, and the question is how best to predict the resulting peak stress from that thickness distribution.

## Problem setup

- **Structure:** cantilever beam built from a lattice of centered rectangular unit cells; strut thickness varies linearly based on randomly generated nodal thickness values, defined by 226 thickness parameters per sample.
- **Loading:** fixed point load at the mid-right edge of the beam; only the thickness distribution varies across the dataset.
- **Data:** thickness profiles (`output.xlsx`, one row per sample, 226 columns) and per-sample von Mises stress values at ~3304-3305 node/strut locations from FEA simulation.

## Approach 1: CNN on max stress directly

Reduces each sample's stress field to a single label - its maximum von Mises stress - and trains a CNN to predict that scalar from the thickness distribution.

- The 226-length thickness vector is reshaped into a 16×16 grid (with padding) so it can be fed through convolutional layers.
- Architecture: Conv2D → MaxPooling → Conv2D → MaxPooling → Flatten → Dense → Dropout.
- Early stopping used to control overfitting.
- **Result:** an initial 50-epoch run overfits clearly (training loss drops to ~0.002 while validation loss climbs to ~0.016). Early stopping halts training around epoch 10, but a persistent gap between training and validation loss remains, the model can't fully close it.
- **Takeaway:** the residual gap points to a structural limitation, not just an overfitting issue, collapsing the whole stress field down to one scalar throws away the spatial information the model would need to predict it well.

## Approach 2: U-Net on the full stress field

Instead of predicting a single scalar, predicts the von Mises stress at every node, then takes the max over the predicted field.

- Both the thickness distribution and the stress values are mapped onto a spatial grid (thickness: 14×17 → mapped onto a 56×59 coordinate grid; both padded to 64×64) so an image-to-image model can be used.
- U-Net's encoder → decoder structure (built with Keras' Functional API, since it branches and merges rather than stacking layers sequentially) learns spatial features - edges, local thickness gradients, local concentrations, and reconstructs a full stress field.
- An initial run (~4,000 trainable parameters) underfits, with very high test loss. Increasing capacity to ~1.86M trainable parameters and normalizing the input data substantially improved training and validation loss convergence.

## Conclusion

The CNN's training and validation loss curves track each other but never converge. This is evidence of a persistent bias the model can't shed, because collapsing the stress field to a single max value throws away the spatial relationships between points. The U-Net, by predicting the stress at every node and only then taking the max, uses that spatial information and produces a noticeably better prediction of maximum stress. For this problem, predicting the full field first is the stronger approach, at the cost of a larger, more data-hungry model.

## Notes

This README currently covers the CNN vs. U-Net comparison (project question 6a). Other approaches explored in the broader project (gradient-based optimization, dimensionality reduction, classification framing, etc.) are not yet reflected here.
