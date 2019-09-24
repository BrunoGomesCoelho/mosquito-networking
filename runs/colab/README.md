# Logging of my ideas and thoughts during the experiments.

## 1/

The first 6 hour and 100 epochs resulted in no learning. Loss is still 0.69 basically (expected for random with crossentropy).

A analysis of the prediction shows we in fact didn't learning anything - the model is predicting a constant value.

I'm going to try scale the data differently - since we have many zeros, I'm going to exclude them from the mean/var calculations columns wise.

Going to try bigger batch\_size; Might try overfitting very small amount of data.

Might have to include Batch Norm - in fact, I should really analyze the size of the network better.
