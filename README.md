# MNIST 99.4 under 8k parameters

## Target

Reach 99.4 under 8k parameters

```
Changes made:
- Removed two conv layer to reduce parameters
- reduced kernel size to reduce parameters
- 
```

### Analysis

1. Removing the data augmentation reduce the strictness of the dataset.
2. Adding Bias increased the parameters so that a bit more features are learned.

Model Parameters: 7,994

Maximum Training Accuracy: 99.08

Manimum Testing Accuracy: 99.45

No of times 99.4 is hit - 7 times out of 15 epochs

We have been very close to pur target, we hit 99.3+ many times but didnt cross 99.4.

## Target

Reach 99.4 under 8k parameters

```
Changes made:
- Removed Data Augmentation - Random Apply, Random crop and Resize.
- Step_size of Step_LR increased to 8
- added Bias to the model to increase the parameter for a bit more learning for the last push
```

### Analysis

1. Removing the data augmentation reduce the strictness of the dataset.
2. Adding Bias increased the parameters so that a bit more features are learned.

Model Parameters: 7,994

Maximum Training Accuracy: 99.08

Manimum Testing Accuracy: 99.45

No of times 99.4 is hit - 7 times out of 15 epochs

We have finally achieved our target which is crazy.
