# MNIST 99.4 under 8k parameters

# Model 1

## Target

Reach 99.4 under 13.5k parameters, here we will set the basic model right.

```
Changes made:
- Removed two conv layer to reduce parameters
- reduced kernel size to reduce parameters
- remove dropouts
```

### Analysis

Here we are setting the base structure of model right, and ensuring that we just need to reduce the parameters and fine tune hyper parameters.

Model Parameters: 13,476

Maximum Training Accuracy: 98.56

Manimum Testing Accuracy: 99.40




# Model 2

## Target

Reach 99 under 8k parameters

```
Changes made:
- Removed two conv layer to reduce parameters
- reduced kernel size to reduce parameters
```

### Analysis

1. Removing the data augmentation reduce the strictness of the dataset.
2. Adding Bias increased the parameters so that a bit more features are learned.

Model Parameters: 7,901

Maximum Training Accuracy: 97.14

Manimum Testing Accuracy: 99.08

We have reduced the total number of parameters to the desired parameters, just need to fine tune it.

# Model 3

## Target

Reach 99.4 under 8k parameters

```
Changes made:
- Removed Data Augmentation - Random Apply, Random crop and Resize.
- Step_size of Step_LR increased to 8
- added Bias to the model to increase the parameter for a bit more learning for the last push
- Removed all dropouts.
```

### Analysis

1. Removing the data augmentation reduce the strictness of the dataset.
2. Adding Bias increased the parameters so that a bit more features are learned.

Model Parameters: 7,994

Maximum Training Accuracy: 99.08

Manimum Testing Accuracy: 99.45

No of times 99.4 is hit - 7 times out of 15 epochs

We have finally achieved our target which is crazy.
