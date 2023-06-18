# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model used is a Random Forest Classifier.
Trained with the default parameters privided by scikit-learn.
Created by Moises Gonzalez.

## Intended Use

The objective of the model is to predict the salary of a person given some socioeconomic details.

## Training Data

The data used for the model training is the *_Census Income_* retrieved from the UC Irvine  Machine Learning Repository.
The target label is the "salary" and the values available are greater and less than 50k.

### Data Split

To train the model, the data was split in an 80-20 ratio for train and test.

To use the data, the leading and trailing spaces were removed. A One-Hot-Encoder was used for the categorical features.
Finally a label binarizer was used on the target.

## Evaluation Data

## Metrics

The metrics used for the evaluation of the model are: Precision, Recall & Fbeta.

The overall performance of the model on the test dataset is:

- Precision -> 0.7419847328244275
- recall -> 0.6198979591836735
- fbeta -> 0.6754690757470465

## Ethical Considerations

Some features (gender, race, native country) are present at the dataset and this could lead to some sort of discrimination.


## Caveats and Recommendations

The census was donated on 1996. Therefore, the data is not an updated representation of the current salaries nor the population.
This census is based on data from the USA, so it is expected to perform better on predictions for the USA.  
