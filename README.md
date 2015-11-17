% Predicting Boston housing prices
% Renato L. F. Cunha
% 2015-11-16

\newpage

This document describes the implementation of a Machine Learning regressor that
is capable of predicting Boston housing prices. The data used here is loaded in
[sklearn's `load_boston` dataset](
http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html
)
and comes from the StatLib library which is maintained at Carnegie Mellon University.

# Statistical analysis and data exploration

Some information about the data set can be found below.

 * Number of samples: 506
 * Number of features: 13
 * Minimum housing price: 5.0
 * Maximum housing price: 50.0
 * Mean house price: 22.5328063241
 * Median house price: 21.2
 * Standard deviation of housing prices: 9.18801154528

# Evaluating model performance

The problem of *predicting* housing prices is clearly not a classification
problem and, therefore, is a regression problem, as the labels (the prices) are
continuous numerical data. Hence, for evaluating the model's performance, we
can consider performance metrics for regression.

For this and subsequent sections it will be assumed that the true values of
labels are part of vector $y$ and the $i$-th element is indexed as $y_i$.
Vector $\hat{y}$ contains the *predicted* values.

## Measures of model performance

This document will pitch the *Mean Absolute Error* (MAE) against the *Mean
Squared Error* (MSE). As their names indicate, we use error metrics that are
concerned with the relative distance of the predicted from the true value and,
therefore, disallow metrics that have the possibility of returning negative
values. (As a side effect, we also guarantee that any negative values do not
cancel out positive values found, giving the impression that the error is lower
than it really is.)

MAE can be defined as $\frac{1}{N}\sum_{i=1}^{N}|\hat{y}_i-y_i|$, while MSE can
be defined as $\frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i)^2$.  For this model in
particular, both metrics work equally well (determined empirically, but not
shown here), but notice that if $y$ has a unit, the MSE will have a squared
unit.

In general, though, MSE has more interesting properties: it is a differentiable
function, and, being an error metric, penalizes large errors more than low
errors (*i.e.* an absolute error of $0.1$ will account for $0.01$ in the final
error metric, while an error of $10$ will be counted as $100$). Because of
that, in this document the MSE metric was used.

## Splitting the data

To properly evaluate the model, the data we have must be split into two sets:
a training set and a testing set. The training set's objective is to train the
model, while the testing set is needed for evaluating the trained model's
performance in data that it has "never"[^1] seen before. If we don't split the
data, we risk having a model that can only make good predictions with the
training data set and, hence, we would end up with an overfit model.

[^1]: "Never" is in quotes because although that data is not seen explicitly,
we actually tune the model by evaluating its performance in the test data.
Therefore, the model carries our biases when fully learned.

## Cross-validation

Even if we split the data, our knowledge while tuning a model's parameters can
add biases to the model, which can still be overfit to the test data.
Therefore, ideally we need a third set the model has never seen to truly
evaluate its performance. The drawback of splitting the data into a third set
for model validation is that we lose valuable data that could be used to better
tune the model.

An alternative to separating the data is to use *cross-validation*.
Cross-validation is a way to predict the fit of a model to a hypothetical
validation set when such a set is not explicitly available. There is a [variety
of
ways](https://en.wikipedia.org/wiki/Cross-validation_(statistics)#Common_types_of_cross-validation)
in which cross-validation can be done. These methods can be divided into two
groups: exhaustive cross-validation, and non-exhaustive cross-validation.
Exhaustive methods are methods which learn and test on **all** possible ways to
divide the original data into a training and a testing set. As such, these
methods can take a while to compute, especially as the amount of data
increases. Non-exhaustive methods, as the name says, do not compute all ways of
splitting the data.

The most complete exhaustive method is probably leave-$p$-out cross-validation,
which involves using $p$ points as the validation set and the rest of the data
as the training set. This has to compute $\binom{n}{p}$ ways of splitting the
data, where $n$ is the total number of points in it. As one can imagine, as $n$
grows, this becomes impossible to calculate.

Non-exhaustive methods usually approximate leave-$p$-out. One such a method is
$k$-fold cross-validation, which consists of randomly partitioning the data
into $k$ equal sized subsets. Of these, one subset becomes the validation set,
while the other sets are used for training the model, and the process is
executed $k$ times, one for each validation subset. The performance of the
model, then, is the average of the performance of model in each of the $k$
executions. This method is attractive because all data points are used in the
overall process, with each point used only once for validation.

## Grid Search

As mentioned before, machine learning models usually have some parameters
(usually called hyperparameters) that won't be learned by the model and,
therefore, must be tuned for improving prediction performance. One option used
all-too-often is to try different parameters in an ad-hoc way until the model
has a good enough performance. That's not a good method, as it is error prone
and consumes precious human time.

A better approach for optimizing a model's parameters is to combine human
insight with automated methods: a user gives the ranges of parameters which she
thinks would work well, and the computer interpolates them making a grid and
evaluates each point of this grid. In the end, the combination of parameters
with best performance is chosen. One can basically do an exhaustive search in
all parameters, or sample points in the grid. The trade-off is in the solution
quality *versus* run time axis. The exhaustive search guarantees the best value
will be found if it is in the grid, but may run for a long time depending on
the model's complexity. The sampled search will run faster, but may lose a few
points and the optimal. Because our model is relatively simple, we use
exhaustive search in this document.

# Analyzing model performance

Up to this point we've been describing the techniques used: MSE as performance
metric, splitting the data between training and test, $k$-fold cross-validation
for validation and an exhaustive grid search for finding the best parameters.
In this section we will analyse the model constructed.

The model we built is a decision tree regressor in which we varied the maximum
tree depth by passing the `max_depth` argument to sklearn's
[`DecisionTreeRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html).

## Trends

We plotted 10 graphs of the decision tree performance, one for each maximum
tree depth, and an interesting trend is found. The more we increase the tree
depth, the more we reduce the training error, which goes down to practically
zero. The training error, though, seems to find its best values around depths
6 & 5, and then starts to increase with the maximum tree depth.

## Bias and Variance

[lc1]: learning-curve-1.png "Learning curve of the model with `max_depth=1`"
![Learning curve of the model with `max_depth=1`.\label{fig:lc1}][lc1]

[lc10]: learning-curve-10.png "Learning curve of the model with `max_depth=10`"
![Learning curve of the model with `max_depth=10`.\label{fig:lc10}][lc10]

Looking closely to the model performance with depths 1 and 10 in Figures
\ref{fig:lc1} and \ref{fig:lc2}, we see the two extremities of the model. In
the first figure, the model is clearly suffering from high bias, as it is
unable to do well even in the training set. The tree is probably too shallow
for it to learn the relationships inside the data. On the other end of the
spectrum, Figure \ref{fig:lc2} shows an overfit model, for the model with
a tree of depth 10 has training error zero, but as we analyzed before, has
worse performance than previous models when looking at the testing error.

## Model complexity relationships

[model]: model-complexity.png "Learning curve of the model with `max_depth=1`"
![Learning curve of the model with `max_depth=1`.\label{fig:model}][model]

The conclusions of the previous section are further evidenced when we look at
the model complexity graph shown in Figure \ref{fig:model}, which shows that
the error plateaus with `max_depth` values higher than approximately 5.
Therefore, the model that best generalizes the data is the one with `max_depth
= 5`, which is confirmed by calling the `best_params_` member of
[`GridSearchCV`](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html),
which gives values in the range `[5, 6]` with high frequency, and sometimes
`4`, or `7`. This is due to the random sampling in the way cross-validation is
done. So we should choose the least complex model that explains the data, and
I'd go with 5 here.

The complete output of the grid search is given below:

```
Fitting 3 folds for each of 10 candidates, totalling 30 fits
[Parallel(n_jobs=1)]: Done  30 out of  30 | elapsed:    0.1s finished
GridSearchCV(cv=None, error_score='raise',
  estimator=DecisionTreeRegressor(criterion='mse', max_depth=None,
	max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
	min_samples_split=2, min_weight_fraction_leaf=0.0, presort=False,
	random_state=None, splitter='best'),
  fit_params={}, iid=True, n_jobs=1,
  param_grid={'max_depth': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)},
  pre_dispatch='2*n_jobs', refit=True,
  scoring=make_scorer(performance_metric, greater_is_better=False),
  verbose=True)
Best Parameters:  {'max_depth': 6}
```

# Model prediction

After this whole discussion, we can finally make a prediction. In particular,
we would like to predict the price of a house with the following features:
`[11.95, 0.0, 18.1, 0, 0.659, 5.609, 90.0, 1.385, 24, 680.0, 20.2, 332.09,
12.13]`. When making a prediction with the learned model, we end up with
a prediction of 20.76598639. This prediction's value is well within the range
of the mean and standard deviation, and is 97% of the median house price.

