# Feature-Reduction
Various dimensionality reduction techniques applied on datasets and results compared.


### Feature Selection
In Feature Selection, the features which are most relevant to the data and contribute to the prediction are identified and selected. It uses a simple logic- if the target variable, i.e. what we want to predict, is not affected by a certain input feature, that feature is discarded (just like we ought to discard toxic people who don’t contribute to our well-being!).  

Statistical measures such as chi-squared test, correlation coefficient etc. are used to rank features for this purpose and only those are kept which display a strong relationship with the target variable.

Feature selection algorithm applied on Wine Quality dataset- Chi-squared test.

Formula for Chi-squared test:

![Test Image 1](https://media.geeksforgeeks.org/wp-content/uploads/20210831120801/chiformula.PNG)

Where O= Observed Value(s)
E= Expected Value(s)


Link to dataset: [redwinequality.csv](https://github.com/aishanii/Feature-Reduction/files/7090356/redwinequality.csv)



### Feature extraction
Consider ‘Feature extraction’ to be the cooler cousin. It doesn’t select the best features, but creates an entirely new set of features from the existing ones, without losing the value that the original features hold. It does so by combining and reformatting primary features such that the new features are the most optimized for machine learning models to make accurate predictions.  

Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA) are the most common feature extraction methods.

Firstly, what are Principal Components?

Briefly, Principal Components are the new variables or features that are given birth as a result of applying 'linear combinations' to the initial features (basically, if features are considered as axes of a space, say a 3-D space , linear combinations are functions that allow them to move around and change their orientation in that space.)

The reduction is done in a way that the most of the information in the dataset is contained in the first few Principal Components, which contain the maximum variance. They are determined first by plotting a covariance matrix between all the features, which tells us which features are pretty much the same hence redundant, then by computing the eigen vectors and eigen values, which tell us the direction of the principal components and the amount of variance carried respectively. 

Naturally, the principal components represented by eigen vectors having the highest eigen value will be ranked first and will contain most of the information. 

Principal component analysis(PCA) performed on Banknote Authentication dataset.
Link to dataset: [data_banknote_authentication.csv](https://github.com/aishanii/Feature-Reduction/files/7090350/data_banknote_authentication.csv)





