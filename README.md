# svm_logistic_reg_naive_bayes

**Description**

I use the email SPAM data set referenced in the well-known machine
learning book The Elements of Statistical Learning Theory
(http://statweb.stanford.edu/~tibs/ElemStatLearn/). It consists of 4601 email messages, from
which 57 features have been extracted. The features are as follows:
• 48 features giving the percentage (0 to 100) of words in a given message that match a
given word on a list that contains such words like “business”, “free”, etc.
• 6 features giving the percentage (0 to 100) of characters in the email that match a given
character on a list ; ( [ ! $ #
• Feature 55 is the average length of an uninterrupted sequence of capital letters (max is
40.3, mean is 4.9).
• Feature 56 is the length of the longest uninterrupted sequence of capital letters (max is
45.0, mean is 52.6).
• Feature 57 is the sum of the lengths of uninterrupted sequence of capital letters

The class label (Y) is binary: 1= SPAM and 2 = NOT SPAM.

**Overview**

I compare the performance of a Logistic Regression classifier, a Naïve Bayes Classifier,
and a Support Vector Machine classifier.
