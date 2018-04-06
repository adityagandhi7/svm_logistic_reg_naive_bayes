% Loading data into Matlab
load('spamdata.mat');

% Calculate mean and variance to normalize the data, and transposing before
% the next step
train_data_trans = train_data';
train_mean_features = mean(train_data_trans);
train_sd_features = std(train_data_trans);
norm_train_features = train_data_trans - train_mean_features;
for i = 1:size(train_data_trans,1)
    for j = 1:size(train_data_trans,2)
        norm_train_features(i,j) = norm_train_features(i,j) / train_sd_features(1,j);
    end
end

% Logistic Regression using mnrfit
[result, dev, stats] = mnrfit(norm_train_features, ytrain);
train_result = mnrval(result, norm_train_features);

%Creating a predicted class matrix
Class1_Prob = train_result(:,1);
Class2_Prob = train_result(:,2);
Pred_Class = Class1_Prob;
for i = 1:size(Class1_Prob,1)
    if Class1_Prob(i,1) >= Class2_Prob(i,1)
        Pred_Class(i,1) = 1;
    else
        Pred_Class(i,1) = 2;
    end
end
Pred_Class = Pred_Class';

%Confusion matrix calculation and accuracy
C = confusionmat(ytrain, Pred_Class);
correct_classified = C(1,1) + C(2,2);
train_spam_accuracy = correct_classified*100 / 3065;

%Normalizing the test data
test_data_trans = test_data';
norm_test_features = test_data_trans - train_mean_features;
for i = 1:size(test_data_trans,1)
    for j = 1:size(test_data_trans,2)
        norm_test_features(i,j) = norm_test_features(i,j) / train_sd_features(1,j);
    end
end

%Applying the model on the test data
test_result = mnrval(result, norm_test_features);

%Creating a predicted class matrix for the test data, fitting the model
%and calculating the accuracy
Class1_Prob_test = test_result(:,1);
Class2_Prob_test = test_result(:,2);
Pred_Class_test = Class1_Prob_test;
for i = 1:size(Class1_Prob_test,1)
    if Class1_Prob_test(i,1) >= Class2_Prob_test(i,1)
        Pred_Class_test(i,1) = 1;
    else
        Pred_Class_test(i,1) = 2;
    end
end
Pred_Class_test = Pred_Class_test';
C_test = confusionmat(ytest, Pred_Class_test);
correct_classified_test = C_test(1,1) + C_test(2,2);
test_spam_accuracy = correct_classified_test*100 / 1536;

% Naive Bayes Classifier for training data
NBModel = fitcnb(norm_train_features, ytrain);
Pred_Class_NB = resubPredict(NBModel);
C_train_NB = confusionmat(ytrain,Pred_Class_NB);
correct_classified_NB = C_train_NB(1,1) + C_train_NB(2,2);
train_spam_accuracy_NB = correct_classified_NB*100 / 3065;

% Using Naive Bayes model to fit the test data
Pred_Class_test_NB = predict(NBModel, norm_test_features);
C_test_NB = confusionmat(ytest,Pred_Class_test_NB);
correct_classified_test_NB = C_test_NB(1,1) + C_test_NB(2,2);
test_spam_accuracy_NB = correct_classified_test_NB*100 / 1536;

%Support Vector machine implementation on training data
svm=fitcsvm(norm_train_features,ytrain,'solver','SMO','IterationLimit',50000);
Pred_Class_train_SVM = predict(svm, norm_train_features);
C_train_SVM = confusionmat(ytrain, Pred_Class_train_SVM);
correct_classified_train_SVM = C_train_SVM(1,1) + C_train_SVM(2,2);
train_spam_accuracy_SVM = correct_classified_train_SVM*100 / 3065;

%Support Vector machine implementation on test data
Pred_Class_test_SVM = predict(svm, norm_test_features);
C_test_SVM = confusionmat(ytest, Pred_Class_test_SVM);
correct_classified_test_SVM = C_test_SVM(1,1) + C_test_SVM(2,2);
test_spam_accuracy_SVM = correct_classified_test_SVM*100 / 1536;
