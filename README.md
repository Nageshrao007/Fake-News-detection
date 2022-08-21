# Fake-News-detection
Problem Description 
This project is intended to get a classifier that can handle the Spam email problem. The main idea behind the classification (supervised learning) is to get a data set with labeled object, and the classifier will be able to process these objects and learn from it. So later when you provide a new object, the classifier will be mapping this object to one of the learned classes (labels). 
In this project, we used the “Fake News” Dataset that was provided by the instructor. It consists of two subtypes, the Fake news dataset, and the celebrity dataset. We combined both datasets to form one big dataset that consists of 980 documents (even though it is a small dataset, but we were able to get accepted results). 
We have used Naïve Bayse Classifier with features selection technique to enhance the results and reduce the training time.  
Learning Methods Used 
We have used many of the techniques we learned during the current semester in this course to handle the 	data beside some new techniques that we have learned and applied quickly 
Data Preprocessing 
We have used NLTK library to process tokenize and stem the different terms. 
Data Vectorization 
We have used the TF-IDF to represent the data as a vector, this allowed us to weight the terms and to reduce 	the effect of useless terms. 
Features Selections 
In this task, we used the Mutual Information Score to get a score for each term that represents how much one 	term could affect the decision of the classifier in determining the predicted class. 
To select best K, we have tested the top scores over a sample of the data (30%), this sample was used to train 	the model, where we divided the sample to 80% for training and 20% for validation. 
Classification 
Here, we have used the Naïve Bayse Classifier since it has been proved to be the best in this domain of 		classification. We used the MultinomialNB package from SKLearn. 
After getting the best Top K from the previous task, we trained the full dataset using the top K features 	only over the full data set. Where we divided that by 80% training and 20% testing. 
Results 
For Features selection, we got the following plot showing us the accuracy over Top N: 
 
From this data, we get that Top 559 features was giving the best result over the sample dataset with accuracy of 	74.5%. 
Noting that originally, we had 11,379 features. where 9,684 features with score > 0. so, we tested the top k only 	for the Non 0 score features. 
Then we trained the NB Classifier over the full dataset, with only the top 559 features as we learned previously. 	That allowed us to reduce the number of features used for training by almost 95%. 
 Finally, the result of training the full dataset gave us an accuracy of 84.6%.  
