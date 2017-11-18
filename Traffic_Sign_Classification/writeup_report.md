#**Traffic Sign Recognition**

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./index.png
[image2]: ./count.png
[image3]: ./graybefore.png
[image4]: ./grayafter.png
[image5]: ./german1.png
[image6]: ./german2.png
[image7]: ./german3.png
[image8]: ./german4.png
[image9]: ./german5.png

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it!

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ? 34799
* The size of test set is ? 12630
* The shape of a traffic sign image is ? (32,32,3)
* The number of unique classes/labels in the data set is ? 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.

Here is an exploratory visualization of the data set. It is a bar chart showing the count for each class.

![alt text][image1]
![alt text][image2]
###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because the three channels, RGB, could be complex.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3]
![alt text][image4]

As a last step, I normalized the image data because we want our network to be more generalized and avoid the overfitting.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

Here I use the validation set, training set and test set provided by dataset. And I also tried to split the training set into the new traning set and validation set, where the rate is 4:1.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16      		|
| RELU          |                        |
| Max pooling        | 2x2 stride, Input = 10x10x16. Output = 5x5x16.
| Flatten       |   Input = 5x5x16. Output = 400.
| Fully connected		| Input = 400. Output = 120.   									|
| RELU          |    |
| Fully Connected  | Input = 120. Output = 84. |
| RELU          |                        |
| Fully Connected  | Input = 84. Output = 43. |



####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook.

To train the model, I used an Adam Optimizer, and the batch size is 128. And the numbers of epochs I have tried are 10, 20, 30, 50 and 60, the optimal choice is 50. The learning rates I have tried are 0.0005, 0.001, 0.0015, 0.002 and 0.003, and the optimal learning rate is 0.001.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?  99.4%
* validation set accuracy of ?   94.1%
* test set accuracy of ?   92.9%
* Discussion: I have tried different combinations of epochs and learning rates, and the optimal is 50 epochs and 0.001 learning rate. Too many epochs make no sense and lead to overfitting sometimes. Too large learning rate sometimes makes us cannot find the optimal, and too small learning rate wastes too much time.

<!--
If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
-->

If a well known architecture was chosen:
* What architecture was chosen? A max pooling layer after two convolution layers, then three fully connected layers, then the output.
* Why did you believe it would be relevant to the traffic sign application? The project is similar to the lab we researched in the class. The two convolution layers are enough to extract the characteristics of the image from the 32x32 images. The three fully connected layers with specific size could fit characteristics to the classes well.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? I achieved 99.4% accuracy on traning set, 94.1% accuracy on validation set and 92.9% accuracy on test set, which looks pretty well.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]
![alt text][image9]

The third image might be difficult to classify because it is a little blurred.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Speed limit (20km/h)      | Speed limit (20km/h) 									|
| Speed limit (30km/h)    	| Speed limit (30km/h)							|
| Speed limit (60km/h)			| Speed limit (60km/h)								|
| Speed limit (80km/h)   		| Speed limit (80km/h)				 				|
| No passing       	        | No passing      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 92.9%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Speed limit (20km/h) (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.         			| 0  									|
| .00     				| 1									|
| .00					    | 28						|
| .00	      			| 4		 				|
| .00				      | 6   							|

For the second image, the model is relatively sure that this is a Speed limit (30km/h) sign (probability of 1), and the image does contain a stop sign. The top five soft max probabilities were
| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.         			| 1  									|
| .00     				| 6									|
| .00					    | 5						|
| .00	      			| 2		 				|
| .00				      | 0   							|

For the third image, the model is relatively sure that this is a Speed limit (60km/h) sign (probability of 1), and the image does contain a stop sign. The top five soft max probabilities were
| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.         			| 3  									|
| .00     				| 6									|
| .00					    | 2						|
| .00	      			| 1		 				|
| .00				      | 5   							|

For the third image, the model is relatively sure that this is a Speed limit (80km/h) sign (probability of 1), and the image does contain a stop sign. The top five soft max probabilities were
| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.         			| 5  									|
| .00     				| 1									|
| .00					    | 2						|
| .00	      			| 6		 				|
| .00				      | 8   							|

For the third image, the model is relatively sure that this is a No passing sign (probability of 1), and the image does contain a stop sign. The top five soft max probabilities were
| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.         			| 9  									|
| .09     				| 35									|
| .00					    | 11						|
| .00	      			| 12		 				|
| .00				      | 13   							|

For the question 9, I have tried many times to plot the feature plots, but I failed, there were always errors and I couldn't figure out how to wolve it. So I hope you could kindly modify my code in the .ipynb file, so that I could get the idea that how to plot it. Also, I hope you can give me some guide on the tf.nn.top_k and the softmax function, how we understand it and how could we predict the probability.
