This Github is for Create and Using Pawdel Model for Predicting Pawpularity Score Base on Cute Pet Picture and Its Auxiliary Data
This Problem is From Kaggle Compettion: PetFinder.my - Pawpularity Contest (https://www.kaggle.com/competitions/petfinder-pawpularity-score)
This Work is Part of My Coursework: Applied Deep Learning

Problem:
This Competition Wants Competitor to Create a Machine Learning Model for Predicting Pawpularity Score Which Told That "How Much Interest of This Pet Picture"
and Leading to Chance and Period of Adoption. High Pawpularity Score Means High Chance and in a Short Time This Pet in the Picture Will Get Adoption.

Data:
Seperate to 2 Type
1. Pet Picture
2. Each Picture Auxiliary Data

Exploring Data:
After Exploring The Data I Found That There are 2 Data Type. That is Pet Picture and Each Picture Auxiliary Data. The Problem is Each Pet Picture Has Difference Size and Pet's Position (The Pet in Each Picture Appear in Differece Position,
e.g., Top-Left and Bottom-Right). In Short Period of Working on This Work, To Overcome This Problem I Decided to Do a Multiple Random Crop on Each Picture, Making it Faster to Preprocess. And The Auxiliary Data, I Found That it is True/False of That Thing
is Appeared on Picture or Not, e.g., Eye, Face and Blur.

Model Design Problem:
After Exploring The Data, The First Question That Come to My Mind is That "How Do I Merge Pet Picture With This Auxiliary Data?" So I Research on Internet and Find One Interesting Method That is Project All Auxiliary Data to Size of Picture and Then Plus the Picture With This.
And It Work Well But In My Thought This Method is Creating Another Matrix That Has Size Equal To Picture Which Means If I Using 64 Batch (Picture) for Each Training Epoch, With This Method it Will Generate Another Matrix With Size of Picture so The Memory Must Hold 128 Picture
Which is too Large Leading to Speed of Training and Out of Memory Problems. To Overcome This Problem, I Decided to Project It to 3 Dimension Which will Scale How Much Each Picture's Color Dimension will Use. For Example If It Project
To 0.3, 0.6 and 0.1 It Will Multiply Picture's Red, Green and Blue Dimension With 0.3, 0.6 and 0.1 Respectively. I Assume That This Method Help Adding More Information To Pet Picture By Using Auxiliary Data Which Increase Model's Accuracy.
But I Have Another Assumption That is "What If The Auxiliary Data Corrupted Model Inferrence" so I Make an Ablation Study About It Also.

Model:
After Adding More Information to Pet Picture by Using Auxiliary Data.
The Main Model For Predicting Pawpularity Score I Decided To Use EfficientNet Model Architecture Which is One of The Best Model, It Has Great Accuracy, Compact and Fast Training. But To Enhance My Knowledge and Implementing Skill, I Decided to Reading Its Paper To Get Core Concept
That How It Work and Create It From Scratch.

Setting:
Split Data to Train and Test Set With 80 and 20 Percent Ratio.
Model's Parameter Respect To EfficientNetB0's Parameter. Set Batch Size as 64, Using MSE Loss, Adam Optimizer With 0.001 Learning Rate and Early Stopping Within 5 Epoch, Evaluate by RMSE.

Result:
In Test Set Found That Best RMSE Score is at Epoch 3 With 20.29418 and After That RMSE is Decreased Indicate That It Might be Overfitting and After Submission on Kaggle Found That The RMSE Score on Public and Private Dataset 
is 20.74327 and 20.66654 Respectively. Compare To The First Place on Leaderboard of Private Dataset With Only 16.82256 Which is 18.60% Better Than My Model.

Ablation Study:
I Decided To Make My Model Without Using Auxiliary Data Version and Found that, on Test Set, Best RMSE is at Epoch 3 With 20.87968 and After Submission on Kaggle Found That The RMSE Score on Public and Private Dataset 
is 20.70581 and 20.80263 Respectively. From This Result I Found That The RMSE Score of Public Dataset is Better Than The Using Auxiliary Data Model Version But In Private Dataset It Worse Than It, 
I Assume That The Model Without Using Auxiliary Data Might Overfitting on Public Dataset So It Underfitting on Private Dataset, Otherwise The Model Using Auxiliary Data Might Overfitting Less in Public Dataset But Generalize Better in Private Dataset.
Showing That by My Adding Information Using Auxiliary Data Method Making The Model Predicting More Generalize and More Accuracy.

Conclusion:
Despite My Model Doesn't Beat The First Place But This Work Taught Me The Knowledge About EfficientNet and Its Core Concept, Critical Thinking Skill About Saw a Problem and Solve It by Thinking a New Idea and The Skill of Implementing Machine Learning Model Using Pytorch
