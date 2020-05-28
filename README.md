# 1. intro-to-pytorch

* these are codes i wrote as i was learning pytorch
* i followed this tutorial: https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/
* since i didn't have a lot of experience with OOP, some of the code blocks required more time and effort to follow

# 2. custom_dataset.py

* this code describes how to create a custom dataset in pytorch
* the code was influenced by this tutorial: https://www.journaldev.com/36576/pytorch-dataloader

## some key takeaways

* defining a custom dataset requires overwriting \_\_len\_\_ and \_\_getitem\_\_ methods

# 3. train_dev_split.py

* this is to learn how to split the data into training and dev set using random\_split

# 4. mlp_binaryClassification.py

* code to do binary classification with a multi layer perceptron (mlp)

# 5. mlp_multiclassClassification.py

* code to do mutli-class classification with a multi layer perceptron (mlp)

## iris dataset

* the dataset has 4 inputs and 1 output
* the inputs are:
  * sepal length (cm)
  * sepal width (cm)
  * petal length (cm)
  * petal width (cm)
 * the outputs could be either Iris setosa, Iris versicolour, Iris virginica
 
 ![image](./images/iris.jpg)
 
 (image from https://medium.com/@jebaseelanravi96/machine-learning-iris-classification-33aa18a4a983)
 
 # 5. mlp_regression.py

* code to do regression with a multi layer percept

## boston housing

* the dataset has 13 inputs and 1 output
* one of the input is a binary class
* the rest of the inputs are continuous
