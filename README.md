<h1 align="center">Covid-Trend-Tracker📉</h1>
<h2>Goal🎯</h2>To analyze the intricate correlation between the percentage of
people vaccinated and the change in Covid strain using Gradient Descent Model.
<h2>What is Gradient Descent?</h2>
The million-dollar question!
Let’s say you are playing a game where the players are at the top of a mountain, and they are asked to reach the lowest point of the mountain. Additionally, they are blindfolded. So,what approach do you think would make you reach the lowest point?
Take a moment to think about this before you read on.
The best way is to observe the ground and find where the land descends. From that position, take a step in the descending direction and iterate this process until we reach the
lowest point.<br>

<img src="https://github.com/vaibhav5140/covid-trend-tracker/assets/85643531/1af28cb3-1dc5-4203-a1ba-758e9542cbc8" align="centre" height ="225" width="370" ><br>

So in ritualistic terms Gradient descent is an interactive optimization algorithm for finding the local minimum of a function. A Gradient measures how much the output of a function changes if you change the inputs a little bit.Suppose you have a ball and and a bowl. No matter wherever you slide the ball in the bowl, it will eventually land in the bottom of the bowl.As you see this ball follows a path that ends at the bottom of the bowl. We can also say that the ball is descending in the bottom of the bowl. As you can see from the image the red lines are gradient of the bowl and the blue line is the path of the ball and as the path of the ball’s slope is decreasing, it is called as gradient descent.<br>

<img src="https://github.com/vaibhav5140/covid-trend-tracker/assets/85643531/a3b3a4a2-5c6d-4a62-94ea-afff11987ca8" align="centre" height ="225" width="370" ><br>

In my model my goal is to reduce the cost in my input data. the cost function is used to monitor the error in predictions of a model. So minimizing this, basically means getting to the lowest error value possible or increasing the accuracy of the model.<br>
<br>
First Let’s start by the topic that you know till now i.e. Linear Algebra. Let first use linear algebra and its formula<br>
<img src="https://github.com/vaibhav5140/covid-trend-tracker/assets/85643531/c2d4407a-bcc9-4cfc-9cf9-dd4e9d95fcce" align="centre" height ="225" width="370" ><br
                                                                                                                                                               
<i>(This is a random example of data points just for explanation)</i><br>
The basic formula that we can use in this model is<br>
y = mx+b <br>
where,
y = predictor, m = slope, x = input, b= y-intercept.<br>
A standard approach to solving this type of problem is to define an error function (also called a cost function) that measures how “good” a given line is. This function will take in a (m,b) pair and return an error value based on how well the line fits our data. To compute this error for a given line, we’ll iterate through each (x,y) point in our data set and sum the square distances between each point’s y value and the plotted line’s y value (computed at mx + b).It is also called residual value and denoted by(R).<br>
R^2 = Σ(Observed value-Predicted value)^2<br>
Here, Predicted value = y = mx +b (because we are changing the values of m and b)<br>
It’s conventional to square this distance to ensure that it is positive and to make our error function differentiable. The sum of residual values is called error, cost or lost function. It is defined as a function that measures the performance of a model for any given data. Cost Function quantifies the error between predicted values and predicted values and presents it in the form of a single real number.<br><br>
The sum of residual values is called error, cost or lost function. It is defined as
a function that measures the performance of a model for any given data. Cost
Function quantifies the error between predicted values and predicted values and presents it
in the form of a single real number

<img src="https://github.com/vaibhav5140/covid-trend-tracker/assets/85643531/596eb1c0-f13a-4a3a-8d68-6ae70cff825b" align="centre" height ="225" width="370" ><br>
Lines that fit our data better (where better is defined by our error
function) will result in lower error values. If we minimize this
function, we will get the best line for our data<br>
This graph below is plotted between between slope(m),intercept(b)
and sum of squared residuals(R^2).

<img src="https://github.com/vaibhav5140/covid-trend-tracker/assets/85643531/15de4150-ea28-42b8-ad44-a83d3ff232ef" align="centre" height ="225" width="370" ><br>

NOW, we want to find the values for the intercept and slope that
gives us the minimum sum of squared residuals. And to find that
values of m and b we need to find the derivative of R^2 = Σ(Observed
value-Predicted value)^2.<br>
R^2 = Σ(Observed value- m*x +b)^2<br>
We have to take the partial derivative of (R^2) because two
variables(m,b) are involved in this function.<br>
NOTE: When we have two or more derivatives of the same function,
they are called Gradient .We will use this Gradient function to descend
to lowest point in this Loss function which in our case is Sum of
residuals.This why algorithm is called Gradient descent.<br>
After finding the derivative we wanna find those values of m and b for
which we could get the lowest value of loss function(R^2) i.e . is the red
point in below graph. In the derivative of R^2 we would take some
random values of variables(lets say m=0 and b=1) that would the
starting point in the below graph. From that starting point we would start
moving toward lowest point(red point in graph) and we would move the
way as shown by white dots in the below curve. We can observe that we
take large steps when we are far away from lowest point and we take
small steps as we move close to lowest point. This is called Step size or
we can say iteration.<br>
Step size (intercept)= dR^2/d intercept * Learning rate
<br>
Step size (slope)= dR^2/d slope * Learning rate<br>
<i>(Learning rate gives the rate of speed where the gradient moves during
gradient descent. It controls how large of a step we take downhill during each
iteration. If we take too large of a step, we may step over the minimum.We
might start with a large value like 0.1, then try exponentially lower values:
0.01, 0.001, etc.)</i>

<img src="https://github.com/vaibhav5140/covid-trend-tracker/assets/85643531/f9b0709d-e069-4694-a030-95c2141ced6b" align="centre" height ="225" width="370" ><br
                                                                                                                                                               
<h2>To figure out -Correlation among Percentage of people
vaccinated Vs Relative change in strain of Covid:</h2>
Consider the dataset:<br>
Total number of vaccination doses administered per 100 people in the total
population. (X)<br>
X: [1,2,3,4,5]<br>
Relative change after Vaccination (Y)<br>
Y: [ 5,7,9,11,13]<br>
Now, our aim is to find a predictive model by using gradient Descent Algorithm
to predict the change after vaccination if the percentage of people vaccinated
is being given

<img src="https://github.com/vaibhav5140/covid-trend-tracker/assets/85643531/baf52a4c-2715-4d5f-bbbf-fb7efe9b0305" align="centre" height ="225" width="370" ><br>

It’s simply sum of all (actual-predicted) ^2 and then divided by n. here n is
number of data-points.(In our case it’s 10.)<br>
<img src="https://github.com/vaibhav5140/covid-trend-tracker/assets/85643531/bae31aef-eab6-4272-804d-3f0ff99381f7" align="centre" height ="225" width="370" ><br>

This is our cost-Function, and we have to somehow reach the red-point from
where we will find parameters (slope, intercept) which will give us the efficient
line of our predictive model with least error.<br>
Now how we go from any given point towards the red-point?<br>
<img src="https://github.com/vaibhav5140/covid-trend-tracker/assets/85643531/e8261156-217f-4d64-9183-abaa12e08d9c" align="centre" height ="225" width="370" ><br>

Let ‘b’ be some parameter, then we have move in such a way that while
approaching towards the red-point our step should be smaller compared to
when we are far from it.<br>
So how we have to move:<br>
<img src="https://github.com/vaibhav5140/covid-trend-tracker/assets/85643531/0049c431-b751-4f00-bb64-b5803cd1f2a5" align="centre" height ="225" width="370" ><br>

As we have two parameters so partial differentiation is done by which we are
moving opposite to the gradient of cost function to get the minima of that
function.
<br>Changing the value of slope and intercept iteratively towards the minima point
will give us the minima at some ith iteration.<br>

<h2>Output</h2>I now have all the tools needed to run gradient descent. I can initialize my
search to start at any pair of m and b values (i.e., any line) and let the gradient
descent algorithm march downhill on my error function towards the best line.
Each iteration will update m and b to a line that yields slightly lower error than
the previous iteration. The direction to move in for each iteration is calculated
using the two partial derivatives from above.<br>

<img src="https://github.com/vaibhav5140/covid-trend-tracker/assets/85643531/bb9cb8f9-ef13-456e-9d1b-d9da2397d41a" align="centre" height ="225" width="370" ><br>

So, the line with slope as 2 and intercept as 3(approx.) will be the predictive
model for covid data-set we took as an example.
