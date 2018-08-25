# Cascade-keypoint-localization
An end-to-end cascade keypoint detection model implemented using Pytorch.
Also used in FashionAI challenge.

![alt text](https://github.com/xiaodongww/Cascade-keypoint-localization/blob/master/pics/network.png)

We design a two-stage model to predict the location of keypoints.  
* stage 1:  predict a rough location ![](http://chart.googleapis.com/chart?cht=tx&chl=$x^K$).  
* stage 2:   use the location predicted in the first stage as the center and crop an area of 31*31 to extract local features of the keypoint. These local features will be used to precdict the differece between the prediction of stage 1 and the groundtrouth ![](http://chart.googleapis.com/chart?cht=tx&chl=$\triangle+x^k$).  

Example:
![alt text](https://github.com/xiaodongww/Cascade-keypoint-localization/blob/master/pics/result.gif)


