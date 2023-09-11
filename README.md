# UWB_IMU_KF_2D_location
A repository of year project which is "a following robot with UWB/IMU dynamic locationing"

THE CODE COOKBOOK IS COMING SOON

# Implementation

1. record all imu and uwb data while car is accelerating, stopping, deccelerating, moving with constant speed.
2. Filter uwb data with wavelet algo by hands and lable as pairs <wavelet_params, uwb1>, <wavelet_params, uwb2>
3. Train 2 ai models(eg.SVM, Rendom Forest, classifycation or regration models are both ok) on step 2.
4. Dynamicly filters uwb1, uwb2 while car is working and calucates a location of robot and named as UWB LOCATION POINT.
5. calulate odom with IMU as IMU LOCATION POINT.
6. use KF to fusse UWB LOCATION POINTS and IMU LOCATION POINT.
