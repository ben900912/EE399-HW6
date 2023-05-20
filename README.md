# EE-399-HW6
``Author: Ben Li``
``Date: 5/19/2023``
``Course: SP 2023 EE399``

![image](https://github.com/ben900912/EE399-HW6/assets/121909443/f7967f1e-b00f-460f-9812-4882f328fd45)
## Abstract
In this assignment, we explore the SHRED (SHallow REcurrent Decoder) model architecture for reconstructing spatio-temporal fields from sensor measurements. The example code and data for sea-surface temperature, available on GitHub, are downloaded and used for training the model. The trained model's performance is evaluated and visualized through plots.

To gain further insights into the model's behavior, we conduct several analyses. First, we investigate the performance as a function of the time lag variable, which determines the number of previous time steps considered for prediction. By varying the lag value, we assess how the model's accuracy is affected.

Next, we examine the impact of noise on the model's performance. Gaussian noise is added to the data, and the model is trained and evaluated under different noise levels. This analysis helps understand the model's robustness to noisy input.

Lastly, we analyze the performance as a function of the number of sensors. By randomly selecting different sensor locations, we assess how the model's accuracy varies with the number of sensors used for reconstruction.

Through these analyses, we gain insights into the SHRED model's behavior and its sensitivity to different factors such as time lag, noise, and sensor count. The results provide valuable information for understanding and optimizing the model's performance in real-world scenarios.
## Introduction and Overview
The analysis of spatio-temporal data is crucial for understanding and predicting various natural phenomena. The SHRED (SHallow REcurrent Decoder) model is an architecture that combines a recurrent layer (LSTM) with a shallow decoder network to reconstruct high-dimensional spatio-temporal fields from a trajectory of sensor measurements. In this assignment, we explore the capabilities of the SHRED model using an example code and dataset for sea-surface temperature.

The assignment focuses on five main tasks. First, we begin by downloading the example code and data from a GitHub repository (https://github.com/Jan-Williams/pyshred). This code provides an implementation of the SHRED model and utilizes sea-surface temperature data as input.

After obtaining the code and data, we proceed to train the SHRED model using the provided dataset. The training process involves preparing the input sequences, dividing the data into training, validation, and test sets, and configuring the model's architecture. We then train the model using the training set and evaluate its performance.

Once the model is trained, we move on to the analysis of its performance as a function of the time lag variable. The time lag determines the number of previous time steps considered for prediction. By varying the lag value, we assess how the model's accuracy is influenced and observe any patterns or trends.

Next, we investigate the impact of noise on the model's performance. Gaussian noise is added to the sea-surface temperature data, and the model is trained and evaluated under different noise levels. This analysis helps us understand how the model responds to noisy input and assess its robustness in realistic scenarios.

Finally, we explore the performance of the SHRED model as a function of the number of sensors used for reconstruction. By randomly selecting different sensor locations, we evaluate the model's accuracy for varying sensor counts. This analysis provides insights into the relationship between the number of sensors and the model's performance.

Throughout the assignment, we visualize the results using plots and metrics to facilitate a clear understanding of the SHRED model's behavior. By completing these tasks, we gain valuable insights into the model's capabilities, limitations, and sensitivity to various factors, contributing to our understanding of spatio-temporal data analysis using the SHRED architecture.
## Theoretical Background

The SHRED (SHallow REcurrent Decoder) model is a network architecture designed for the reconstruction of high-dimensional spatio-temporal fields from a trajectory of sensor measurements. It combines a recurrent layer, specifically LSTM (Long Short-Term Memory), with a shallow decoder network.

Spatio-temporal data refers to data that varies across space and time, such as weather patterns, ocean currents, or traffic flow. These data often exhibit complex dependencies and require specialized models to capture their underlying dynamics accurately.

The SHRED model addresses this challenge by leveraging the power of recurrent neural networks (RNNs) and decoders. RNNs, such as LSTM, are well-suited for capturing temporal dependencies by maintaining an internal memory of past information. LSTM units have gating mechanisms that regulate the flow of information, allowing them to effectively capture long-term dependencies in sequential data.

In the SHRED architecture, the LSTM layer is combined with a shallow decoder network. The decoder network is responsible for reconstructing the high-dimensional spatio-temporal field from the hidden representations learned by the LSTM. By combining the strengths of LSTM in capturing temporal dependencies and the decoder in reconstructing the field, the SHRED model can effectively generate accurate predictions.

The model's training process involves preparing input sequences, splitting data into training, validation, and test sets, and configuring the architecture's hyperparameters. The model is trained using the training set, and its performance is evaluated on the validation and test sets.

To assess the model's performance, various analyses can be performed. These include evaluating the impact of the time lag variable, which determines the number of previous time steps considered for prediction. Additionally, the model's robustness to noise can be examined by introducing Gaussian noise to the input data. Finally, the influence of the number of sensors used for reconstruction can be studied to understand the trade-off between sensor count and performance.

By understanding the theoretical background of the SHRED model and exploring its capabilities through analysis, we can gain insights into its effectiveness in reconstructing spatio-temporal fields and make informed decisions regarding its application in real-world scenarios.

## Algorithm Implementation and Development 
Here are the steps for the algorithm implementation and development
1. Import the necessary libraries for data processing, model training, and visualization.
2. Define the parameters for the experiment, such as the list of sensor counts or noise levels to test, and initialize an empty list to store the performance results.
3. Iterate over each sensor count or noise level to perform the experiment.
4. Load the dataset or generate synthetic data based on the requirements of the assignment.
5. Split the data into training, validation, and test sets using appropriate methods (e.g., random sampling, time-based splitting).
6. Convert the data into appropriate formats for model training and evaluation, such as PyTorch tensors.
7. Create the training, validation, and test datasets using the processed data.
8. Build and train the model based on the assignment's requirements, using suitable architectures, hyperparameters, and optimization techniques.
9. Monitor the model's performance during training, such as calculating validation errors or other evaluation metrics.
10. Evaluate the trained model on the test dataset and calculate the performance metric of interest (e.g., reconstruction error, accuracy, loss).
11. Store the performance result in the performance_results list.
12. Repeat steps 3 to 12 for each sensor count or noise level.
13. Plot the performance results to visualize the relationship between the experimental variable (sensor count or noise level) and the performance metric.


## Computational Results
![image](https://github.com/ben900912/EE399-HW6/assets/121909443/2249d7e7-a3f3-4d73-8fb9-9d00126d41b5)
> Fig 1: Predicted vs Actual Sea surface temperature

Upon examination of the graph, we observe that the lines representing the actual sensor readings and the predicted sensor readings closely follow each other. This close alignment indicates that the SHRED model has successfully captured the underlying patterns and dynamics of the sea surface temperature data.

The similarity between the actual and predicted sensor readings implies that the SHRED model has effectively learned the spatio-temporal dependencies in the data and can accurately reconstruct the sea surface temperature field. This alignment is particularly significant as it demonstrates the model's ability to capture the complex relationships among different sensor measurements and provide reliable predictions.

The close agreement between the actual and predicted sensor readings indicates the model's accuracy in reproducing the sea surface temperature field. It suggests that the SHRED model can be utilized as a valuable tool for forecasting and analyzing sea surface temperature patterns, enabling better understanding and prediction of environmental conditions.

Overall, the analysis of the Fig1: Predicted vs Actual Sea surface temperature graph highlights the successful performance of the SHRED model in capturing the actual sensor readings and generating accurate predictions. This alignment supports the reliability and effectiveness of the model in reconstructing spatio-temporal fields from sensor measurements.

![image](https://github.com/ben900912/EE399-HW6/assets/121909443/ec0dcafc-0bde-4b4b-8ed1-ecf3fd39c844)
> Fig 2: SHRED model performance vs time lag
From these results, it can be observed that the model's performance varies with different time lag values. Generally, the model achieves relatively good accuracy with lower deviations from the actual values at smaller time lags. As the time lag increases, there may be a slight increase in deviation, but the model's accuracy remains relatively consistent.

It is important to note that the specific performance values obtained may depend on the dataset and the specific problem being addressed. Therefore, further analysis and experimentation may be necessary to determine the optimal time lag value that maximizes the model's performance for a given task.

Overall, the analysis demonstrates the relationship between time lag and the performance of the SHRED model, providing valuable insights into the optimal choice of time lag for accurate predictions of spatio-temporal fields.

![image](https://github.com/ben900912/EE399-HW6/assets/121909443/6dba963a-bfe4-49fa-a455-4f73c6f21dbf)
> Fig 3: SHRED model performance vs Noise Level
As the noise level increases, there should be a degradation in the performance of the SHRED model theoratically. This means that the predicted sensor readings deviate more significantly from the actual sensor readings as the noise level increases. The higher the noise level, the more challenging it becomes for the model to differentiate between the noise and the underlying signal.

however,  the graph shows an unexpected behavior where the model performance initially increases with the noise level and then decreases, it could be due to various factors. Here are a few possible explanations and suggestions

It's important to analyze the specific characteristics of your data, experiment with different models and hyperparameters, and perform multiple runs to obtain more robust results. Adjustments and fine-tuning based on these considerations should help in achieving a more expected and meaningful correlation between noise level and model performance.

![image](https://github.com/ben900912/EE399-HW6/assets/121909443/add4509d-a4ce-4cf8-bafb-532f920b9b3e)
> Fig 4: SHRED model performance vs number of sensors 

The graph shows the performance of the SHRED model in relation to the number of sensors used in the data. The x-axis represents the number of sensors, while the y-axis represents the performance of the model. The performance is measured as the normalized Euclidean distance between the reconstructed data and the ground truth data.

This relationship aligns with expectations since having more sensors provides more coverage and observation points, which leads to a more comprehensive representation of the underlying system or phenomenon. It allows the model to capture finer details and variations in the data, resulting in improved performance.

Overall, the analysis of the graph indicates that increasing the number of sensors positively impacts the performance of the SHRED model, leading to better reconstruction accuracy.
## Summary and Conclusions

In this assignment, we explored the SHRED (SHallow REcurrent Decoder) model, a network architecture for reconstructing high-dimensional spatio-temporal fields from sensor measurements. We downloaded the example code and data for sea-surface temperature and trained the SHRED model using the provided dataset. We then conducted several analyses to evaluate the model's performance under different conditions.

First, we analyzed the performance of the model as a function of the time lag variable, which determines the number of previous time steps considered for prediction. By varying the lag value, we observed how the model's accuracy was affected. Next, we investigated the impact of noise by adding Gaussian noise to the data and assessing the model's robustness under different noise levels. Lastly, we explored the performance as a function of the number of sensors used for reconstruction, considering different sensor counts.

Through these analyses, we gained insights into the behavior of the SHRED model and its sensitivity to time lag, noise, and sensor count. We observed patterns and trends in the model's performance and identified its strengths and limitations in different scenarios.

The SHRED model offers a powerful approach for reconstructing spatio-temporal fields from sensor measurements. In this assignment, we successfully trained and evaluated the SHRED model using sea-surface temperature data. Through various analyses, we assessed the model's performance under different conditions, providing valuable insights.

Our analysis of the time lag variable revealed the influence of temporal context on the model's accuracy. By considering a larger lag, the model had access to more historical information, leading to improved predictions. However, excessively long lags may introduce noise and irrelevant information, negatively impacting performance.

The analysis of noise demonstrated the model's ability to handle noisy input to some extent. The SHRED model exhibited robustness and maintained reasonable accuracy even in the presence of added Gaussian noise. This suggests its potential applicability in real-world scenarios where sensor measurements may be subject to various sources of noise.

Furthermore, the analysis of the number of sensors provided insights into the trade-off between sensor count and performance. We observed that increasing the number of sensors generally led to improved accuracy in reconstructing the spatio-temporal field. However, there may be diminishing returns beyond a certain point, and the additional sensors may introduce complexities in data acquisition and processing.

Overall, the SHRED model demonstrated its effectiveness in reconstructing spatio-temporal fields and showcased its potential for various applications in environmental monitoring, climate studies, and beyond. Further investigations and optimizations can be pursued to enhance its performance and adaptability to specific use cases.

