2. TextBased Analytical Report

 Data Analysis Key Findings

   Data Preparation (AirPassengers Dataset):
       The `AirPassengers` dataset was loaded, preprocessed, and transformed from a floatbased time column to a `DatetimeIndex` with 144 monthly entries spanning from January 1949 to December 1960.
       Initial visualization showed an upward trend and increasing variance, indicating clear nonstationarity.
       A natural logarithm transformation was applied to stabilize the variance, making the fluctuations more consistent over time.
       Firstorder differencing (lag 1) was applied to remove the trend. An Augmented DickeyFuller (ADF) test on the firstdifferenced series yielded a pvalue of `0.0711`, suggesting that the series was still nonstationary, likely due to a persistent seasonal component.
       Subsequent seasonal differencing (lag 12, appropriate for monthly data) successfully achieved stationarity, with the ADF test producing a pvalue of `0.0002`, well below the 0.05 significance level.
       The stationary series (`passengers_diff_seasonal`) was then scaled using `MinMaxScaler` to a range of (0, 1). The scaler object was preserved for inverse transformations.
       Feature engineering involved creating 12 lagged versions of the `passengers_scaled` variable to capture past dependencies (a common practice for sequence models), along with `year` and `month` features to help the models account for remaining seasonality and longterm patterns.
       The data was chronologically split into training (70%), validation (15%), and test (15%) sets, ensuring no future information leaked into past observations. This resulted in: `train_data` (83 samples), `val_data` (17 samples), and `test_data` (19 samples).

   SARIMA Benchmark Model Performance:
       A Seasonal AutoRegressive Integrated Moving Average (SARIMA) model with `order=(1,1,1)` and `seasonal_order=(1,1,1,12)` was chosen as the benchmark, reflecting the observed nonseasonal (d=1) and seasonal (D=1) differencing. The (p, q) and (P, Q) orders were chosen as a reasonable starting point.
       The model was trained separately for validation and test periods, using the logtransformed series to align with the preprocessing steps.
       A `ConvergenceWarning` was occasionally observed during model fitting, indicating potential issues with optimization or model complexity, but predictions were still generated.

   Attentionbased EncoderDecoder Model Development:
       An EncoderDecoder sequencetosequence deep learning model was implemented using LSTM layers in TensorFlow/Keras.
       Hyperparameters: `n_steps_in = 12` (encoder lookback window), `n_steps_out = 12` (decoder forecast horizon), `n_features = 3` (representing `passengers_scaled`, `year`, `month` per timestep), and `latent_dim = 128` (LSTM state dimensionality).
       Architecture:
           Encoder: An LSTM layer taking input sequences of shape `(None, n_steps_in, n_features)` and returning its last hidden state (`state_h`), cell state (`state_c`), and outputs for each timestep (`encoder_outputs`).
           Decoder: The encoder's final hidden state (`state_h`) was repeated `n_steps_out` times to serve as input to the decoder. A separate LSTM layer was initialized with the encoder's final states (`state_h`, `state_c`) and produced `decoder_outputs` for each forecast step.
           SelfAttention Mechanism: A Keras `Attention` layer was incorporated, computing attention scores between `decoder_outputs` (query) and `encoder_outputs` (value/key). The resulting context vector was concatenated with `decoder_outputs`.
           Output Layer: A `TimeDistributed(Dense(1, activation='linear'))` layer processed the combined context to generate the final `n_steps_out` predictions.
       Model Summary: The combined model had a total of 199,425 trainable parameters.
       Hyperparameter Tuning Strategy: For this project, a direct hyperparameter tuning loop using rolling crossvalidation was not explicitly implemented due to time constraints but a fixed set of hyperparameters for the attention model (latent_dim, epochs, batch_size) were chosen as a reasonable starting point. Early stopping based on validation loss was used to prevent overfitting and implicitly manage the number of training epochs.

   Attentionbased EncoderDecoder Model Performance:
       Data for the neural network was structured into input sequences (`X_enc_all`) of shape `(108, 12, 3)` and target sequences (`y_dec_all`) of shape `(108, 12, 1)`. These were split into NN training (75 samples), validation (16 samples), and test (17 samples) sets.
       The model was compiled with the Adam optimizer and Mean Squared Error (`mse`) loss function.
       Training utilized `EarlyStopping` with `patience=10` and `ModelCheckpoint` to save the best weights based on validation loss.
       A custom `inverse_transform_single_value` function was developed to accurately revert the scaled, differenced, and logtransformed predictions and actuals back to the original passenger count scale. This function meticulously reverses each preprocessing step.

 Comparative Performance Analysis

Performance Metrics (RMSE, MAE, MAPE)

| Model | Set | Horizon | RMSE (Lower is Better) | MAE (Lower is Better) | MAPE (Lower is Better) | 
| : | : | : | : | : | : | 
| SARIMA | Validation | h=1 | 11.21 | 11.21 | 3.30% |
| SARIMA | Validation | h=12 | 32.56 | 30.19 | 8.27% |
| NN Attention | Validation | h=1 | 7.75 | 6.42 | 1.61% |
| NN Attention | Validation | h=12 | 15.22 | 12.67 | 3.06% |
| SARIMA | Test | h=1 | 20.04 | 20.04 | 4.25% |
| SARIMA | Test | h=12 | 14.05 | 9.97 | 2.26% |
| NN Attention | Test | h=1 | 15.43 | 12.94 | 3.15% |
| NN Attention | Test | h=12 | 20.69 | 15.26 | 3.39% |

 3. Textual Description and Interpretation of Learned Attention Mechanism Weights

The heatmaps visualize the attention weights for three selected test samples. The xaxis represents the encoder input timesteps, from `t12` (past 12 months, index 0) to `t1` (past 1 month, index 11). The yaxis represents the decoder output timesteps, from `t` (current month, index 0) to `t+11` (future 11 months, index 11).

Observations:

1.  Diagonal Pattern (Recency Bias): For predictions closer to the current time (`t`, `t+1`, `t+2`), the model tends to place higher attention on the most recent encoder input timesteps (`t1`, `t2`). This is visible as a diagonal line of slightly higher intensity in the topright corner of the heatmaps, although it's not sharply defined.

2.  Weak but Persistent Seasonal Pattern: While not overwhelmingly strong, there appears to be a subtle tendency for the model to attend to timesteps 12 months prior (index 0 on the xaxis) for certain predictions. This suggests a recognition of annual seasonality, but it's not the dominant pattern.

3.  Diffuse Attention: Overall, the attention weights are quite diffused across many past timesteps. This indicates that the model considers a broad range of past information rather than focusing sharply on one or two specific points. The color intensity is relatively uniform across much of the heatmap, implying that no single past month strongly dictates any single future prediction.

4.  Lack of Strong Localized Attention: Unlike models that might heavily focus on the exact same month in the previous year for a given prediction (e.g., `t` attends strongly to `t12`), this model's attention is distributed. This might be a consequence of the strong differencing and scaling applied, which removes much of the direct linear and seasonal dependencies, forcing the model to find more complex, nonlinear relationships across the entire input sequence.

Interpretation:

The attention mechanism in this model seems to implement a form of 'soft' or 'diffuse' attention. Instead of identifying a few critical past timesteps, it appears to aggregate information from the entire lookback window. This could be beneficial in complex time series where dependencies are not always clearcut or singlepoint driven. The slightly elevated attention on recent past values is intuitive, as the immediate past usually holds significant predictive power. The faint seasonal hints suggest the attention mechanism might be weakly capturing the remaining seasonal information after differencing, or it's combining with the timebased features (`year`, `month`) to infer seasonality indirectly.

It's important to note that the model is making predictions on the scaled, seasonally differenced, and firstorder differenced logtransformed data. This preprocessing significantly alters the nature of the time series, and therefore, the attention patterns might reflect the more subtle relationships within this transformed space rather than direct raw data patterns.

 4. Summary of Project Findings, Comparison, and Discussion

 Project Findings:

   Data Preparation: The `AirPassengers` dataset required significant preprocessing including log transformation, firstorder differencing, and seasonal differencing to achieve stationarity, which is crucial for traditional time series models like SARIMA and beneficial for deep learning models.
   Benchmark Model: The SARIMA model provided a solid baseline, capturing both trend and seasonality in the logtransformed data.
   Attentionbased Model: An EncoderDecoder LSTM with selfattention was successfully built and trained to forecast the transformed passenger data.
   Performance: Both models were evaluated using RMSE, MAE, and MAPE for short (h=1) and medium (h=12) horizons on validation and test sets.

 Comparison of Model Performances:

Looking at the performance tables, the Attentionbased EncoderDecoder model generally outperformed the SARIMA benchmark model on the validation set, especially for the shortterm forecast (h=1) (NN MAPE: 1.61% vs SARIMA MAPE: 3.30%). For the validation set, the NN model consistently showed lower RMSE, MAE, and MAPE across both horizons. This indicates the potential of deep learning models to capture more complex patterns when enough data is available and tuned properly.

However, on the test set, the SARIMA model demonstrated superior performance for the h=12 horizon (SARIMA MAPE: 2.26% vs NN MAPE: 3.39%), while the NN Attention model performed slightly better for h=1 (NN MAPE: 3.15% vs SARIMA MAPE: 4.25%). The SARIMA model's relative strength on the test set for longer horizons might suggest a more robust generalization to unseen data under certain conditions, or that the deep learning model might be more sensitive to the limited size of the dataset for training after sequence creation, or require more extensive hyperparameter tuning.

 Discussion of Insights:

   Preprocessing Impact: The extensive preprocessing (log transformation, double differencing, scaling) was critical for stabilizing the time series, making it amenable to both SARIMA and neural network models. However, this also means the models are learning patterns in a highly transformed space, which complicates direct interpretability.
   Attention Mechanism Insights: The attention weight analysis revealed a diffuse attention pattern, indicating that the model considers a broad range of past timesteps rather than focusing intensely on a few. A subtle recency bias was observed for immediate forecasts, and a weak seasonal pattern suggested some recognition of annual dependencies even after explicit seasonal differencing. This diffuse nature is likely a direct consequence of the aggressive differencing, which removed strong, obvious dependencies. The attention mechanism here acts more as an aggregator of diverse past information rather than a highlighter of singular critical events.
   Model Complexity vs. Data Size: The attentionbased deep learning model, while powerful, requires more data and careful tuning. Its performance variability between validation and test sets (especially for h=12) suggests it might be prone to overfitting or could benefit from more data, more robust time series crossvalidation strategies (like rolling origin), or further hyperparameter optimization.
   Future Work: Next steps would include implementing a formal rolling origin crossvalidation strategy for hyperparameter tuning of the attention model, exploring different deep learning architectures (e.g., Transformers), and experimenting with alternative preprocessing techniques to see how they influence both predictive performance and attention patterns. It would also be insightful to compare attention mechanisms directly on raw or less transformed data to observe if more localized attention patterns emerge.
