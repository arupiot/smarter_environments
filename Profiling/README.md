#iotdesks\_occupancy\_profiling.py
- Contains a class for iotdesks occupancy profiling
	- Used by train\_and\_test\_occupancy.py
- Method: preprocess\_data():
   - Obtains raw data and formats all the columns in a required manner
   - strips time and chooses one value for a minute (If there are more than one value recorded within a minute, this will drop duplicates)
- Method: feature\_extraction():
	- Feature extraction from time series data
	- Each row contains 5 minutes data, a column for each minute and an overlap of one minute
- Method: return\_pca\_model():
	- Fits pca model for dimensionality reduction, returns data with dimension=2
- Method: return\_gmm\_model():
	- Fits Gaussian Mixture Model and predicts the cluster to which the data row belongs to
- Method: label\_each\_time\_instant():
	- Based on the prediction, labels the time series data that we had initially
- Method: plot\_occupancy\_prediction(): 
	- Plots the data with prediction as plots
- Method: Occupancy_prediction():
	- Call the above methods in the required order for occupancy prediction
- Method: separate\_day\_wise\_data():
	- separates the data (with predictions) into 7 based on the day of week
- Method: kernel\_density\_estimation():
	- Fits kde on each data separately
	- profiling of occupancy done which outputs the probability of presence of a person throughout a day
- Method: occupancy\_profiling():
	- predicts occupancy
	- separates data with prediction based on day of week
	- profiling - estimates probability of occupancy

#train\_and\_test\_occupancy.py
- Method: train\_from\_csv()
   - Train model using data from data\_path+file\_name
   - Outputs occupancy prediction on trained data and profiling for different days of week
- Method: train\_from\_db()
   - Train model using data from database
   - query database with start\_date\_str and end\_date\_str for training
   - Outputs occupancy prediction on trained data and profiling for different days of week
- First block of code:
   - Set database details
- Second block of code:
	- Either train\_from\_csv() or train\_from\_db() can be used depending upon train data requirement
- Third block of code:
	- predict occupancy using the trained model
	- select data from database using test\_start\_date\_str and test\_end\_date\_str
	- Outputs occupancy prediction for the test data
	- Can run this block in a loop to predict 5 minutes once by making changes while querying
   
#iotdesks\_power\_profiling.py
- Contains a class for iotdesks power profiling
	- Used by train\_and\_test\_power.py
- Most of the methods are similar to the ones present in iotdesks\_occupancy\_profiling.py
	- Following method descriptions are done for the methods that are different or not present in iotdesks\_occupancy\_profiling.py
- Method: power\_prediction():
	- preprocesses and extract features
	- pca and gmm are used for first level of prediction - Outputs two clusters
	- Finds cluster that denotes OFF stage. The remaining data is chosen for second level prediction
	- Again pca and gmm are used for the next level of prediction
	- Combine three state clusters and label the data as OFF, hibernation and ON
- Method: kernel\_density\_estimation():
	- Fits kde on each data separately
	- profiling of laptop usage which outputs the probability of each state throughout a day
- Method: power\_profiling():
	- predicts laptop usage - OFF, Hibernation and ON
	- separates data with prediction based on day of week
	- profiling - estimates of laptop usage for three different states


#train\_and\_test\_power.py
- Train model using data from data\_path+file\_name
- Outputs laptop usage prediction on trained data and profiling for different days of week 
- This code can be replicated similar to train\_and\_test\_occupancy.py, for training and testing on real-time data connected to a database

#utils.py
- Contains useful methods for testing data in real time

#Folder: Sample-Results
- Contains sample outputs of training and testing saved as CSVs

#Folder: Sample-Plots
- Contains sample plots of profiling
 
