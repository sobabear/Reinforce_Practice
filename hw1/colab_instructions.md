## Running the homework on Colab

If you are getting installation issues or don't have compute, follow these steps to run the homework on Colab.

1. Upload the project folder to Google Drive
2. Go to [https://colab.research.google.com](https://colab.research.google.com/) and create a new notebook
3. In a new cell, run
	```
	from google.colab import drive
	drive.mount('/content/gdrive')
	```
4. Enter your credentials
5. Navigate to the location of your folder with terminal commands (cd and ls)
6. Install the requirements
	```
	!pip install -r requirements.txt
	```
7. Install the module
	```
	!pip install -e .
	```
8. Locate the tab called "Files" on the left and navigate to your project.
9. Click on the python files to edit them.
10. For an example script to run your code, try:
	```
	!python gcb6206/scripts/run_hw1.py \
		--expert_policy_file gcb6206/policies/experts/Ant.pkl \
		--env_name Ant-v4 --exp_name bc_ant --n_iter 1 \
		--expert_data gcb6206/expert_data/expert_data_Ant-v4.pkl \
		--video_log_freq -1
	```
