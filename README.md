# Steam Sentiment Analysis
Binary classifier that uses a neural network (and Vader) to predict the sentiment of Steam video game reviews.

# Getting the dataset

1) Download the Steam game reviews dataset from this link: https://www.kaggle.com/smeeeow/steam-game-reviews

2) Unzip the folder game_rvw_csvs to the project directory.

# How to install

Cd into the project directory and run the following commands:
 - conda create -n steam_sa python=3.8  
 - conda activate steam_sa              
 - pip install -r steam_sa_tf2.txt
 - conda install tensorflow-gpu==2.3.0

Next, search for "Edit the System environment variables" and edit the PATH variable under User Variables to include paths to
 - The Cuda toolkit (usually C:\Users\YOURUSERNAME\Anaconda3\pkgs\cudatoolkit-[version-number]\Library\bin)
 - The CuDNN (usually C:\Users\YOURUSERNAME\Anaconda3\pkgs\cudnn-[version-number]-cuda10.0_0\Library\bin)

For TF 2.3.0, you will need CUDA 10.1 and CUDNN 7.6.5.

Then restart the terminal.

# How to run

Cd into the project directory and run the following command:

python main_app.py
