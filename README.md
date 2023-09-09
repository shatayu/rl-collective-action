# Setup

Run the below commands:

```
git clone https://github.com/shatayu/rl-collective-action.git
cd rl-collective-action
pip install -r requirements.txt
```

# Training Model

The valid reward functions are `sum` and `prop`. The example below uses `sum` - replace it with `prop` for a prop model

Run the below commands:

`python train_model.py sum`

It will print a message upon completion like below. This is the model trained:

```
Final checkpoint (reward function: sum): models/model_sum_checkpoint_000001_20230909131920478766
```

# Evaluating Model

To evaluate a model, run the below command. Replace the model checkpoint and the reward function with the ones applicable

```
python eval_model.py model_sum_checkpoint_000001_20230909131920478766 sum
```

It will print a message upon completion like below: These are the results of the model playing 10,000 games.

```
Results saved in: results/results_sum_20230909132015823385.pkl
```

Before proceeding, repeat the above steps for the prop model.

# Analyzing Results

1. Please open [the Google Colab notebook](https://colab.research.google.com/drive/11mIDYlcyS_HlXv5UrVmvVIMZUxGKoDSc#scrollTo=qLOyYJU8xJdk)

2. In the `root` folder, upload your results. There should be two files with names similar to: `results_sum_20230909132015823385.pkl` and `/root/results_prop_20230513121319691750.pkl`. These exact files are also in the `results` folder of the GitHub file.

3. (OPTIONAL) In the root folder, create a folder called `montserrat`. Upload all of the files in the `montserrat` folder in this GitHub repo to that folder.

4. Run all cells