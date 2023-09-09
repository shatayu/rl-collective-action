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

It will print a message upon completion like below. The file name will differ slightly. This is the model trained:

```
Final checkpoint (reward function: sum): models/model_sum_checkpoint_000001_20230909131920478766
```

# Evaluating Model

To evaluate a model, run the below command. Replace the model checkpoint and the reward function with the ones applicable:

```
python eval_model.py model_sum_checkpoint_000001_20230909131920478766 sum
```

It will print a message upon completion like below. The file name will differ slightly. These are the results of the model playing 10,000 games.

```
Results saved in: results/results_sum_20230909132015823385.pkl
```

**Before proceeding, repeat the above steps for the prop model. You should end up with two new files in results, `results_sum_xxx.pkl` and `results_prop_xxx.pkl` where `xxx` is some timestamp. There are also two existing files in `results`. These are the results from my original run.**

# Analyzing Results

1. Please open [the Google Colab notebook](https://colab.research.google.com/drive/11mIDYlcyS_HlXv5UrVmvVIMZUxGKoDSc#scrollTo=qLOyYJU8xJdk)

2. In the `root` folder, upload your results. There should be two files with names similar to: `results_sum_20230909132015823385.pkl` and `results_prop_20230513121319691750.pkl`. These exact files are also in the `results` folder of the GitHub file. These are the results from my original run.

3. Update the `SUM_RESULTS` and `PROP_RESULTS` constants with the file paths of your results files. 

* Do this by right clicking `results_sum_20230909132015823385.pkl` and clicking `"Copy path"`, then pasting the value for `SUM_RESULTS`. Repeat this for `results_prop_20230513121319691750.pkl`. Note that your file names may differ slightly, but they should still take the form of `results_sum_xxx.pkl` and `results_prop_xxx.pkl`

4. (OPTIONAL) In the root folder, create a folder called `montserrat`. Upload all of the files in the `montserrat` folder in this GitHub repo to that folder.

5. Run all cells.