# Introduction

This is a very simple walk through on how to replicate the world model paper. 

#### Trained Model
![Alt Text](visualize.gif)


#### Setup

Create a conda environment

```
conda create --name worldmodel python=3.5.4
```

Activate the Environment
```
conda activate worldmodel
```

Install dependencies
```
conda install -c anaconda pip

conda install tensorflow-gpu=1.8.0

conda install -c theochem cma

conda install -c anaconda mpi4py

conda install -c anaconda jupyter

conda install -c akode gym

pip install gym==0.9.4

pip install gym[box2d]==0.9.4
```

Additional Dependencies

These dependencies are required to avoid an issue with earlier versions of scipy and pillow

```
pip install cython

conda install -c anaconda scipy=1.1.0

conda install -c anaconda pillow

conda install -c conda-forge pyglet
```



#### Extract data to train the VAE and MDM-RNN
Modify extract.bash to disable GPU. Since we are running 64 instances parallely if we do not do this step, each 
process will try to create the model on the GPU. As they cannot share memory it will run out very quickly. 
To do so add the following line at the beginning of extract.bash

```
set CUDA_VISIBLE_DEVICES=-1
```

If running on a device with a lower number of cores we can still generate all the samples.
In order to do se we need to change the bash script by reducing the number of parallel threads to say 8.
We could then use a wait command and then rerun the loop as many times as we want.


```
for i in `seq 1 8`;
do
  echo worker $i
  # on cloud:
  xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python extract.py &
  # on macbook for debugging:
  #python extract.py &
  sleep 1.0
done
```

If generating a lower number of files you would then need to change the file vae_train.py to reduce the number of files 
when loading your dataset.

Finally run the command
```
sh ./extract.bash
```


#### Train VAE and MDM-RNN
To do so execute the command
```
sh ./gpu_jobs.bash
```

For this script we do not want to disable the GPU. It will train the VAE and then train the Mixture Density Network.

#### Train Controller

Copy the trained files to the following directories

```
cp vae.json vae/

cp initial_z.json initial_z/

cp rnn.json rnn/
```

Again we will be running on 64 threads so disable the GPU by add the following line to gce_train.bash.
```
set CUDA_VISIBLE_DEVICES=-1
```

Train the Controller
```
sh ./gce_train.bash
```


#### Visualize the results
Run the Jupyter notebook plot_training_progress.ipynb

You can then visualize the trained model by running the following command

```
python model.py render log/carracing.cma.16.64.best.json
```

You can also save a video of the agent playing the game by adding a gym Monitor Wrapper to the environment in env.py.
In make_env you can add the following line.
```
env = gym.wrappers.monitoring.Monitor(env, 'video', force=True)
```


Visualizing training
![Alt Text](carracing.cma.16.64.png)

