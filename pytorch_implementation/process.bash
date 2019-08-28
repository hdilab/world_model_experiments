xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python generate_data.py

echo "Completed generation of data"

python train_vae.py

python series.py

python train_rnn.py

echo "Completed VAE and RNN training"

xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python train_v_m_c.py
