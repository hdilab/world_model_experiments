export CUDA_VISIBLE_DEVICES="";

xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python generate_data.py;

unset CUDA_VISIBLE_DEVICES;

python train_vae.py;

python series.py;

python train_rnn.py;

export CUDA_VISIBLE_DEVICES="";

xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python train_v_m_c.py;
