# VideoGen_VAE
Video generation pretraining experiments repo
For training:
pip install git+https://github.com/ollanoinc/hyvideo.git
pip install flash-attn
sudo apt-get update
sudo apt-get install libglvnd-dev libgl1 -y
sudo apt-get install libglib2.0-0 -y
pip install -r requirements.txt
NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=2 train.py --deepspeed --config examples/ltx_video.toml 

ffmpeg -i generated_video_genuine.mp4 -c:v libvpx-vp9 -b:v 1M -c:a libopus generated_video_genuine.webm


For personalized dataloading:
data_pipline.py
main_main.py
saving_ltx_models.py


pip install xformers # experimental


OWN TRY:
pip install opencv-python-headless
pip install git+https://github.com/lucidrains/video-diffusion-pytorch.git
