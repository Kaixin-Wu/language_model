# theano device
device=gpu0
script=build_model.py

THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn,gpuarray.preallocate=6000 \
nohup python -u $script > nohup.log &
