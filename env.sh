#https://tk233.gitbook.io/notes/ml-rl/setting-up-nvidia-tools/deprecated-setting-up-nvidia-isaac-gym-on-ubuntu-22.04-20.04
# solve libpython3.8.so.1.0 not found issue
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH

# solve "generated code is out of date and must be regenerated with protoc >= 3.19.0" issue
# export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python