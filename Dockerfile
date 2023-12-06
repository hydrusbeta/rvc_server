# Use Nvidia Cuda container base, sync the timezone to GMT, and install necessary package dependencies. Binaries are
# not available for some python packages, so pip must compile them locally. This is why gcc, g++, and python3.9-dev are
# included in the list below. Cuda 11.8 is used instead of 12 for backwards compatibility. Cuda 11.8 supports compute
# capability 3.5 through 9.0.
FROM nvidia/cuda:11.8.0-base-ubuntu20.04
ENV TZ=Etc/GMT
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt update && apt install -y --no-install-recommends \
    ffmpeg \
    gcc \
    git \
    g++ \
    locales \
    python3.9-dev \
    python3.9-venv \
    wget

# Enable UTF-8 so we can download files with Chinese characters in the filename.
RUN sed -i '/^#.* en_US.UTF-8.* /s/^#//' /etc/locale.gen && \
    locale-gen

# Switch to a limited user
ARG LIMITED_USER=luna
RUN useradd --create-home --shell /bin/bash $LIMITED_USER
USER $LIMITED_USER

# Some Docker directives (such as COPY and WORKDIR) and linux command options (such as wget's directory-prefix option)
# do not expand the tilde (~) character to /home/<user>, so define a temporary variable to use instead.
ARG HOME_DIR=/home/$LIMITED_USER

# Another step for enabling UTF-8
ENV LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8

# Download the pretrained Hubert model.
RUN mkdir -p ~/hay_say/temp_downloads/hubert/ && \
    wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt --directory-prefix=$HOME_DIR/hay_say/temp_downloads/hubert/

# Download the RVC pretrained models.
RUN mkdir -p ~/hay_say/temp_downloads/pretrained/ && \
    wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/D32k.pth --directory-prefix=$HOME_DIR/hay_say/temp_downloads/pretrained/ && \
    wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/D40k.pth --directory-prefix=$HOME_DIR/hay_say/temp_downloads/pretrained/ && \
    wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/D48k.pth --directory-prefix=$HOME_DIR/hay_say/temp_downloads/pretrained/ && \
    wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/G32k.pth --directory-prefix=$HOME_DIR/hay_say/temp_downloads/pretrained/ && \
    wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/G40k.pth --directory-prefix=$HOME_DIR/hay_say/temp_downloads/pretrained/ && \
    wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/G48k.pth --directory-prefix=$HOME_DIR/hay_say/temp_downloads/pretrained/ && \
    wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0D32k.pth --directory-prefix=$HOME_DIR/hay_say/temp_downloads/pretrained/ && \
    wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0D40k.pth --directory-prefix=$HOME_DIR/hay_say/temp_downloads/pretrained/ && \
    wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0D48k.pth --directory-prefix=$HOME_DIR/hay_say/temp_downloads/pretrained/ && \
    wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0G32k.pth --directory-prefix=$HOME_DIR/hay_say/temp_downloads/pretrained/ && \
    wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0G40k.pth --directory-prefix=$HOME_DIR/hay_say/temp_downloads/pretrained/ && \
    wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0G48k.pth --directory-prefix=$HOME_DIR/hay_say/temp_downloads/pretrained/

# Download the UVR5 weights
RUN mkdir -p ~/hay_say/temp_downloads/uvr5_weights/ && \
    mkdir -p ~/hay_say/temp_downloads/uvr5_weights/onnx_dereverb_By_FoxJoy/ && \
    wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP2-%E4%BA%BA%E5%A3%B0vocals%2B%E9%9D%9E%E4%BA%BA%E5%A3%B0instrumentals.pth --directory-prefix=$HOME_DIR/hay_say/temp_downloads/uvr5_weights/ && \
    wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP2_all_vocals.pth --directory-prefix=$HOME_DIR/hay_say/temp_downloads/uvr5_weights/ && \
    wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP3_all_vocals.pth --directory-prefix=$HOME_DIR/hay_say/temp_downloads/uvr5_weights/ && \
    wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP5-%E4%B8%BB%E6%97%8B%E5%BE%8B%E4%BA%BA%E5%A3%B0vocals%2B%E5%85%B6%E4%BB%96instrumentals.pth --directory-prefix=$HOME_DIR/hay_say/temp_downloads/uvr5_weights/ && \
    wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP5_only_main_vocal.pth --directory-prefix=$HOME_DIR/hay_say/temp_downloads/uvr5_weights/ && \
    wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/VR-DeEchoAggressive.pth --directory-prefix=$HOME_DIR/hay_say/temp_downloads/uvr5_weights/ && \
    wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/VR-DeEchoDeReverb.pth --directory-prefix=$HOME_DIR/hay_say/temp_downloads/uvr5_weights/ && \
    wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/VR-DeEchoNormal.pth --directory-prefix=$HOME_DIR/hay_say/temp_downloads/uvr5_weights/ && \
    wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/onnx_dereverb_By_FoxJoy/vocals.onnx --directory-prefix=$HOME_DIR/hay_say/temp_downloads/uvr5_weights/onnx_dereverb_By_FoxJoy/

# Create virtual environments for RVC and Hay Say's rvc_server.
RUN python3.9 -m venv ~/hay_say/.venvs/rvc; \
    python3.9 -m venv ~/hay_say/.venvs/rvc_server

# Python virtual environments do not come with wheel, so we must install it. Upgrade pip while
# we're at it to handle modules that use PEP 517.
RUN ~/hay_say/.venvs/rvc/bin/pip install --timeout=300 --no-cache-dir --upgrade pip wheel; \
    ~/hay_say/.venvs/rvc_server/bin/pip install --timeout=300 --no-cache-dir --upgrade pip wheel

# Install all python dependencies for RVC.
# Note: This is done *before* cloning the repository because the dependencies are likely to change less often than the
# RVC code itself. Cloning the repo after installing the requirements helps the Docker cache optimize build time.
# See https://docs.docker.com/build/cache
RUN ~/hay_say/.venvs/rvc/bin/pip install \
    --timeout=300 \
    --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    absl-py==1.4.0 \
    audioread==3.0.0 \
    colorama==0.4.6 \
    Cython==0.29.35 \
    gradio==3.33.1 \
    httpx==0.23.0 \
    fairseq==0.12.2 \
    faiss-cpu==1.7.3 \
    ffmpeg-python==0.2.0 \
    fsspec==2023.5.0 \
    Jinja2==3.1.2 \
    joblib==1.2.0 \
    json5==0.9.14 \
    librosa==0.9.1 \
    llvmlite==0.39.0 \
    Markdown==3.4.3 \
    matplotlib==3.7.1 \
    matplotlib-inline==0.1.6 \
    numba==0.56.4 \
    numpy==1.23.5 \
    onnxruntime-gpu==1.15.0 \
    Pillow==9.5.0 \
    praat-parselmouth==0.4.3 \
    pyasn1==0.5.0 \
    pyasn1-modules==0.3.0 \
    pydub==0.25.1 \
    pyworld==0.3.4 \
    PyYAML==6.0 \
    resampy==0.4.2 \
    scipy==1.9.3 \
    scikit-learn==1.2.2 \
    soundfile==0.12.1 \
    starlette==0.27.0 \
    sympy==1.12 \
    tabulate==0.9.0 \
    tensorboard==2.13.0 \
    tensorboardX==2.6 \
    torchcrepe==0.0.18 \
    tornado==6.3.2 \
    tqdm==4.65.0 \
    uc-micro-py==1.0.2 \
    uvicorn==0.22.0 \
    Werkzeug==2.3.4

# Install the dependencies for the Hay Say interface code.
RUN ~/hay_say/.venvs/rvc_server/bin/pip install --timeout=300 --no-cache-dir \
    hay_say_common==1.0.7 \
    jsonschema==4.19.1

# Clone RVC and checkout a specific commit that is known to work with this Docker file and with Hay Say.
# Important! The RVC code is modified a little later in this Dockerfile. If you update the commit hash here, be sure to
# update the later section too, if needed (e.g. line numbers might change).
RUN git clone -b main --single-branch -q https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI ~/hay_say/rvc
WORKDIR $HOME_DIR/hay_say/rvc
RUN git reset --hard d97767494c83e01083030ebe21f7f4b296e41fab

# Clone the Hay Say interface code
RUN git clone -b database-cache --single-branch -q https://github.com/hydrusbeta/rvc_server ~/hay_say/rvc_server

# Add command line functionality to RVC
RUN git clone -b main --single-branch -q https://github.com/hydrusbeta/rvc_command_line ~/hay_say/rvc_command_line && \
    mv ~/hay_say/rvc_command_line/command_line_interface.py ~/hay_say/rvc/

# Modify RVC's code to add a command-line option for preventing its gradio server from automatically starting
RUN sed -i '60 i\            cmd_opts.commandlinemode,\n' ~/hay_say/rvc/config.py && \
    sed -i '50 i\        parser.add_argument(\n            "--commandlinemode", action="store_true", help="Do not automatically launch the server when importing infer-web.py"\n        )\n' ~/hay_say/rvc/config.py && \
    sed -i '31 i\            self.commandlinemode,' ~/hay_say/rvc/config.py && \
    sed -i '1304,1991 s\^\    \' ~/hay_say/rvc/infer-web.py && \
    sed -i '1304 i\if not config.commandlinemode:\n' ~/hay_say/rvc/infer-web.py

# Create directories that are used by the Hay Say interface code
RUN mkdir -p ~/hay_say/rvc/input/ && \
    mkdir -p ~/hay_say/rvc/output/

# Expose port 6578, the port that Hay Say uses for RVC.
# Also expose port 7865, in case someone wants to use the original RVC UI.
EXPOSE 6578
EXPOSE 7865

# Move the pretrained models to the expected directories.
RUN mv ~/hay_say/temp_downloads/hubert/* ~/hay_say/rvc/ && \
    mv ~/hay_say/temp_downloads/pretrained/* ~/hay_say/rvc/pretrained/ && \
    mv ~/hay_say/temp_downloads/uvr5_weights/* ~/hay_say/rvc/uvr5_weights/

# Execute the Hay Say interface code
CMD ["/bin/sh", "-c", "~/hay_say/.venvs/rvc_server/bin/python ~/hay_say/rvc_server/main.py --cache_implementation file"]