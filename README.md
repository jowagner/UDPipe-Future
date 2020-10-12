# UDPipe-Future

UDPipe-Future is a prototype for UDPipe 2.0. The prototype consists of tagging
and parsing and is purely in Python. It participated in CoNLL 2018 UD Shared
Task and was one of three winners.

The `master` branch of this repository contains post-Shared-Task improvements.
The original system submitted to the Shared Task can be found in `shared-task`
branch.

UDPipe 2.0 is currently being developed in the
[udpipe-2 branch](https://github.com/ufal/udpipe/tree/udpipe-2) of the
[UDPipe repository](https://github.com/ufal/udpipe) and a first
[release](https://github.com/ufal/udpipe/releases) is expected in Q4 2020.

# Installation

## New to PIP and virtualenv?

This can be skipped if you already have pip and virtualenv.

```
wget https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py --user
pip3 install --user virtualenv
```

Append to `.bashrc` and re-login:
```
# for our own pip and virtualenv:
export PATH=$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.local/lib
```

## UDPipe-future Dependencies

### Using Virtualenv

```
git clone git@github.com:CoNLL-UD-2018/UDPipe-Future.git
cd UDPipe-Future/
virtualenv -p /usr/bin/python3.7 venv-udpf
vi venv-udpf/bin/activate
```

Note: Some of our experiments were run on a second cluster using Python 3.6 instead of 3.7.

Add `LD_LIBRARY_PATH` for a recent CUDA with CuDNN
that works with TensorFlow 1.14 to `bin/activate`,
e.g.
on the ADAPT clusters:
```
LD_LIBRARY_PATH=/home/support/nvidia/cuda10/lib64:/home/support/nvidia/cudnn/cuda10_cudnn7_7.5/lib64:"$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH
```

As in the above configuration, we used CUDA 10.0 and matching CuDNN.
TODO: Why do we not use `UDPIPE_FUTURE_LIB_PATH` in `config/locations.sh`?

```
source venv-udpf/bin/activate
pip install tensorflow-gpu==1.14
pip install cython
pip install git+https://github.com/andersjo/dependency_decoding
```

TODO: Could we use the script provided by udpipe-future? What is the `venv` module that it uses?

### Using Conda

On ICHEC, conda needs to be loaded first:
```
module load conda/2
```

Then:
```
conda create --name udpf python=3.7 \
    tensorflow-gpu==1.14 \
    cython
```

If this is the first time using conda on ICHEC, you need to run
`conda init bash`, re-login, run `conda config --set auto_activate_base false`
and re-login again.

TODO: ICHEC support says to use `source activate udpf` instead, not
requiring initialisation. Test this on next install. (This will also
require adjustments to most of the shell scripts
`mtb-tri-training/scripts/*.sh`.

Then:
```
conda activate udpf
pip install git+https://github.com/andersjo/dependency_decoding
conda deactivate
```

If CUDA and CuDNN libraries are not in your library path already, you need to
set `UDPIPE_FUTURE_LIB_PATH` in `config/locations.sh`, e.g.
on ICHEC:
```
UDPIPE_FUTURE_LIB_PATH="/ichec/packages/cuda/10.0/lib64":"$HOME/cudnn-for-10.0/lib64"
```

## FastText

FastText is recommended for creating external word embeddings for
UDPipe-Future.
The FastText output, however, needs to be converted to the `.npz`
format expected by UDPipe-Future.

### Installation

As of October 2019,
(FastText)[https://fasttext.cc/docs/en/support.html] is installed by
cloning the repository, `cd`-ing into it, running `make` and copying
the `fasttext` binary to a folder in your `PATH`, e.g. `$HOME/.local/bin`.
As the binary is built with `-march=native`, it needs to be built
on a machine supporting all desired CPU features, e.g. AVX.

### Extract tokenised text

For Irish, we observed that the udpipe tokeniser fails to separate neutral
double quotes as they do not occur in the treebank. However, for consistency
with other languages, we do not address this issue here.

We use truecase as UDpipe-future recently added supports both truecase and
we expect the character-based fasttext embeddings to learn the relationship
between lowercase and uppercase letters.

### Train FastText

As Straka et al. (2019), we run fasttext with `-minCount 5 -epoch 10 -neg 10`, e.g.
```
fasttext skipgram -minCount 5 -epoch 10 -neg 10 -input Irish.txt -output model_ga
```

Note that fasttext uses only a few GB of RAM.
FastText can use AVX instructions to speed up training.
An English model takes about 2 1/2 days
to train on wikipedia and common crawl data.
A model for a low-resourced language can finish in a few minutes.


### Conversion to UDPipe-future .npz format

The `.vec` files can be converted with `convert.py` provided with
UDPipe-future.
We assume all FastText embeddings in the `.npz` format for UDPipe-future are
in a single folder with filenames `fasttext-xx.npz` where `xx` is a language code.
As Straka et al. (2019), we limit the vocabulary to the 1 million most frequent
types.

```
for LCODE in ga ug ; do
    echo "== $LCODE =="
    python3 ~/tri-training/UDPipe-Future/embeddings/sources/convert.py \
        --max_words 1000000 model_$LCODE.vec fasttext-$LCODE.npz
done
```

## ELMo For Many Languages

These contextualised word embeddings can substantially improve accuracy
of the parser.

https://github.com/HIT-SCIR/ELMoForManyLangs

```
git clone git@github.com:HIT-SCIR/ELMoForManyLangs.git
```

### Using Virtualenv

```
cd ELMoForManyLangs/
virtualenv -p /usr/bin/python3.7 venv-efml
vi venv-efml/bin/activate
```

Add `LD_LIBRARY_PATH` for CUDA 10.1 and matching CUDNN
to `bin/activate`, e.g. `/usr/local/cuda-10.1/lib64`.

```
source venv-efml/bin/activate
pip install torch torchvision
pip install allennlp
pip install h5py
```

### Using Conda on ICHEC

```
module load conda/2
conda create --name efml python=3.7 h5py
conda activate efml
pip install torch torchvision
pip install allennlp
conda deactivate
```

### Module Installtion not Required

It is not necessary to run `python setup.py install`:
The command `python -m elmoformanylangs test`
in `get-elmo-vectors.sh` work because we `cd`
into the efml folder.

### Models

After extracting the elmoformanylangs model files, the
`config_path` variable in the `config.json` files has
to be adjusted.

```
mkdir ug_model
cd ug_model
unzip ../downloads/175.zip
vi config.json
```

We assume that the elmo configuration and models are in a single
folder and to be able to re-use the same `.json` files
on different systems, we use symlinks:

```
cd
mkdir elmo
cd elmo/
ln -s $HOME/tri-training/ELMoForManyLangs/configs/
ln -s /spinning/$USER/elmo/ga_model/
```

