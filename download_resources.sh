# Clean house
rm *.sqlite
rm -rf toolbench.egg-info
rm -rf cache/
rm -rf data/


# Prepare test data folder
pip install gdown==4.7.1
gdown 16nMO_bIRS9ZIC3xs4FqjJ_Wwk0jwC4ph
tar xf data.tar
rm data.tar


# Download resources for code as polices environment
export PIP_CACHE_DIR=cache/
cd evaluator/code_as_policies_env

rm -rf ur5e/ robotiq_2f_85/ bowl/
gdown --id 1Cc_fDSBL6QiDvNT4dpfAEbhbALSVoWcc
gdown --id 1yOMEm-Zp_DL3nItG9RozPeJAmeOldekX
gdown --id 1GsqNLhEl9dd4Mc3BM0dX3MibOI1FVWNM
unzip ur5e.zip
unzip robotiq_2f_85.zip
unzip bowl.zip
rm robotiq_2f_85.zip bowl.zip ur5e.zip

cd ..


# Download Webshop Env
export JAVA_HOME=
rm -rf webshop
pip install typing_extensions==4.4.0
git clone https://github.com/princeton-nlp/webshop.git webshop
cd webshop
./setup.sh -d small
cd ../..


## Downgrade faiss, because latest faiss==1.7.3 is buggy and `webshop` installed it.
conda uninstall faiss-cpu
conda install six numpy
