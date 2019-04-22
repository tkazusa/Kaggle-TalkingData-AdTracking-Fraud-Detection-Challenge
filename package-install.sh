sudo -u ec2-user -i <<'EOF'

# This will affect only the Jupyter kernel called "conda_python3".
source activate python3
# Replace myPackage with the name of the package you want to install.
conda install -y -q -c conda-forge lightgbm
conda install -y -q -c conda-forge gensim
conda update -y pandas

source deactivate
