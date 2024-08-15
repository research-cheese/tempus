# ========================================
# Download model checkpoint
# ========================================
mkdir model_checkpoint
curl https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -o model_checkpoint/sam_vit_b_01ec64.pth



# ========================================
# Download environments
# ========================================
# mkdir environments

# # Abandoned Park -- 1.8.1
# curl https://github.com/microsoft/AirSim/releases/download/v1.8.1-windows/AbandonedPark.zip -o .\environments\AbandonedPark.zip

# # City Environ -- 1.8.1
# curl https://github.com/microsoft/AirSim/releases/download/v1.8.1-windows/CityEnviron.zip.001 -o .\environments\CityEnviron.zip.001
# curl https://github.com/microsoft/AirSim/releases/download/v1.8.1-windows/CityEnviron.zip.002 -o .\environments\CityEnviron.zip.002



# ========================================
# Install dependencies
# ========================================
pip install -r requirements.txt

# Install GroundingDINO
git clone https://github.com/IDEA-Research/GroundingDINO.git third_party/GroundingDINO
pip install -e ./third_party//GroundingDINO