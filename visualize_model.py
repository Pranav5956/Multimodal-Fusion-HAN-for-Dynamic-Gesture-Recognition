from torchinfo import summary
from models.MFHAN import MFHAN
import yaml

batch_size = 32
sample_size = 8

depth_feature_dim = (96, 128, 4)
skeleton_feature_dim = (22, 6)

config = yaml.load(open("configs/config.yml", 'r'), yaml.SafeLoader)

model = MFHAN(config["num_classes"], config["model"])

summary(model, ((batch_size, sample_size, *skeleton_feature_dim),
        (batch_size, sample_size, *depth_feature_dim)), depth=2)
