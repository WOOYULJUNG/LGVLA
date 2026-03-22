from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

print("loading config...")
config = _config.get_config("pi05_droid")

print("downloading checkpoint if needed...")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")
print("checkpoint_dir:", checkpoint_dir)

print("creating trained policy...")
policy = policy_config.create_trained_policy(config, checkpoint_dir)

print("SUCCESS: policy loaded")
