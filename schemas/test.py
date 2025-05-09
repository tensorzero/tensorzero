import toml
import json


# Load the TOML file
with open("tensorzero.toml", "r") as toml_file:
  toml_data = toml.load(toml_file)


# Convert the TOML data to JSON
with open("tensorzero.config.json", "w") as json_file:
  json.dump(toml_data, json_file, indent=2)



print("TOML data has been converted to JSON!")
