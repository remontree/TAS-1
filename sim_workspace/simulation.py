import json
from core.rocket import *

JSON_PATH = "./rocket.json"

# JSON 파일 읽기
with open(JSON_PATH, "r", encoding="utf-8") as f:
    rocket_config = json.load(f)

rocket1 = Rocket(rocket_config["TAS-I"])

rocket1.ignite()
