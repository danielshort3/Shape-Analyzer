# local_test.py
import inference, base64, json, pathlib
img_b64 = base64.b64encode(pathlib.Path("test.png").read_bytes()).decode()
event   = {"body": json.dumps({"b64": img_b64})}
print(inference.handler(event, None))