"""Invoke the real Lambda handler locally against the caller's selected weights."""
import argparse
import base64
from io import BytesIO
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="State dict exported by the training command")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", type=Path, help="Image to classify")
    source.add_argument("--draw-circle", action="store_true", help="Use a freshly drawn synthetic circle")
    args = parser.parse_args()
    if args.model is not None:
        if not args.model.is_file():
            parser.error("The requested model file is unavailable.")
        os.environ["MODEL_PATH"] = str(args.model)
    if args.image is None:
        from PIL import Image, ImageDraw
        sample = Image.new("L", (28, 28))
        ImageDraw.Draw(sample).ellipse((4, 4, 24, 24), outline=255, width=2)
        out = BytesIO()
        sample.save(out, format="PNG")
        image = out.getvalue()
    else:
        image = args.image.read_bytes()
    from inference import handler
    event = {"body": json.dumps({"b64": base64.b64encode(image).decode("ascii")})}
    response = handler(event, None)
    print(json.dumps(response))
    if response["statusCode"] != 200:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
