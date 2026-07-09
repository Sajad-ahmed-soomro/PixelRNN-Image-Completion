# server.py
import io
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import torch
import torchvision.transforms as T
from model import UNetGenerator

app = FastAPI(title="Pix2Pix Inference")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = os.environ.get("PIX2PIX_CKPT", "checkpoints/ckpt_epoch_200.pth")
IMG_SIZE = 256

transform_in = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

# Load model with graceful fallback
model = UNetGenerator(in_channels=3, out_channels=3).to(DEVICE)
if not os.path.exists(CKPT_PATH):
    # if no checkpoint, keep model but warn (you probably want to train or supply ckpt)
    print(f"Warning: checkpoint not found at {CKPT_PATH}. Model has random weights.")
else:
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['G'])
model.eval()

def tensor_to_png_bytes(tensor):
    t = tensor.clamp(-1, 1).detach().cpu()
    t = (t + 1.0) / 2.0
    t = (t * 255).permute(1,2,0).byte().numpy()
    pil = Image.fromarray(t)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return buf

@app.post("/api/infer")
async def infer(file: UploadFile = File(...)):
    if file.content_type not in ("image/png", "image/jpeg", "image/jpg"):
        raise HTTPException(status_code=400, detail="Unsupported file type")
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    # If the user draws a single-channel sketch (black-white), convert to RGB already done above.
    inp = transform_in(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(inp)
        out_img = out[0]

    buf = tensor_to_png_bytes(out_img)
    return StreamingResponse(buf, media_type="image/png")
