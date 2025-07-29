#!/usr/bin/env python
# shapes_pipeline.py
"""
Pipeline to train a ResNet‑18–based model that (1) classifies a drawing as
circle / triangle / square / hexagon / octagon and (2) predicts its
“regularity” score (0‑100 %).  Synthetic drawings + Google QuickDraw.

The file is organised top‑to‑bottom like this:
  1. CONFIG  – all tunables in one dataclass
  2. HELPERS – fast polygon / circle generator and misc utilities
  3. DATA    – Dataset objects, transforms, loaders
  4. MODEL   – ShapeNet backbone + Lightning wrapper
  5. TRAIN   – Trainer, callbacks, entry‑point
  6. INFER   – Minimal interactive canvas demo (optional)
"""

## 1. CONFIG
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np, functools, math, random, json, requests
import torch, torch.nn as nn
import lightning.pytorch as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
import torchmetrics
from torchvision import transforms as T
import torchvision.transforms.v2 as Tv2
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

torch.backends.cuda.matmul.fp32_precision = "tf32"      # matmuls
torch.backends.cudnn.conv.fp32_precision  = "tf32"      # convolutions

@dataclass
class CFG:
    # ----- data ---------------------------------------------------------
    CLASSES   : tuple[str, ...] = ("circle", "triangle", "square",
                                   "hexagon", "octagon")
    SIDE_MAP  : dict[str, int]  = field(
        default_factory=lambda: {
            "circle": 0, "triangle": 3, "square": 4,
            "hexagon": 6, "octagon": 8
        }
    )
    OUT_RES         : int   = 256
    DRAW_RES        : int   = 64
    STROKE_PX       : int   = 10
    TRAIN_FRAC      : float = 0.9
    N_BINS          : int   = 10
    N_PER_SHAPE     : int   = 50_000
    HOLD_QD_PER_SHAPE: int  = 10_000
    RNG_SEED        : int   = 42

    # optimiser / schedule ----------------------------------------------
    MAX_EPOCHS : int   = 100
    BATCH_SIZE : int   = 1024
    LR         : float = 3e-4
    WD         : float = 1e-4
    PRECISION  : str   = "16-mixed"

    # file paths ---------------------------------------------------------
    DATA_PATH  : Path  = Path("shapes_data.npz")
    HOLD_PATH  : Path  = Path("qd_holdout.npz")
    CACHE_DIR  : Path  = Path("qd_cache")
    CKPT_DIR   : Path  = Path("checkpoints")

cfg = CFG()                # single importable object

rng = np.random.default_rng(cfg.RNG_SEED)   # global RNG


## 2. HELPERS  – shape synthesis & utilities
from skimage.draw import polygon_perimeter
from skimage.transform import resize as sk_resize

@functools.lru_cache(maxsize=None)
def _unit_vertices(n_sides:int) -> np.ndarray:
    """Regular polygon / surrogate circle centred on origin."""
    ang = np.linspace(0,2*np.pi,n_sides,endpoint=False,dtype=np.float32)
    return np.stack([np.cos(ang),np.sin(ang)],1)          # (N,2)

def _curve_amp(reg_pct:float,
               amp_min:float=0.0,amp_max:float=0.35,gamma:float=0.60):
    frac = 1.0 - reg_pct/100
    return amp_min + (amp_max-amp_min)*(frac**gamma)

def _chaikin(pts:np.ndarray,n_iter:int=1)->np.ndarray:
    for _ in range(n_iter):
        Q = 0.75*pts[:-1]+0.25*pts[1:]; R = 0.25*pts[:-1]+0.75*pts[1:]
        pts = np.vstack([Q.T,R.T]).T.reshape(-1,2)
    return pts

def _sample_vertices(n_sides:int,reg_pct:float,n_circle_pts:int=128):
    amp = _curve_amp(reg_pct)
    # —— circle branch ————————————————————————————
    if n_sides==0:
        base = _unit_vertices(n_circle_pts)
        if reg_pct>=50: return base
        k = 1+(50-reg_pct)/20
        sx,sy = 1+rng.normal(0,amp*1.5*k,2)
        shear = rng.normal(0,amp*0.8*k)
        affine = np.array([[sx,shear],[0,sy]],np.float32)
        verts  = (affine@base.T).T
        n_lobes=rng.integers(1,4); phase=rng.uniform(0,2*np.pi)
        ripple=1+amp*0.6*k*np.sin(np.linspace(phase,phase+2*np.pi*n_lobes,
                                              n_circle_pts,dtype=np.float32))
        verts*=ripple[:,None]
        return _chaikin(verts,1)
    # —— polygon branch ————————————————————————————
    base=_unit_vertices(n_sides)
    radial=rng.normal(1.0,amp,n_sides).astype(np.float32)
    angle=rng.normal(0.0,amp*0.6,n_sides).astype(np.float32)
    c,s=np.cos(angle),np.sin(angle)
    rot=np.stack([c,s,-s,c],-1).reshape(-1,2,2)
    verts=(rot@base[:,:,None]).squeeze(-1)*radial[:,None]
    if reg_pct>=50: return verts

    bulged=[]
    bulge=amp*0.8*min(1.0,n_sides/6)
    for i in range(n_sides):
        v0,v1=verts[i],verts[(i+1)%n_sides]
        mid=(v0+v1)/2
        nrm=np.array([v1[1]-v0[1],v0[0]-v1[0]],np.float32)
        nrm/=np.linalg.norm(nrm)+1e-7
        bulged.extend([v0,mid+bulge*nrm])
    return _chaikin(np.array(bulged,np.float32),1)

def _scale_from_reg(reg_pct:float,
                    r_min:float=0.20,r_max:float=0.40)->float:
    return r_min + (r_max-r_min)*(reg_pct/100)

def _rasterise(verts:np.ndarray,canvas:np.ndarray,pct_scale:float):
    h,w=canvas.shape; cx,cy=w//2,h//2; r=h*pct_scale
    pts=verts*r+np.array([cx,cy],np.float32)
    rr,cc=polygon_perimeter(pts[:,1],pts[:,0],shape=canvas.shape,clip=True)
    canvas[rr,cc]=255

def generate_batch(shape:str,reg_pct_array:np.ndarray)->np.ndarray:
    n_sides=cfg.SIDE_MAP[shape]
    B=len(reg_pct_array)
    canv=np.zeros((B,cfg.DRAW_RES,cfg.DRAW_RES),np.uint8)
    for i,rpct in enumerate(reg_pct_array):
        verts=_sample_vertices(n_sides,rpct)
        _rasterise(verts,canv[i],_scale_from_reg(rpct))
    imgs=sk_resize(canv,(B,cfg.OUT_RES,cfg.OUT_RES),
                   order=0,preserve_range=True).astype(np.uint8)
    return imgs

# ---- download & NP‑cache QuickDraw bitmaps ----------------------------
cfg.CACHE_DIR.mkdir(exist_ok=True)
_NPZ_CACHE:dict[str,np.lib.npyio.NpzFile]={}

def load_npz(path:Path):
    if path.as_posix() not in _NPZ_CACHE:
        _NPZ_CACHE[path.as_posix()]=np.load(path,mmap_mode="r")
    return _NPZ_CACHE[path.as_posix()]

@functools.lru_cache(maxsize=None)
def _qd_array(shape:str):
    p = cfg.CACHE_DIR/f"{shape}.npy"
    if not p.exists():
        url=f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{shape}.npy"
        with requests.get(url,stream=True) as r:
            r.raise_for_status()
            with open(p,"wb") as f:
                for chunk in r.iter_content(1<<16): f.write(chunk)
    return np.load(p,mmap_mode="r")


## 3. DATA  – datasets & dataloaders
def build_synthetic_dataset(force=False):
    if cfg.DATA_PATH.exists() and not force: return
    imgs,cls,reg=[],[],[]
    for cid,name in tqdm(enumerate(cfg.CLASSES), total=len(cfg.CLASSES)):
        for b in tqdm(range(cfg.N_BINS), total=cfg.N_BINS, leave=False):
            tgt=rng.uniform(b*10,(b+1)*10,
                            cfg.N_PER_SHAPE//cfg.N_BINS)
            imgs.append(generate_batch(name,tgt))
            cls .append(np.full_like(tgt,cid,np.int8))
            reg .append(tgt / 100.0)
    # ---- stack into uniform arrays -----------------------------------    
    imgs = np.concatenate(imgs, axis=0)           # (N, 256, 256)   uint8
    cls  = np.concatenate(cls,  dtype=np.int8)    # (N,)            int8
    reg  = np.concatenate(reg).astype(np.float32) # (N,)            float32

    np.savez_compressed(cfg.DATA_PATH, imgs=imgs, cls=cls, reg=reg)
    print(f"✅ Wrote {cfg.DATA_PATH}  ({imgs.shape[0]:,} samples)")

def build_qd_holdout(force=False):
    if cfg.HOLD_PATH.exists() and not force: return
    imgs,cls,reg=[],[],[]
    print("Building QuickDraw hold‑out …")
    for shape in tqdm(cfg.CLASSES, total=len(cfg.CLASSES)):
        src=_qd_array(shape)
        choice=rng.choice(len(src),cfg.HOLD_QD_PER_SHAPE,replace=False)
        for idx in choice:
            arr=src[idx].reshape(28,28)
            img=Image.fromarray(arr).resize((cfg.OUT_RES,cfg.OUT_RES),
                                            Image.NEAREST)
            imgs.append(np.asarray(img,np.uint8))
            cls.append(cfg.CLASSES.index(shape))
            reg.append(-1.0)
            
    imgs = np.stack(imgs, axis=0).astype(np.uint8)
    cls  = np.asarray(cls,  np.int8)
    reg  = np.asarray(reg,  np.float32)

    np.savez_compressed(cfg.HOLD_PATH, imgs=imgs, cls=cls, reg=reg)
    print(f"✅ Wrote {cfg.HOLD_PATH}  ({len(imgs):,} samples)")

class NPShapes(Dataset):
    def __init__(self,npz=cfg.DATA_PATH,indices=None,tfm=None):
        z=load_npz(npz)
        self.imgs,self.cls,self.reg=z["imgs"],z["cls"],z["reg"]
        self.idxs=np.arange(len(self.imgs)) if indices is None else indices
        self.tfm=tfm
    def __len__(self): return len(self.idxs)
    def __getitem__(self,i):
        j=self.idxs[i]
        img=Image.fromarray(self.imgs[j])
        if self.tfm: img=self.tfm(img)
        return img,\
               torch.tensor(int(self.cls[j])),\
               torch.tensor(float(self.reg[j]))

# -------- transforms ---------------------------------------------------
train_tfm=Tv2.Compose([
    Tv2.RandomAffine(degrees=15,translate=(0.15,0.15),
                     scale=(0.8,1.2),fill=0),
    Tv2.GaussianBlur(3,(0.2,1.2)),
    Tv2.RandomErasing(p=0.3,scale=(0.002,0.01),
                      ratio=(0.2,5.0),value=0),
    Tv2.ColorJitter(0.4,0.4),
    Tv2.ToImage(), Tv2.ToDtype(torch.float32,scale=True),
    Tv2.Lambda(lambda x: x + (torch.rand_like(x)<0.02).float()*
                         (2*torch.rand_like(x)-1)),
    Tv2.Normalize((0.5,),(0.5,))
])
val_tfm=Tv2.Compose([
    Tv2.ToImage(), Tv2.ToDtype(torch.float32,scale=True),
    Tv2.Normalize((0.5,),(0.5,))
])

def make_loaders():
    all_npz=load_npz(cfg.DATA_PATH)
    N=len(all_npz["cls"])
    perm=rng.permutation(N)
    split=int(N*cfg.TRAIN_FRAC)
    tr_idx,va_idx=perm[:split],perm[split:]
    ds_tr=NPShapes(indices=tr_idx,tfm=train_tfm)
    ds_va=NPShapes(indices=va_idx,tfm=val_tfm)
    ds_ho=NPShapes(npz=cfg.HOLD_PATH,tfm=val_tfm)
    loader=lambda ds,shuf: DataLoader(
        ds,batch_size=cfg.BATCH_SIZE,shuffle=shuf,
        num_workers=8,pin_memory=True,persistent_workers=True)
    return loader(ds_tr,True), loader(ds_va,False), loader(ds_ho,False)



## 4. MODEL
class ShapeNet(nn.Module):
    def __init__(self,n_cls=len(cfg.CLASSES),dropout_p=0.4):
        super().__init__()
        base=resnet18(weights=None)
        base.conv1=nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.encoder=nn.Sequential(*list(base.children())[:-2])
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.drop=nn.Dropout(dropout_p)
        dim=base.fc.in_features
        self.cls_head=nn.Linear(dim,n_cls)
        self.reg_head=nn.Sequential(nn.Linear(dim,128),nn.ReLU(inplace=True),
                                    nn.Linear(128,1)),nn.Sigmoid()
    def forward(self,x):
        f=self.drop(self.gap(self.encoder(x)).flatten(1))
        return self.cls_head(f), self.reg_head(f).squeeze(1)

class ShapeLit(L.LightningModule):
    def __init__(self,model:nn.Module):
        super().__init__()
        self.model=model
        self.loss_cls=nn.CrossEntropyLoss()
        self.loss_reg=nn.MSELoss()
        self.train_acc=torchmetrics.Accuracy(task="multiclass",
                                             num_classes=len(cfg.CLASSES))
        self.val_acc=torchmetrics.Accuracy(task="multiclass",
                                           num_classes=len(cfg.CLASSES))
    def forward(self,x): return self.model(x)
    def _step(self,batch,stage):
        x,yc,yr=batch
        logits,reg=self(x)
        cls_loss=self.loss_cls(logits,yc)
        mask=yr>=0
        reg_loss=self.loss_reg(reg[mask],yr[mask]) if mask.any() else 0.0
        loss=cls_loss+reg_loss
        acc=self.train_acc if stage=="train" else self.val_acc
        acc.update(logits.softmax(-1),yc)
        self.log_dict({f"{stage}_{k}":v for k,v in
                      dict(loss=loss,cls_loss=cls_loss,
                           reg_loss=reg_loss,acc=acc).items()},
                      prog_bar=True,sync_dist=True)
        return loss
    training_step=lambda self,b,_: self._step(b,"train")
    validation_step=lambda self,b,_: self._step(b,"val")
    def configure_optimizers(self):
        opt=torch.optim.AdamW(self.parameters(),lr=cfg.LR,weight_decay=cfg.WD)
        sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,"min",patience=2,
                                                       factor=0.5)
        return {"optimizer":opt,"lr_scheduler":{"scheduler":sch,
                                                "monitor":"val_loss"}}



    ## 5. TRAIN
    def train():
    build_synthetic_dataset(); build_qd_holdout()
    dl_tr, dl_va, _ = make_loaders()

    model = ShapeNet()
    lit   = ShapeLit(model)

    early = EarlyStopping(monitor="val_loss", patience=3, mode="min")
    ckpt  = ModelCheckpoint(dirpath=cfg.CKPT_DIR,
                            save_top_k=1, monitor="val_loss", mode="min")

    trainer = L.Trainer(
        max_epochs   = cfg.MAX_EPOCHS,
        accelerator  = "gpu" if torch.cuda.is_available() else "cpu",
        precision    = cfg.PRECISION,
        callbacks    = [early, ckpt],
        log_every_n_steps = 10,
    )
    trainer.fit(lit, dl_tr, dl_va)

train()



## 6. INFER  (run after training – optional)
def interactive_demo(ckpt_path:Path):
    import cv2, matplotlib.pyplot as plt
    device="cuda" if torch.cuda.is_available() else "cpu"
    lit=ShapeLit.load_from_checkpoint(ckpt_path,map_location=device,
                                      model=ShapeNet())
    lit.eval().to(device)
    val_tfm=T.Compose([T.ToTensor(),T.Normalize([0.5],[0.5])])

    CANVAS,TH=cfg.OUT_RES,cfg.STROKE_PX
    canvas=np.zeros((CANVAS,CANVAS),np.uint8); drawing=False
    def _mouse(e,x,y,f,p):
        nonlocal drawing
        if e==cv2.EVENT_LBUTTONDOWN: drawing=True
        elif e==cv2.EVENT_MOUSEMOVE and drawing:
            cv2.line(canvas,p[:2],(x,y),255,TH); p[:2]=x,y
        elif e==cv2.EVENT_LBUTTONUP: drawing=False
    win="Draw (s=pred r=reset q=quit)"
    cv2.namedWindow(win); pt=[-1,-1]; cv2.setMouseCallback(win,_mouse,pt)
    while True:
        cv2.imshow(win,canvas); k=cv2.waitKey(1)&255
        if k==ord("s") and canvas.max():
            small=cv2.resize(canvas,(cfg.DRAW_RES,cfg.DRAW_RES),
                             cv2.INTER_AREA)
            inp=cv2.resize(small,(CANVAS,CANVAS),cv2.INTER_NEAREST)
            x=val_tfm(Image.fromarray(inp)).unsqueeze(0).to(device)
            with torch.no_grad(),torch.cuda.amp.autocast():
                logits,reg=lit(x)
            probs=logits.softmax(-1).cpu().numpy()[0]
            cls=probs.argmax(); name=cfg.CLASSES[cls]
            plt.figure(); plt.imshow(inp,cmap="gray"); plt.axis("off")
            plt.text(2,16,f"{name} {probs[cls]*100:.1f}%",
                     color="white",weight="bold",
                     bbox=dict(fc="black",alpha=.6,pad=3))
            plt.show()
        elif k==ord("r"): canvas[:]=0
        elif k==ord("q"): break
    cv2.destroyAllWindows()