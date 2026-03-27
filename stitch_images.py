import cv2
import numpy as np
import os

def stitch(name):
    path1 = f"/Users/jclaw/.openclaw/workspace/APS_Project/test_images/{name}.jpg"
    path2 = f"/Users/jclaw/.openclaw/workspace/APS_Project/output/{name}_retouched.jpg"
    out_path = f"/Users/jclaw/.openclaw/workspace/APS_Project/compare_{name}.jpg"
    
    if not os.path.exists(path1) or not os.path.exists(path2):
        print(f"Skipping {name}, files not found.")
        return
        
    i1 = cv2.imread(path1)
    i2 = cv2.imread(path2)
    
    h = 900
    w1 = int(i1.shape[1] * (h / i1.shape[0]))
    w2 = int(i2.shape[1] * (h / i2.shape[0]))
    
    i1 = cv2.resize(i1, (w1, h))
    i2 = cv2.resize(i2, (w2, h))
    
    # Add Text Background Box
    cv2.rectangle(i1, (0, 0), (450, 80), (0, 0, 0), -1)
    cv2.rectangle(i2, (0, 0), (450, 80), (0, 0, 0), -1)
    
    cv2.putText(i1, "ORIGINAL (RAW)", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.putText(i2, "APS RETOUCHED", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    
    comp = np.hstack((i1, i2))
    cv2.imwrite(out_path, comp)
    print(f"Saved {out_path}")

stitch("test1")
stitch("test3")
