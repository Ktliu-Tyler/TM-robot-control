import argparse
import cv2
import numpy as np

# Camera axes: right, down, forward. Distances in mm.
def estimate(rgb, depth):
    if depth.dtype != np.uint16 or depth.shape != rgb.shape[:2]:
        raise ValueError("Expected aligned RGB and uint16 depth (mm)")
    v, u = np.indices(depth.shape)
    xyz = np.dstack(((u-286.54)*depth/613.57, (v-151.36)*depth/613.54, depth))
    valid = depth > 0
    sample = xyz[valid][::12]
    rng = np.random.default_rng(7)
    best = 0
    for _ in range(160):
        a,b,c = sample[rng.choice(len(sample), 3, replace=False)]
        n = np.cross(b-a,c-a)
        n /= max(np.linalg.norm(n), 1e-12)
        inlier = abs((sample-a) @ n) < 3
        if inlier.sum() > best:
            best = inlier.sum()
            table = sample[inlier]
    center = table.mean(axis=0)
    normal = np.linalg.svd(table-center, full_matrices=False)[2][-1]
    if normal[2] > 0:
        normal *= -1
    height = (xyz-center) @ normal
    mask = cv2.morphologyEx((valid & (height > 8)).astype('uint8'), cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    poses = []
    for i in range(1,count):
        if stats[i,cv2.CC_STAT_AREA] < 90:
            continue
        region = labels == i
        h = np.percentile(height[region],80)
        top = xyz[region & (abs(height-h)<3)]
        if len(top)<20:
            continue
        # Blocks rest flat. Fit a rectangle in the tabletop plane.
        ex = np.array([1.,0.,0.]) - normal*normal[0]
        ex /= np.linalg.norm(ex)
        ey = np.cross(normal,ex)
        rect = cv2.minAreaRect(np.column_stack((top@ex,top@ey)).astype('float32'))
        box = cv2.boxPoints(rect)
        edges = np.roll(box,-1,axis=0)-box
        edge = edges[np.argmax(np.linalg.norm(edges,axis=1))]
        x = edge[0]*ex + edge[1]*ey
        x /= np.linalg.norm(x)
        if x[0]<0:
            x *= -1
        y = np.cross(normal,x)
        pos = rect[0][0]*ex + rect[0][1]*ey + (center@normal+h/2)*normal
        angles = np.degrees([np.arctan2(y[2],normal[2]),np.arcsin(-x[2]),np.arctan2(x[1],x[0])])
        poses.append((pos,angles))
    return poses

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rgb")
    parser.add_argument("depth")
    parser.add_argument("--output", default="part_d_result.png")
    args = parser.parse_args()
    rgb = cv2.imread(args.rgb)
    depth = cv2.imread(args.depth,cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None:
        parser.error("Cannot read image pair")
    poses = estimate(rgb,depth)
    panel = np.zeros((max(rgb.shape[0],24*(len(poses)+1)),rgb.shape[1]+510,3), np.uint8)
    panel[:rgb.shape[0],:rgb.shape[1]] = rgb
    for i,(p,r) in enumerate(poses,1):
        at = tuple(np.rint([613.57*p[0]/p[2]+286.54,613.54*p[1]/p[2]+151.36]).astype(int))
        line = f"{i}  xyz(mm) "+' '.join(f"{n:.1f}" for n in p)+"  rpy(deg) "+' '.join(f"{n:.1f}" for n in r)
        print(line)
        cv2.circle(panel,at,3,(0,0,255),-1)
        cv2.putText(panel,str(i),at,0,.55,(255,255,255),2)
        cv2.putText(panel,line,(rgb.shape[1]+6,24*i),0,.43,(255,255,255),1,cv2.LINE_AA)
    if not cv2.imwrite(args.output,panel):
        raise IOError(args.output)

if __name__ == "__main__":
    main()
