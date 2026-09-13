"""Optical flow on lower box markers, with an explicit rotation error budget.

These are visible material reference sites, not claimed physical contact
corners. A conservative lever-arm term bounds marker/corner discrepancies.
No changing segmentation-derived bottom is used as a vertical baseline.
"""

import cv2
import numpy as np


def locate_markers(hsv, side):
    blue=cv2.inRange(hsv,(92,75,25),(125,255,255))
    blue[:140]=0
    if side==0: blue[:,256:]=0
    else: blue[:,:384]=0
    n,labels,stats,centers=cv2.connectedComponentsWithStats(blue)
    candidates=[]
    for k in range(1,n):
        x,y,w,h,area=map(int,stats[k])
        if area>=35 and 12<=h<=140 and w>=3 and h>=.5*w:
            candidates.append((area,k,h))
    if not candidates: return None
    candidates.sort(reverse=True)
    _,leader,leader_h=candidates[0]
    chosen=[leader]
    for _,k,h in candidates[1:]:
        if h>=.65*leader_h and abs(centers[k][0]-centers[leader][0])<=110:
            chosen.append(k)
            break
    mask=np.isin(labels,chosen).astype(np.uint8)*255
    yy,xx=np.nonzero(mask)
    points=[]
    for select in (xx<=np.quantile(xx,.4),xx>=np.quantile(xx,.6)):
        x,y=xx[select],yy[select]
        low=y>=np.quantile(y,.97)
        points.append([float(np.median(x[low])),float(np.median(y[low]))])
    return {'mask':mask,'sites':np.asarray(points,np.float32),
            'center':np.asarray([xx.mean(),yy.mean()],np.float32),
            'components':len(chosen)}


class MarkerReference:
    def __init__(self,gray,hsv,side,slope,config):
        self.gray=gray
        self.side=side
        self.slope=slope
        self.config=config
        self.reference=locate_markers(hsv,side)
        self.points=None
        self.initial_offset=None
        if self.reference is None: return
        mask=cv2.dilate(self.reference['mask'],np.ones((3,3),np.uint8))
        green=cv2.inRange(hsv,(35,60,25),(90,255,255))
        mask[cv2.dilate(green,np.ones((3,3),np.uint8))>0]=0
        self.points=cv2.goodFeaturesToTrack(gray,maxCorners=90,qualityLevel=.005,
                                           minDistance=3,blockSize=3,mask=mask)

    def update(self,gray,hsv):
        failed={'flow_valid':False,'side':'left' if self.side==0 else 'right'}
        if self.points is None or len(self.points)<6:
            return {**failed,'reason':'insufficient_reference_marker_features'}
        current=locate_markers(hsv,self.side)
        guess=self.points.copy()
        if current is not None:
            guess+=current['center']-self.reference['center']
        lk=dict(winSize=(25,25),maxLevel=4,
                criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,40,.01),
                flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        forward,ok,_=cv2.calcOpticalFlowPyrLK(self.gray,gray,self.points,guess,**lk)
        if forward is None: return {**failed,'reason':'forward_flow_failed'}
        backward,back_ok,_=cv2.calcOpticalFlowPyrLK(gray,self.gray,forward,self.points.copy(),**lk)
        if backward is None: return {**failed,'reason':'backward_flow_failed'}
        fb=np.linalg.norm(backward[:,0]-self.points[:,0],axis=1)
        valid=(ok[:,0]>0)&(back_ok[:,0]>0)&np.isfinite(forward[:,0]).all(axis=1)&(fb<=1.5)
        pixels=np.rint(np.nan_to_num(forward[:,0],nan=-1000,posinf=-1000,neginf=-1000)).astype(int)
        inside=(pixels[:,0]>=0)&(pixels[:,0]<640)&(pixels[:,1]>=0)&(pixels[:,1]<480)
        valid&=inside
        allowed=np.ones(len(valid),bool)
        green=cv2.inRange(hsv,(35,60,25),(90,255,255))
        allowed[inside]=green[pixels[inside,1],pixels[inside,0]]==0
        if current is not None:
            mask=cv2.dilate(current['mask'],np.ones((9,9),np.uint8))
            allowed[inside]&=mask[pixels[inside,1],pixels[inside,0]]>0
        valid&=allowed
        if int(valid.sum())<6:
            return {**failed,'reason':'insufficient_consistent_marker_features','valid_features':int(valid.sum())}
        source=self.points[:,0][valid]
        target=forward[:,0][valid]
        matrix,inliers=cv2.estimateAffinePartial2D(source,target,method=cv2.RANSAC,
                    ransacReprojThreshold=1.5,maxIters=2000,confidence=.99)
        if matrix is None or inliers is None or int(inliers.sum())<6:
            return {**failed,'reason':'rigid_fit_failed'}
        keep=inliers[:,0]>0
        source_good=source[keep]
        predicted=source_good@matrix[:,:2].T+matrix[:,2]
        residual=np.linalg.norm(predicted-target[keep],axis=1)
        error=float(np.quantile(residual,.9))
        fb_error=float(np.quantile(fb[valid][keep],.9))
        scale=float(np.linalg.norm(matrix[0,:2]))
        rotation=float(np.degrees(np.arctan2(matrix[1,0],matrix[0,0])))
        span=np.ptp(source_good,axis=0)
        ref_span=np.ptp(self.points[:,0],axis=0)
        if not .75<=scale<=1.30 or error>2.5 or abs(rotation)>35:
            return {**failed,'reason':'unstable_marker_projection'}
        if span[0]<max(4,.20*ref_span[0]) or span[1]<6:
            return {**failed,'reason':'insufficient_spatial_coverage'}
        original=self.reference['sites']
        transformed=original@matrix[:,:2].T+matrix[:,2]
        delta=transformed-original
        raw=self.slope*delta[:,0]-delta[:,1]
        if self.initial_offset is None: self.initial_offset=raw.copy()
        movement=np.minimum(raw,raw-self.initial_offset)
        lever=float(self.config['contact_lever_arm_px'])
        if self.reference['components']==1: lever*=1.5
        rotation_error=lever*abs(np.sin(np.radians(rotation)))+5*abs(1-scale)
        band=float(self.config['flow_error_floor_px'])+error+fb_error+rotation_error
        height,width=gray.shape
        visible=(transformed[:,0]>=2)&(transformed[:,0]<width-2)&(transformed[:,1]>=2)&(transformed[:,1]<height-2)
        center=self.reference['center']@matrix[:,:2].T+matrix[:,2]
        return {'flow_valid':True,'side':failed['side'],'corner_displacement_px':movement.tolist(),
                'corner_visible':visible.tolist(),'uncertainty_margin_px':band,
                'lower_marker_sites':transformed.tolist(),'reference_marker_sites':original.tolist(),
                'center':center.tolist(),'fit_error_p90_px':error,'fb_error_p90_px':fb_error,
                'rotation_error_budget_px':rotation_error,'box_rotation_deg':rotation,'scale':scale,
                'feature_inliers':int(keep.sum()),'initial_appearance_offset_px':self.initial_offset.tolist(),
                'measurement_kind':'lower_marker_motion_with_rotation_budget_not_exact_contact_corners'}


class MarkerFlowPair:
    def __init__(self,reference_frame,config):
        gray=cv2.cvtColor(reference_frame,cv2.COLOR_BGR2GRAY)
        hsv=cv2.cvtColor(reference_frame,cv2.COLOR_BGR2HSV)
        detected=[locate_markers(hsv,k) for k in range(2)]
        self.angle0=None
        slope=0.0
        if all(item is not None for item in detected):
            a,b=[item['center'] for item in detected]
            slope=float((b[1]-a[1])/max(1,b[0]-a[0]))
            self.angle0=float(np.degrees(np.arctan2(b[1]-a[1],b[0]-a[0])))
        self.initial_angle_offset=None
        self.references=[MarkerReference(gray,hsv,k,slope,config) for k in range(2)]

    def update(self,frame):
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        sides=[reference.update(gray,hsv) for reference in self.references]
        angle=None
        if self.angle0 is not None and all(s.get('flow_valid') for s in sides):
            a,b=[s['center'] for s in sides]
            raw=float(np.degrees(np.arctan2(b[1]-a[1],b[0]-a[0])))-self.angle0
            if self.initial_angle_offset is None: self.initial_angle_offset=raw
            angle=raw-self.initial_angle_offset
        return {'sides':sides,'angle_change_deg':angle}
