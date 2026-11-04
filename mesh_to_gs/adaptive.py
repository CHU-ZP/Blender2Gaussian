"""World-space material and curvature importance for surface initialization."""
from collections import OrderedDict
import numpy as np

from .surface import linear_texture


def bilinear(image,uv):
    h,w=image.shape[:2]
    xy=np.column_stack((uv[:,0]*w-.5,(1-uv[:,1])*h-.5))
    lo=np.floor(xy).astype(np.int64);f=xy-lo;x,y=lo.T
    return ((1-f[:,0,None])*(1-f[:,1,None])*image[y%h,x%w]
            +f[:,0,None]*(1-f[:,1,None])*image[y%h,(x+1)%w]
            +(1-f[:,0,None])*f[:,1,None]*image[(y+1)%h,x%w]
            +f[:,0,None]*f[:,1,None]*image[(y+1)%h,(x+1)%w])


class TextureFeatures:
    def __init__(self,image):
        self.image=linear_texture(image)
        self.levels=[]
        level=self.image
        while True:
            h,w=level.shape[:2]
            gu=(np.roll(level,-1,axis=1)-np.roll(level,1,axis=1))*(w/2)
            gv=(np.roll(level,1,axis=0)-np.roll(level,-1,axis=0))*(h/2)
            self.levels.append((gu,gv))
            if max(h,w)<=2:break
            # Area mean of a periodic 2x2 footprint, also for odd dimensions.
            yy=np.arange(0,h,2);xx=np.arange(0,w,2)
            level=(level[yy[:,None],xx]+level[(yy[:,None]+1)%h,xx]
                   +level[yy[:,None],(xx+1)%w]+level[(yy[:,None]+1)%h,(xx+1)%w])*.25

    def gradients(self,uv,jacobian,spacing):
        # jacobian maps world displacements to (u,v). The mip level follows
        # world footprint/UV stretch rather than raw texture pixel gradients.
        jacobian=np.broadcast_to(jacobian,(len(uv),2,3))
        h,w=self.image.shape[:2]
        footprint=spacing*.35*np.sqrt(np.sum((jacobian*np.array([w,h])[None,:,None])**2,axis=(1,2))/2)
        lod=np.floor(np.log2(np.maximum(footprint,1))).astype(int).clip(0,len(self.levels)-1)
        result=np.empty((len(uv),3,3),np.float32)
        for level in np.unique(lod):
            take=lod==level;gu,gv=self.levels[level]
            a,b=bilinear(gu,uv[take]),bilinear(gv,uv[take])
            result[take]=a[:,:,None]*jacobian[take,0,None,:]+b[:,:,None]*jacobian[take,1,None,:]
        return result

    def score(self,uv,jacobian,spacing):
        g=self.gradients(uv,jacobian,spacing)
        contrast=spacing*np.sqrt(np.mean(np.sum(g*g,axis=2),axis=1))
        return np.clip((contrast-.04)/.20,0,1)


class TextureCache:
    def __init__(self,loader,limit=3):
        self.loader=loader;self.limit=limit;self.items=OrderedDict()
    def get(self,image):
        if image is None:return None
        key=image.as_pointer() if hasattr(image,'as_pointer') else id(image)
        if key in self.items:
            value=self.items.pop(key)
        else:value=TextureFeatures(self.loader(image))
        self.items[key]=value
        while len(self.items)>self.limit:self.items.popitem(last=False)
        return value


def uv_jacobians(surface, *, strict=False):
    e1=surface.triangles[:,1]-surface.triangles[:,0]
    e2=surface.triangles[:,2]-surface.triangles[:,0]
    inv_cross=1/(2*surface.areas if strict else np.maximum(2*surface.areas,1e-30))
    b1=np.cross(e2,surface.normals)*inv_cross[:,None]
    b2=np.cross(surface.normals,e1)*inv_cross[:,None]
    d1=surface.uv[:,1]-surface.uv[:,0];d2=surface.uv[:,2]-surface.uv[:,0]
    result=d1[:,:,None]*b1[:,None,:]+d2[:,:,None]*b2[:,None,:]
    if strict and not np.isfinite(result).all():
        raise FloatingPointError('UV Jacobian cannot be represented at source-face precision')
    return result


def prepare_importance(surface,texture,spacing,texture_weight=2.,curvature_weight=1.,detail_scale=1.):
    """Return face importance; exact same field is used within large faces."""
    n=len(surface.areas)
    score=np.zeros(n,np.float32)
    jac=None
    if texture is not None and n:
        jac=uv_jacobians(surface)
        bary=np.array([[1/3,1/3,1/3],[.8,.1,.1],[.1,.8,.1],[.1,.1,.8],
                       [.45,.45,.1],[.45,.1,.45],[.1,.45,.45]])
        for start in range(0,n,32768):
            stop=min(start+32768,n)
            uv=np.einsum('ki,nij->nkj',bary,surface.uv[start:stop]).reshape(-1,2)
            j=np.repeat(jac[start:stop],len(bary),axis=0)
            values=texture.score(uv,j,spacing).reshape(stop-start,-1)
            score[start:stop]=values.mean(1)
    curvature=np.zeros(n) if surface.curvature is None else np.clip(surface.curvature*spacing/.2,0,1)
    importance=1+detail_scale*(texture_weight*score+curvature_weight*curvature)
    surface.importance=importance
    surface.texture_scores=score
    surface.curvature_scores=curvature
    surface.uv_jacobian=jac
    surface.adaptive_config=(spacing,texture_weight,curvature_weight,detail_scale)
    return importance


def face_sampler(surface,face,texture):
    if texture is None:return None
    spacing,tw,cw,ds=surface.adaptive_config
    uv=surface.uv[face];jac=surface.uv_jacobian[face]
    base=1+ds*cw*surface.curvature_scores[face]
    def evaluate(bary):
        return base+ds*tw*texture.score(bary@uv,jac,spacing)
    return evaluate
