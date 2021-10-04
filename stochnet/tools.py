import torch, os, dill

from math import sqrt
from scipy.special import xlogy
from torch.nn.parameter import Parameter

__SN_comp_version__ = [3.1, 'GH_1.0']
__SN_version__ = 'GH_1.0'
__EPS__ = torch.finfo(torch.float32).eps

__out_file__ = None
__term__ = True
__out_dir__ = './'

__OUT_DIR__ = lambda: __out_dir__
__OUT_FILE__ = lambda: __out_file__
__TERM__ = lambda: __term__

def Print(*args, **kwargs):
  term = __TERM__()
  out_file = __OUT_FILE__()
  if term: print(*args, **kwargs)
  if out_file is not None:
    out_file = os.path.join(__OUT_DIR__(), __OUT_FILE__())
    with open(out_file, 'a') as f:
      for arg in args:
        f.write(f'{arg}\n')
      f.flush()
      f.close()

def KL_bin(q, p):
    return xlogy(q, q/p) + xlogy(1-q, (1-q)/(1-p))

def h(q, c, p):
    return KL_bin(q, p) - c

def hp(q, c, p):
    return (1-q)/(1-p) - q/p
    
def Newton_KL(q, c, p0, iter):
    p = p0
    for i in range(iter):
        p -= h(q, c, p) / hp(q, c, p)
        if p>=1:
          p = 1-__EPS__
    return p

def inv_KL(q, c, iter=5):
    b = q + sqrt(c/2)
    if b >= 1:
      b = 1-__EPS__
      iter=20
    return Newton_KL(q, c, b, iter)

def KL_bin_torch(q, p):
    return torch.xlogy(q, q/p) + torch.xlogy(1-q, (1-q)/(1-p))

def h_torch(q, c, p):
    return KL_bin_torch(q, p) - c

def hp_torch(q, c, p):
    return (1-q)/(1-p) - q/p
    
def Newton_KL_torch(q, c, p0, iter):
    p = p0
    for i in range(iter):
        p = torch.clamp(p - h_torch(q, c, p) / (torch.clamp(hp_torch(q, c, p), min=__EPS__)), max=1-__EPS__)
    return p

def inv_KL_torch(q, c, iter=10, **kwargs):
    b = torch.clamp(q + torch.sqrt(c/2), max=1-__EPS__)
    return Newton_KL_torch(q, c, b, iter)

class InvKL(torch.autograd.Function):
  
  @staticmethod
  def forward(ctx, q, c, iter=10):
    out = inv_KL_torch(q, c, iter=iter)
    ctx.save_for_backward(q, out)
    return out
  
  @staticmethod
  def backward(ctx, grad_output):
    q, out = ctx.saved_tensors
    grad_q = grad_c = None
    den = (1-q)/(1-out) - q/out
    den[den==0] = __EPS__
    sign = den.sign()
    den = den.abs_().clamp_(min=__EPS__)
    den *= sign
    grad_c = grad_output / den
    grad_q = (torch.log(torch.clamp((1-q)/(1-out), min=__EPS__)) - torch.log(torch.clamp(q/out, min=__EPS__))) * grad_c * grad_output
    return grad_q, grad_c, None #last None is for iter...

invkl = InvKL.apply

def _mk_perm(dim, device):
  OUT = torch.eye(dim, device=device, requires_grad=False).repeat(dim, 1, 1)
  for k, i in enumerate(OUT):
    i[k,k] = 0.
    i[-1,-1] = 0.
    i[k, -1] = 1.
    i[-1, k] = 1.
  return OUT

def gauss_ccdf(x):
  return (1-torch.erf(x/2**.5))/2

def par_to_buf(module, attr):
  if attr not in module._buffers:
    assert attr in module._parameters
    _attr = module._parameters[attr]
    if _attr is not None:
      temp = _attr.clone().detach()
    else:
      temp = None
    del module._parameters[attr]
    deleted = False
    try:
      delattr(module, attr)
      deleted = True
    except: pass
    module.register_buffer(attr, temp)
    if deleted: setattr(module, attr, temp)

def buf_to_par(module, attr):
  if attr not in module._parameters:
    assert attr in module._buffers
    _attr = module._buffers[attr]
    if _attr is not None:
      temp = Parameter(_attr.clone())
      temp.requires_grad = True
    else:
      temp = None
    del module._buffers[attr]
    deleted = False
    try:
      delattr(module, attr)
      deleted = True
    except: pass
    module.register_parameter(attr, temp)
    if deleted: setattr(module, attr, temp)

def save_lists(*args, name=None, path=None):
  assert name is not None
  if path is None: path = __OUT_DIR__()
  path = os.path.join(path, name)
  with open(path, 'wb') as fp:
    dill.dump(args, fp, protocol=dill.HIGHEST_PROTOCOL)
