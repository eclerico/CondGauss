import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import dill, math, os
from time import strftime

from stochnet.tools import Print, inv_KL, invkl, gauss_ccdf, _mk_perm, save_lists, buf_to_par, par_to_buf, __OUT_DIR__, __EPS__, __SN_version__, __SN_comp_version__

#Available training methods
__methods__ = ['McAll', 'invKL', 'quad', 'lbd', 'ERM']

#Listed attribute for layers
def __attrs__(l_type):
  if l_type == nn.Conv2d:
    return ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups', 'padding_mode']
  elif l_type == nn.Linear:
    return ['in_features', 'out_features']
  else: assert False

#Function to make a layer stochastic
def _make_stoch(layer, weight_v=None, bias_v=None, train_v=False):
  if isinstance(layer, nn.Linear):
    return StochLinear(layer, weight_v, bias_v, train_v)
  if isinstance(layer, nn.Conv2d):
    return StochConv2d(layer, weight_v, bias_v, train_v)
  else: assert False

#Convert prior variances to rho
def _var_to_rho(var):
  return var**(1/3)

#Convert rho to prior standard deviation
def _rho_to_std(rho):
  return torch.abs(rho)**1.5

__CLAMP_RHO__ = False #If True, clamp rho from below to __EPS__


"""A Block is the container for a layer. It contains a function to be applied before the layer (activation), and a function to be applied after the layer (usually pooling of flattening). It can set the dropout as well."""

class Block(nn.Module):

  def __init__(self, layer, b_name=None, name=None, act=None, post=None, nb=None, act_name=None, post_name=None, dropout=0):
    super(Block, self).__init__()
    self.layer = layer
    self.b_name = b_name if b_name is not None else 'B'
    self.name = name if name is not None else ''
    self.nb = nb
    self.act = act if act is not None else lambda x: x
    self.post = post if post is not None else lambda x: x
    self.set_dropout(dropout)
    if act_name is not None:
      self._act_name = act_name
    elif act is None:
      self._act_name = 'None'
    else:
      self._act_name = act.__name__
    if post_name is not None:
      self._post_name = post_name
    elif post is None:
      self._post_name = 'None'
    else:
      self._post_name = post.__name__

  def forward(self, x):
    return self.post(self.dropout(self.layer(self.act(x))))
  
  #used to save and load models
  def selfie(self):
    layer = type(self.layer)
    if layer == StochLinear: layer=nn.Linear
    if layer == StochConv2d: layer=nn.Conv2d
    assert layer in [nn.Linear, nn.Conv2d]
    attrs = dict()
    for attr in __attrs__(layer):
      attrs.update({attr : getattr(self.layer, attr)})
    bias = self.layer.bias is not None
    attrs.update(bias=bias)
    misc = dict(b_name=self.b_name, name=self.name, act=self.act, post=self.post, act_name=self._act_name, post_name=self._post_name, dropout=self._do)
    return dict(layer=layer, attrs=attrs, misc=misc)
  
  def extra_repr(self):
    s = ''
    if self._act_name != 'None':
      s += f'(pre-activation): {self._act_name}'
      if self._post_name != 'None':
        s += f'\n(post-processing): {self._post_name}'
    elif self._post_name != 'None':
      s += f'(post-processing): {self._post_name}'
    return s
  
  def set_dropout(self, dropout):
    if dropout != 0:
      if type(self.layer) in [nn.Linear, StochLinear]:
        self.dropout = nn.Dropout(dropout)
      elif type(self.layer) in [nn.Conv2d, StochConv2d]:
        self.dropout = nn.Dropout2d(dropout)
      else: assert False, f"Cannot automatically implement dropout for layer of type {type(self.layer)}"
    else:
      try: del self.dropout
      except AttributeError: pass
      self.dropout = lambda x: x
    self._do = dropout


"""StochLayer is the mother class for stochastic layers"""

class StochLayer(nn.Module):
  
  def __init__(self, mother, weight_v, bias_v, train_v):
    super(StochLayer, self).__init__()
    for attr in __attrs__(type(mother)):
      setattr(self, attr, getattr(mother, attr))
    self.weight = Parameter(mother.weight.detach().clone(), requires_grad=True)
    if mother.bias is not None:
      self.bias = Parameter(mother.bias.detach().clone(), requires_grad=True)
    else:
      self.register_parameter('bias', None)
    self.train_v = train_v
    self.set_vars(weight_v, bias_v)
    self.freeze_in()

  def forward(self, x):
    assert False, "Not implemented"

  def weight_std(self):
    return _rho_to_std(self.weight_rho)
  
  def bias_std(self):
    if self.bias_rho is not None:
      return _rho_to_std(self.bias_rho)
    else: return None

  #Evaluate the KL between prior and posterior for the layer
  def Penalty(self):
    out = .5 * torch.sum(((self.weight-self.weight_in)**2 + (torch.square(self.weight_std())-torch.square(self.weight_std_in)))/(torch.square(self.weight_std_in)) + torch.log(torch.square(self.weight_std_in/self.weight_std())))
    if self.bias is not None:
      out += .5 * torch.sum(((self.bias-self.bias_in)**2 + (torch.square(self.bias_std())-torch.square(self.bias_std_in)))/(torch.square(self.bias_std_in)) + torch.log(torch.square(self.bias_std_in/self.bias_std())))
    return out

  #Makes the variances trainable
  def activate_vars(self):
    self.train_v = True
    buf_to_par(self, 'weight_rho')
    buf_to_par(self, 'bias_rho')

  #Only the means are trainable
  def deactivate_vars(self):
    self.train_v = False
    par_to_buf(self, 'weight_rho')
    par_to_buf(self, 'bias_rho')

  #Fix the values of the variances
  def set_vars(self, weight_v, bias_v):
    self.weight_rho = Parameter(torch.full(self.weight.shape, _var_to_rho(weight_v), requires_grad=True, device=self.weight.device))
    if self.bias is not None:
      self.bias_rho = Parameter(torch.full(self.bias.shape, _var_to_rho(bias_v), requires_grad=True, device=self.weight.device))
    else:
      self.register_parameter('bias_rho', None)
    if not self.train_v: self.deactivate_vars()

  #Change the prior to the current posterior configuration (to be used when training first the prior and then the posterior).
  def freeze_in(self):
    self.register_buffer('weight_in', self.weight.clone().detach())
    self.register_buffer('weight_std_in', self.weight_std().clone().detach())
    if self.bias is not None:
      self.register_buffer('bias_in', self.bias.clone().detach())
      self.register_buffer('bias_std_in', self.bias_std().clone().detach())
    else:
      self.register_buffer('bias_in', None)
      self.register_buffer('bias_std_in', None)

  def clamp_rho(self):
    with torch.no_grad():
      self.weight_rho.clamp_(min=__EPS__)
      if self.bias_rho is not None: self.bias_rho.clamp(min=__EPS__)
  
  def clamp_rho_abs(self, min):
    with torch.no_grad():
      rho = self.weight_rho
      sign = rho.sign()
      rho = rho.abs_().clamp_(min=min)
      rho *= sign
      if self.bias_rho is not None:
        rho = self.bias_rho
        sign = rho.sign()
        rho = rho.abs_().clamp_(min=min)
        rho *= sign


"""StochLinear is the child class of StochLayer dealing with a stochastic linear layer"""

class StochLinear(StochLayer):
  
  def __init__(self, mother, weight_v=.01, bias_v=.01, train_v=True):
    assert isinstance(mother, nn.Linear)
    super(StochLinear, self).__init__(mother, weight_v, bias_v, train_v)

  def forward(self, x):
    bias_v = torch.square(self.bias_std()) if self.bias_rho is not None else None
    A = torch.sqrt(F.linear(x**2, torch.square(self.weight_std()), bias_v))
    M = F.linear(x, self.weight, self.bias)
    N = torch.randn(self.out_features, device=self.weight.device, requires_grad=False) #same ramdomness through same batch
    out = A * N + M
    return out

  def extra_repr(self) -> str:
    return 'in_features={}, out_features={}, bias={}'.format(
        self.in_features, self.out_features, self.bias is not None
    )


"""StochConv2d is a child class of StochLayer, dealing with a stochastic convolutional layer"""

class StochConv2d(StochLayer):
  
  def __init__(self, mother, weight_v=.01, bias_v=.01, train_v=True):
    assert isinstance(mother, nn.Conv2d)
    super(StochConv2d, self).__init__(mother, weight_v, bias_v, train_v)

  def forward(self, x):
    random_weight = self.weight + torch.randn(size = self.weight.shape, device=self.weight.device, requires_grad=False)*self.weight_std()
    random_bias = self.bias + torch.randn(size = self.bias.shape, device=self.weight.device, requires_grad=False)*self.bias_std() if self.bias is not None else None
    out = F.conv2d(x, weight=random_weight, bias=random_bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
    return out

  def extra_repr(self):
    s = ('in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_size}'
         ', stride={stride}')
    if self.padding != (0,) * len(self.padding):
        s += ', padding={padding}'
    if self.dilation != (1,) * len(self.dilation):
        s += ', dilation={dilation}'
    if self.groups != 1:
        s += ', groups={groups}'
    if self.bias is None:
        s += ', bias=False'
    if self.padding_mode != 'zeros':
        s += ', padding_mode={padding_mode}'
    return s.format(**self.__dict__)


"""GhostNet is a deterministic network on which a StochNet can be built. It is initially created as an empty network, with no layers. Layers can be added via the add_block method."""

class GhostNet(nn.Module):

  def __init__(self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), name='gn'):
    super(GhostNet, self).__init__()
    self.name = name
    self.device = device

  #Add a Block to the network. A layer (usually a nn.Module) is required; preactivation and output function can be specified. The Block will be automatically created as an attribute. If a name is provided, the block can be easily called through {Block instance}.{name} and its layer through {Block instance}._{name}.
  def add_block(self, layer, b_name=None, name=None, act=None, post=None, act_name=None, post_name=None, dropout=0):
    if not hasattr(self, 'block_names'): self.block_names = []
    nb = len(self.block_names)
    b_name = f'B{nb}'
    full_name = b_name
    call_fct = True
    if name is None or name == '':
      call_fct = False
    else:
      full_name += '_' + name
    self.block_names.append(full_name)
    block = Block(layer, b_name=b_name, name=name, act=act, post=post, nb=nb, act_name=act_name, post_name=post_name, dropout=dropout)
    block.to(self.device)
    setattr(self, full_name, block)
    if call_fct:
      setattr(self, f'{name}', lambda : getattr(self, full_name))
      setattr(self, f'_{name}', lambda : getattr(self, full_name).layer)

  #List the network layers
  def get_layers(self):
    assert hasattr(self, 'block_names'), "Empty GhostNet: No block has been added."
    return [getattr(self, block).layer for block in self.block_names]

  #List the network blocks
  def get_blocks(self):
    assert hasattr(self, 'block_names'), "Empty GhostNet: No block has been added."
    return [getattr(self, block) for block in self.block_names]

  def first_block(self):
    return self.get_blocks()[0]

  def last_block(self):
    return self.get_blocks()[-1]
  
  def first_layer(self):
    return self.get_layers()[0]
  
  def last_layer(self):
    return self.get_layers()[-1]

  def forward(self, x):
    for block in self.get_blocks():
      x = block.forward(x)
    return x
  
  #Used for saving/loading
  def selfie(self):
    return [b.selfie() for b in self.get_blocks()]

  #Used for saving/loading
  @staticmethod
  def mirror(selfie, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), name='gn'):
    gn = GhostNet(device=device, name=name)
    for b in selfie:
      layer = b['layer'](**b['attrs'])
      gn.add_block(layer, **b['misc'])
    return gn
  
  def Clone(self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), name='gn'):
    gn = self.mirror(self.selfie(), device=device, name=name)
    gn.load_state_dict(self.state_dict())
    return gn
  
  #Train with Cross-Entropy minimization
  def Train(self, dataloader, epoch, lr, momentum=0.9, fullcheck=False, datacheck=None):
    self.train()
    CEL = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)
    for ep in range(epoch):
      running_loss = 0
      tot = 0
      for batch_idx, (data, targets) in enumerate(dataloader):
        data = data.to(self.device)
        batch_size = len(data)
        targets = targets.to(self.device)
        opt.zero_grad()
        outputs = self.forward(data)
        loss = CEL(outputs, targets)
        loss.backward()
        opt.step()
        tot += batch_size
        running_loss += loss.item()*batch_size
      Print(f'Epoch {ep+1} completed -- Average loss {running_loss/tot:.5f}')
      if fullcheck:
        with torch.no_grad():
          Print(f'Checking on train data')
          self.Test(dataloader)
          Print(f'Checking on check data')
          self.Test(datacheck)
    Print('Training Completed')

  def Test(self, dataloader):
    self.eval()
    correct = 0
    total = 0
    with torch.no_grad():
      for idx, (data, target) in enumerate(dataloader):
        data = data.to(self.device)
        target = target.to(self.device)
        outputs = self.forward(data)
        guess = torch.max(outputs, -1)[1]
        total += target.size(0)
        correct += (guess == target).sum().item()
    Print('Accuracy of the network: %0.3f %%' % (100 * correct / total))
    return correct/total

  #Run a training schedule. Epochs and learning rates need to be provided as lists.
  def TrainingSchedule(self, dataloader, EPOCH, LR, notes=None, **kwargs):
    assert type(EPOCH)==list and type(LR)==list
    assert len(EPOCH) == len(LR)
    Print('******************************************************')
    Print('************* Starting training schedule *************')
    Print(f'Network: {self.name}')
    Print(self)
    Print('Schedule:')
    Print('Epochs\tLR')
    for epoch, lr in zip(EPOCH, LR):
      Print(f'{epoch}\t{lr}')
    Print('------------------')
    Print('Objective: CEL')
    Print(f'Dataloader:')
    for key in dataloader.info:
      Print(f'{key}: {dataloader.info[key]}')
    if notes is not None:
      Print('------------------')
      Print('Additional notes:')
      Print(notes)
    for epoch, lr in zip(EPOCH, LR):
      Print('******************************************************')
      out = self.Train(dataloader, epoch, lr, **kwargs)
    Print('************* Training schedule completed ************')
    Print('******************************************************')

  #Create a StochNet centered on the current GhostNet
  def StochSon(self, delta=.025, weight_v=0.01, bias_v=0.01, train_v=True, name='sn', _best=None, keep_dropout=False):
    return StochNet(self, delta=delta, weight_v=weight_v, bias_v=bias_v, train_v=train_v, name=name, _best=_best, keep_dropout=keep_dropout)

  def Save(self, name=None, path=None, no_time=True, timestamp=None, stamp=None):
    if name is None: name = self.name
    if path is None: path = __OUT_DIR__()
    if stamp is not None: name += '_' + stamp
    if not no_time:
      if timestamp is None: timestamp = strftime("%Y%m%d-%H%M%S")
      name += '_' + timestamp
    dirpath = os.path.join(path, name)
    if os.path.exists(dirpath):
      try: assert os.path.isdir(dirpath)
      except AssertionError:
        Print(f'Could not save in {dirpath}')
    else: os.mkdir(dirpath)
    selfie_path = os.path.join(dirpath, 'selfie')
    dict_path = os.path.join(dirpath, 'state_dict')
    vers_path = os.path.join(dirpath, 'version')
    with open(vers_path, 'wb') as f:
      dill.dump(__SN_version__, f)
    with open(selfie_path, 'wb') as f:
      dill.dump(self.selfie(), f)
    torch.save(self.state_dict(), dict_path)

  @staticmethod
  def Load(path, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), name='gn', force_load=False):
    _path = path
    if not os.path.exists(path):
      path = os.path.join(__OUT_DIR__(), path)
      assert os.path.exists(path), f"{_path} not found"
    assert os.path.isdir(path), f"{_path} is not a directory"
    selfie_path = os.path.join(path, 'selfie')
    dict_path = os.path.join(path, 'state_dict')
    vers_path = os.path.join(path, 'version')
    if not force_load:
      try:
        with open(vers_path, 'rb') as f:
          assert dill.load(f) in __SN_comp_version__, f"Failed to load incompatible version: current compatible versions {__SN_comp_version__}, {_path} has version {dill.load(f)}"
      except FileNotFoundError:
        assert False, f"Failed to load incompatible version: current compatible versions {__SN_comp_version__}, unable to find version of {_path}"
    else:
      Print('Skipping version compatibility check')
    with open(selfie_path, 'rb') as f:
      selfie = dill.load(f)
    state_dict = torch.load(dict_path, map_location=device)
    gn = GhostNet.mirror(selfie, device=device)
    gn.load_state_dict(state_dict)
    return gn
  
  #Set a dropout for all blocks but the last one
  def set_dropout(self, dropout):
    for b in self.get_blocks()[:-1]:
      b.set_dropout(dropout)


"""StochNet is the child class of GhostNet dealing with stochastic networks. An instance is created starting from a GhostNet. Each layer is converted to a stochastic layer. The easiest way to create a StochNet is using the StochSon method from a GhostNet."""

class StochNet(GhostNet, nn.Module):

  #delta is the PAC-parameter for the bound, weight_v and bias_v the initial variances
  def __init__(self, mother, delta=.025, weight_v=None, bias_v=None, train_v=True, name='sn', _best=None, keep_dropout=False):
    super(StochNet, self).__init__()
    self.name = name
    self.device = mother.device
    self.weight_v = weight_v
    self.bias_v = bias_v
    self.train_v = train_v
    self._best = 1. if _best is None else _best
    for block in mother.get_blocks():
      layer = _make_stoch(block.layer, weight_v=weight_v, bias_v=bias_v, train_v=train_v)
      dropout=block._do if keep_dropout else 0
      self.add_block(layer, b_name=block.b_name, name=block.name, act=block.act, post=block.post, act_name=block._act_name, post_name=block._post_name, dropout=dropout)
    self.delta = delta
    self.register_buffer('Lambda', torch.zeros(1, device=self.device, requires_grad=True, dtype=torch.float32))

  def clamp_rho(self):
    for l in self.get_layers():
      l.clamp_rho()

  #Variances are trained
  def activate_vars(self):
    self.train_v = True
    for l in self.get_layers():
      l.activate_vars()

  #Only means are trained
  def deactivate_vars(self):
    self.train_v = False
    for l in self.get_layers():
      l.deactivate_vars()

  #Set the prior to the current posterior (to be used to zero the KL at the end of the prior's training)
  def ResetPrior(self):
    for l in self.get_layers():
      l.freeze_in()
  
  #Reset the values of the variances to a given value (just for the posterior)
  def reset_vars(self, weight_v, bias_v):
    for l in self.get_layers():
      l.set_vars(weight_v, bias_v)

  #Evaluate the KL penalty (KL + log(2m/delta)/m, where m is the size of the training dataset
  def Penalty(self, train_size):
    out = sum(l.Penalty() for l in self.get_layers())
    out = out + math.log(2*train_size**.5/self.delta)
    out = out / train_size
    return out

  def Test(self, dataloader, quiet=False, repeat=1):
    self.eval()
    correct_list = []
    with torch.no_grad():
      for _ in range(repeat):
        total = 0
        correct = 0
        for idx, (data, target) in enumerate(dataloader):
          data = data.to(self.device)
          target = target.to(self.device)
          outputs = self.forward(data)
          guess = torch.max(outputs, -1)[1]
          total += target.size(0)
          correct += (guess == target).sum().item()
        correct_list.append(correct)
    correct = sum(correct_list)/repeat
    if not quiet: Print('Accuracy of the network: %0.3f %%' % (100 * correct / total))
    return correct/total

  #Return the output of the last hidden layer
  def PreOUT(self, x):
    for block in self.get_blocks()[:-1]:
      x = block(x)
    return self.last_block().act(x)
    
  def Save(self, name=None, path=None, no_time = True, timestamp=None, stamp=None):
    if name is None: name = self.name
    if path is None: path = __OUT_DIR__()
    if stamp is not None: name += '_' + stamp
    if not no_time:
      if timestamp is None: timestamp = strftime("%Y%m%d-%H%M%S")
      name += '_' + timestamp
    dirpath = os.path.join(path, name)
    if os.path.exists(dirpath):
      try: assert os.path.isdir(dirpath)
      except AssertionError:
        Print(f'Could not save in {dirpath}')
    else: os.mkdir(dirpath)
    selfie_path = os.path.join(dirpath, 'selfie')
    dict_path = os.path.join(dirpath, 'state_dict')
    #mother_path = os.path.join(dirpath, 'mother_dict')
    misc_path = os.path.join(dirpath, 'misc')
    vers_path = os.path.join(dirpath, 'version')
    with open(vers_path, 'wb') as f:
      dill.dump(__SN_version__, f)
    with open(selfie_path, 'wb') as f:
      dill.dump(self.selfie(), f)
    with open(misc_path, 'wb') as f:
      dill.dump(dict(delta=self.delta, weight_v=self.weight_v, bias_v=self.bias_v, train_v=self.train_v, _best=self._best), f)
    torch.save(self.state_dict(), dict_path)
    #torch.save(self.mother_state, mother_path)
    
  def Clone(self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), name='sn'):
    selfie = self.selfie()
    misc = dict(delta=self.delta, weight_v=self.weight_v, bias_v=self.bias_v, train_v=self.train_v, _best=self._best)
    state_dict = self.state_dict()
    #mother_dict = self.mother_state
    gn = GhostNet.mirror(selfie, device=device)
    #gn.load_state_dict(mother_dict)
    sn = gn.StochSon(**misc, name=name)
    sn.load_state_dict(state_dict)
    return sn

  @staticmethod
  def Load(path, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), name='sn', force_load=False):
    _path = path
    if not os.path.exists(path):
      path = os.path.join(__OUT_DIR__(), path)
      assert os.path.exists(path), f"{_path} not found"
    assert os.path.isdir(path), f"{_path} is not a directory"
    selfie_path = os.path.join(path, 'selfie')
    dict_path = os.path.join(path, 'state_dict')
    misc_path = os.path.join(path, 'misc')
    vers_path = os.path.join(path, 'version')
    if not force_load:
      try:
        with open(vers_path, 'rb') as f:
          assert dill.load(f) in __SN_comp_version__, f"Failed to load incompatible version: current compatible versions {__SN_comp_version__}, {_path} has version {dill.load(f)}"
      except FileNotFoundError:
        assert False, f"Failed to load incompatible version: current compatible versions {__SN_comp_version__}, unable to find version of {_path}"
    with open(selfie_path, 'rb') as f:
      selfie = dill.load(f)
    with open(misc_path, 'rb') as f:
      misc = dill.load(f)
    state_dict = torch.load(dict_path, map_location=device)
    gn = GhostNet.mirror(selfie, device=device)
    sn = gn.StochSon(**misc, name=name)
    sn.load_state_dict(state_dict)
    return sn

  def Conditional01(self, input, target, multi=False, samples=100):
    Phi = self.PreOUT(input)
    last_layer = self.last_block().layer
    M = F.linear(Phi, last_layer.weight, last_layer.bias)
    weight_v = torch.square(last_layer.weight_std())
    bias_v = torch.square(last_layer.bias_std()) if last_layer.bias_rho is not None else None
    Q = F.linear(Phi**2, weight_v, bias_v)
    if last_layer.out_features==2 and not multi:
      Z = (M[..., 0] - M[..., 1])/torch.sqrt(Q.sum(-1) + __EPS__)
      return ((1-target)*gauss_ccdf(Z) + target*gauss_ccdf(-Z)).mean()
    else:
      classes = last_layer.out_features
      perm = _mk_perm(classes, device=self.device) #perm : [classes, classes, classes]
      PERM = perm[target] #PERM : [batch, classes, classes]
      Q = (PERM @ Q.unsqueeze(-1)).squeeze(-1) #Q : [batch, classes]
      M = (PERM @ M.unsqueeze(-1)).squeeze(-1) #M : [batch, classes]
      a = torch.sqrt(Q[..., :-1]/(Q[..., -1].unsqueeze(-1) + __EPS__)).unsqueeze(-2) #a : [batch, 1, classes-1]
      m = ((M[..., :-1] - M[..., -1].unsqueeze(-1))/torch.sqrt(Q[..., -1].unsqueeze(-1) + __EPS__)).unsqueeze(-2) #m : [batch, 1, classes-1]
      X = torch.randn(samples, classes-1, device=self.device, requires_grad=False) #X : [samples, classes-1]
      Z = a*X + m #Z : [batch, samples, classes-1]
      out = 1 - gauss_ccdf(Z.max(-1)[0]).mean()
      return out

  #Run a training schedule. Epochs and learning rates need to be provided as lists.
  def TrainingSchedule(self, dataloader, EPOCH, LR, procedure='cond', save_track=True, notes=None, no_time=True, stamp=None, method='invKL', **kwargs):
    timestamp = strftime("%Y%m%d-%H%M%S")
    assert type(EPOCH)==list and type(LR)==list
    assert len(EPOCH) == len(LR)
    assert procedure in ['cond', 'std']
    Print('******************************************************')
    Print('************* Starting training schedule *************')
    Print(f'Network: {self.name}')
    Print(self)
    Print('Initial variances:')
    weight_v = self.weight_v
    bias_v = self.bias_v
    Print(f'weight: {self.weight_v}')
    Print(f'bias (if defined): {self.bias_v}')
    Print('------------------')
    Print(f'Procedure: {procedure}')
    Print(f'Objective: {method}')
    Print(f'Training variances: {self.train_v}')
    Print('Schedule:')
    Print('Epochs\tLR')
    for epoch, lr in zip(EPOCH, LR):
      Print(f'{epoch}\t{lr}')
    Print('------------------')
    Print(f'Dataloader:')
    for key in dataloader.info:
      Print(f'{key}: {dataloader.info[key]}')
    if notes is not None:
      Print('------------------')
      Print('Additional notes:')
      Print(notes)
    OUT = [[], [], [], []]
    for epoch, lr in zip(EPOCH, LR):
      Print('******************************************************')
      if procedure == 'std': out = self.Train(dataloader, epoch, lr, timestamp=timestamp, stamp=stamp, no_time=no_time, **kwargs)
      elif procedure == 'cond': out = self.CondTrain(dataloader, epoch, lr, timestamp=timestamp, stamp=stamp, no_time=no_time, **kwargs)
      else: assert False, f"Undefined procedure {procedure}"
      if type(out) is tuple: #track=True, track_lbd might be True or False
        for i, o in enumerate(out):
          OUT[i] += o
      elif type(out) is list: #track=False, track_lbd=True
        OUT[0].append(out)
    Print('************* Training schedule completed ************')
    Print('******************************************************')
    if len(OUT[0]) > 0:
      if len(OUT[1]) == 0:
        return OUT[0]
      else:
        out = []
        for o in OUT:
          if len(o)>0: out.append(o)
        out = tuple(out)
        if save_track:
          name = 'Progr_' + self.name
          if stamp is not None: name += '_' + stamp
          if not no_time: name += '_' + timestamp
          save_lists(*out, name=name)
        return out

  #Training via the Cond-Gauss algorithm
  def CondTrain(self, dataloader, epoch, lr, momentum=0.9, train_size=None, multi=False, samples=100, track=False, method='invKL', penalty=1, adam=False, lbd_in=None, lr_lbd=None, track_lbd=False, no_save=False, no_time=True, timestamp=None, repeat=1, new=False, stamp=None):
    self.train()
    Conditional01 = self.Conditional01 if not new else self.NewConditional01
    assert method in __methods__
    Print(f'Training started -- epochs: {epoch} -- lr: {lr} -- method: cond-{method}')
    Print(f'Penalty factor: {penalty}')
    if not adam:
      Print(f'Training via SGD with momentum: {momentum}')
    else:
      Print(f'Training via Adam')
    if not multi and self.last_block().layer.out_features==2:
      Print(f'Using binary loss')
    else:
      Print(f'Using multiclass loss with {samples} samples')
    if track:
      loss_list = []
      KL_list = []
      bound_list = []
    if train_size is None:
      train_size = sum([len(data) for data, _ in dataloader])
    optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum) if not adam else torch.optim.Adam(self.parameters(), lr=lr)
    if method == 'lbd':
      if lr_lbd is None: lr_lbd = lr
      if lbd_in is not None:
        assert lbd_in<1 and lbd_in>0
        self.Lambda = torch.tensor([math.atanh(2*lbd_in-1)], device=self.device, requires_grad=True, dtype=torch.float32)
      optimizer_lbd = torch.optim.SGD([self.Lambda], lr=lr_lbd, momentum=momentum) if not adam else torch.optim.Adam([self.Lambda], lr=lr_lbd)
      lbd_epoch = False
      epoch = 2*epoch
    if track_lbd: lbd_list = []
    for ep in range(epoch):
      if method != 'lbd':
        Print(f'Starting epoch {ep+1}')
        opt = optimizer
        ep_track = track
        track_lbd = False
        ep_track_lbd = False
      else:
        if lbd_epoch:
          Print(f'Starting epoch {ep//2+1} - lbd')
          opt = optimizer_lbd
          ep_track = track
          ep_track_lbd = track_lbd
        else:
          Print(f'Starting epoch {ep//2+1}')
          opt = optimizer
          ep_track = False
          ep_track_lbd = False
        lbd_epoch = not lbd_epoch
      running_loss = 0
      tot = 0
      if ep_track:
        KL_track = 0
        loss_track = 0
        bound_track = 0
      for batch_idx, (data, targets) in enumerate(dataloader):
        data = data.to(self.device)
        batch_size = len(data)
        targets = targets.to(self.device)
        opt.zero_grad()
        KL = self.Penalty(train_size)
        if ep_track: KL_track += KL.item()*batch_size
        loss = 0.
        for rep in range(repeat):
          loss = loss + Conditional01(data, targets, multi=multi, samples=samples)/repeat
        #Print(f'Epoch {ep+1} batch {batch_idx+1} KL {KL.item()} loss {loss.item()}')
        if ep_track:
          with torch.no_grad():
            loss_track += loss.item()*batch_size
            bound_track += inv_KL(loss.item(), KL.item())*batch_size
        if method == 'invKL':
          loss = invkl(loss, penalty*KL)
          #Print(f'Epoch {ep+1} batch {batch_idx+1} guessed bound {loss.item()}')
        elif method == 'quad':
          loss = (torch.sqrt(loss + penalty*KL/2) + torch.sqrt(penalty*KL/2))**2
        elif method == 'McAll':
          loss = loss + torch.sqrt(penalty*KL/2)
        elif method == 'lbd':
          loss = 4*(loss + 2*penalty*KL/(1+torch.tanh(self.Lambda)))/(3-torch.tanh(self.Lambda))
        elif method == 'ERM':
          loss = loss
        loss.backward()
        opt.step()
        if __CLAMP_RHO__ and self.train_v: self.clamp_rho()
        tot += batch_size
        running_loss += loss.item()*batch_size
      if ep_track:
        loss_list.append(loss_track/tot)
        KL_list.append(KL_track/tot)
        bound_list.append(bound_track/tot)
        if not no_save:
          if bound_track/tot < self._best:
            self._best = bound_track/tot
            self.Save(name='Best_'+self.name, no_time=no_time, timestamp=timestamp, stamp=stamp)
      if ep_track_lbd: lbd_list.append((1+math.tanh(self.Lambda.item()))/2)
      if method != 'lbd': Print(f'Epoch {ep+1} completed -- Average loss {running_loss/tot:.5f}')
      elif not lbd_epoch: Print(f'Epoch {ep//2+1} completed -- Average loss {running_loss/tot:.5f}')
    Print('Training Completed')
    if track and not track_lbd: return loss_list, KL_list, bound_list
    if track and track_lbd: return loss_list, KL_list, bound_list, lbd_list
    if track_lbd: return lbd_list

  #Training via the standard algorithm
  def Train(self, dataloader, epoch, lr, momentum=0.9, train_size=None, track=False, method='McAll', pmin=None, penalty=1, adam=False, lbd_in=0.5, lr_lbd=None, track_lbd=False, samples=100, no_save=False, no_time = True, timestamp=None, repeat=1, stamp=None):
    self.train()
    assert method in __methods__
    Print(f'Training started -- epochs: {epoch} -- lr: {lr} -- method: {method}')
    Print(f'Penalty factor: {penalty}')
    Print(f'Momentum: {momentum}')
    if train_size is None:
      train_size = sum([len(data) for data, _ in dataloader])
    optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum) if not adam else torch.optim.Adam(self.parameters(), lr=lr)
    if method == 'lbd':
      if lr_lbd is None: lr_lbd = lr
      if lbd_in is not None:
        assert lbd_in<1 and lbd_in>0
        self.Lambda = torch.tensor([math.atanh(2*lbd_in-1)], device=self.device, requires_grad=True, dtype=torch.float32)
      optimizer_lbd = torch.optim.SGD([self.Lambda], lr=lr_lbd, momentum=momentum) if not adam else torch.optim.Adam([self.Lambda], lr=lr_lbd)
      lbd_epoch = False
      epoch = 2*epoch
    CEL = nn.CrossEntropyLoss()
    NLLL = nn.NLLLoss()
    if track:
      loss_list = []
      KL_list = []
      bound_list = []
    if track_lbd: lbd_list = []
    for ep in range(epoch):
      if method != 'lbd':
        Print(f'Starting epoch {ep+1}')
        opt = optimizer
        ep_track = track
        track_lbd = False
        ep_track_lbd = False
      else:
        if lbd_epoch:
          Print(f'Starting epoch {ep//2+1} - lbd')
          opt = optimizer_lbd
          ep_track = track
          ep_track_lbd = track_lbd
        else:
          Print(f'Starting epoch {ep//2+1}')
          opt = optimizer
          ep_track = False
          ep_track_lbd = False
        lbd_epoch = not lbd_epoch
      running_loss = 0
      tot = 0
      if ep_track:
        KL_track = 0
        loss_track = 0
        bound_track = 0
      for batch_idx, (data, targets) in enumerate(dataloader):
        data = data.to(self.device)
        batch_size = len(data)
        targets = targets.to(self.device)
        opt.zero_grad()
        KL = self.Penalty(train_size)
        if ep_track: KL_track += KL.item()*batch_size
        if self.last_block().layer.out_features == 2 and pmin is None:
          loss = 0.
          for rep in range(repeat):
            outputs = self.forward(data)
            loss = loss + (CEL(outputs, targets)/math.log(2))/repeat
        else:
          if pmin is None: pmin = 10**-5
          loss = 0.
          for rep in range(repeat):
            outputs = self.forward(data)
            probs = torch.clamp(torch.softmax(outputs, dim=-1), min=pmin)
            loss = loss + (NLLL(torch.log(probs), targets)/math.log(1/pmin))/repeat
        if ep_track:
          with torch.no_grad():
            loss01 = self.Conditional01(data, targets, samples=samples).item()
            loss_track += loss01*batch_size
            bound_track += inv_KL(loss01, KL.item())*batch_size
        if method == 'invKL':
          loss = invkl(loss, penalty*KL)
        elif method == 'quad':
          loss = (torch.sqrt(loss + penalty*KL/2) + torch.sqrt(penalty*KL/2))**2
        elif method == 'McAll':
          loss = loss + torch.sqrt(penalty*KL/2)
        elif method == 'lbd':
          loss = 4*(loss + 2*penalty*KL/(1+torch.tanh(self.Lambda)))/(3-torch.tanh(self.Lambda))
        elif method == 'ERM':
          loss = loss
        loss.backward()
        opt.step()
        if __CLAMP_RHO__ and self.train_v: self.clamp_rho()
        tot += batch_size
        running_loss += loss.item()*batch_size
      if ep_track:
        loss_list.append(loss_track/tot)
        KL_list.append(KL_track/tot)
        bound_list.append(bound_track/tot)
        if not no_save:
          if bound_track/tot < self._best:
            self._best = bound_track/tot
            self.Save(name='Best_'+self.name, no_time=no_time, timestamp=timestamp, stamp=stamp)
      if ep_track_lbd: lbd_list.append((1+math.tanh(self.Lambda.item()))/2)
      if method != 'lbd': Print(f'Epoch {ep+1} completed -- Average loss {running_loss/tot:.5f}')
      elif not lbd_epoch: Print(f'Epoch {ep//2+1} completed -- Average loss {running_loss/tot:.5f}')
    Print('Training Completed')
    if track and not track_lbd: return loss_list, KL_list, bound_list
    if track and track_lbd: return loss_list, KL_list, bound_list, lbd_list
    if track_lbd: return lbd_list
  
  #Find bound where the loss is replaced by an estimate averaged on {repeat} realizations
  def GuessBound(self, dataloader, train_size=None, repeat=1):
    Print('******************************************************')
    Print(f'Evaluating guessed bound for {self.name}')
    Print(f'Dataloader:')
    for key in dataloader.info:
      Print(f'{key}: {dataloader.info[key]}')
    with torch.no_grad():
      emp_score = 0.
      if train_size is None:
        train_size = sum([len(data) for data, _ in dataloader])
      emp_loss = 1 - self.Test(dataloader, repeat=repeat)
      Print('------------------')
      Print(f'Empirical loss estimate (x{repeat}): {emp_loss}')
      penalty = self.Penalty(train_size).item()
      Print(f'Penalty: {penalty}')
      bound = inv_KL(emp_loss, penalty)
      Print(f'Bound: {bound}')
    Print('******************************************************')
    return bound

  #Actual generalization bound where the upperbound on the loss is obtained with {N_Nets} realizations
  def PrintBound(self, dataloader, N_nets, train_size=None, deltap=0.01):
    Print('******************************************************')
    Print('*********** Evaluating PAC-Bayesian bound ************')
    Print(f'Network: {self.name}')
    Print(self)
    Print('------------------')
    Print(f'Dataloader:')
    for key in dataloader.info:
      Print(f'{key}: {dataloader.info[key]}')
    Print('******************************************************')
    with torch.no_grad():
      emp_score = 0.
      if train_size is None:
        train_size = sum(len(data) for data, _ in dataloader)
      for cnt in range(N_nets):
        emp_score += self.Test(dataloader, quiet=True)/N_nets
        if (cnt+1)*1000 % N_nets == 0: Print(f'Net {cnt+1} of {N_nets}: Current average score {emp_score*N_nets/(cnt+1):.5e}')
      emp_loss = 1 - emp_score
      Print(f'Empirical loss estimate: {emp_loss}')
      emp_loss_bound = inv_KL(emp_loss, math.log(2/deltap)/N_nets)
      Print(f'Bound on the empirical loss: {emp_loss_bound}')
      penalty = self.Penalty(train_size).item()
      Print('penalty', penalty)
      bound = inv_KL(emp_loss_bound, penalty)
    Print(f'With P >= {1-self.delta-deltap}, the true error is bounded by {bound}')
    Print(f'Guessed bound: {inv_KL(emp_loss, penalty)}')
    Print('******************************************************')
    Print('**************** Evaluation completed ****************')
    return bound

  def clamp_rho_abs(self, min):
    for l in self.get_layers():
      l.clamp_rho_abs(min)
