require 'torch'
require 'nn'
require 'nngraph'

require 'densecap.LanguageModel'
require 'densecap.LocalizationLayer'
require 'densecap.modules.BoxRegressionCriterion'
require 'densecap.modules.BilinearRoiPooling'
require 'densecap.modules.ApplyBoxTransform'
require 'densecap.modules.LogisticCriterion'
require 'densecap.modules.PosSlicer'
require 'densecap.modules.BoxIoU'

local box_utils = require 'densecap.box_utils'
local utils = require 'densecap.utils'

local debugger = require('fb.debugger')
local DenseCapModel, parent = torch.class('DenseCapModel', 'nn.Module')


function DenseCapModel:__init(opt, pretrained_model)

  local net_utils = require 'densecap.net_utils'
  opt = opt or {}  
  opt.cnn_name = utils.getopt(opt, 'cnn_name', 'vgg-16')
  opt.backend = utils.getopt(opt, 'backend', 'cudnn')
  opt.path_offset = utils.getopt(opt, 'path_offset', '')
  opt.dtype = utils.getopt(opt, 'dtype', 'torch.CudaTensor')
  opt.vocab_size = utils.getopt(opt, 'vocab_size')
  opt.std = utils.getopt(opt, 'std', 0.01) -- Used to initialize new layers

  -- For test-time handling of final boxes
  opt.final_nms_thresh = utils.getopt(opt, 'final_nms_thresh', 0.3)

  -- Ensure that all options for loss were specified
  utils.ensureopt(opt, 'mid_box_reg_weight')
  utils.ensureopt(opt, 'mid_objectness_weight')
  utils.ensureopt(opt, 'end_box_reg_weight')
  utils.ensureopt(opt, 'end_objectness_weight')
  utils.ensureopt(opt, 'captioning_weight')
  
  -- Options for RNN
  opt.seq_length = utils.getopt(opt, 'seq_length')
  opt.rnn_encoding_size = utils.getopt(opt, 'rnn_encoding_size', 512)
  opt.rnn_size = utils.getopt(opt, 'rnn_size', 512)
  self.opt = opt -- TODO: this is... naughty. Do we want to create a copy instead?
  
  -- This will hold various components of the model
  self.nets = {}
  
  -- This will hold the whole model
  self.net = nn.Sequential()
  
  -- Load the CNN from disk
  local cnn = net_utils.load_cnn(opt.cnn_name, opt.backend, opt.path_offset)
  
  -- We need to chop the CNN into three parts: conv that is not finetuned,
  -- conv that will be finetuned, and fully-connected layers. We'll just
  -- hardcode the indices of these layers per architecture.
  local conv_start1, conv_end1, conv_start2, conv_end2
  local recog_start, recog_end
  local fc_dim
  if opt.cnn_name == 'vgg-16' then
    conv_start1, conv_end1 = 1, 10 -- these will not be finetuned for efficiency
    conv_start2, conv_end2 = 11, 30 -- these will be finetuned possibly
    recog_start, recog_end = 32, 38 -- FC layers
    opt.input_dim = 512
    opt.output_height, opt.output_width = 7, 7
    fc_dim = 4096
  else
    error(string.format('Unrecognized CNN "%s"', opt.cnn_name))
  end

  -- Now that we have the indices, actually chop up the CNN.
  if pretrained_model ~= nil then
    -- pretrained model
    self.nets.conv_net1 = pretrained_model.net:get(1)
    self.nets.conv_net2 = pretrained_model.net:get(2)
    print('Successfully load conv_net 1 and 2')
  else 
    self.nets.conv_net1 = net_utils.subsequence(cnn, conv_start1, conv_end1)
    self.nets.conv_net2 = net_utils.subsequence(cnn, conv_start2, conv_end2)
  end
  self.net:add(self.nets.conv_net1)
  self.net:add(self.nets.conv_net2)

  -- Figure out the receptive fields of the CNN
  -- TODO: Should we just hardcode this too per CNN?
  local conv_full = net_utils.subsequence(cnn, conv_start1, conv_end2)
  local x0, y0, sx, sy = net_utils.compute_field_centers(conv_full)
  self.opt.field_centers = {x0, y0, sx, sy}

  -- Localization layer
  if pretrained_model ~= nil then
    self.nets.localization_layer = pretrained_model.net:get(3)
  else 
    self.nets.localization_layer = nn.LocalizationLayer(opt)
  end
  self.net:add(self.nets.localization_layer)

  --debugger.enter()  

  -- Recognition base network; FC layers from VGG.
  -- Produces roi_codes of dimension fc_dim.
  -- TODO: Initialize this from scratch for ResNet?
  if pretrained_model ~= nil then 
    self.nets.recog_base = pretrained_model.nets.recog_base
  else 
    self.nets.recog_base = net_utils.subsequence(cnn, recog_start, recog_end)
  end 

  -- Objectness branch; outputs positive / negative probabilities for final boxes
  if pretrained_model ~= nil then
    self.nets.objectness_branch = pretrained_model.nets.objectness_branch
  else 
    self.nets.objectness_branch = nn.Linear(fc_dim, 1)
    self.nets.objectness_branch.weight:normal(0, opt.std)
    self.nets.objectness_branch.bias:zero()
  end

  -- Final box regression branch; regresses from RPN boxes to final boxes
  if pretrained_model ~= nil then
    self.nets.box_reg_branch = pretrained_model.nets.box_reg_branch
  else
    self.nets.box_reg_branch = nn.Linear(fc_dim, 4)
    self.nets.box_reg_branch.weight:zero()
    self.nets.box_reg_branch.bias:zero()
  end

  -- Set up LanguageModel
  local lm_opt = {
    vocab_size = opt.vocab_size,
    input_encoding_size = opt.rnn_encoding_size,
    rnn_size = opt.rnn_size,
    seq_length = opt.seq_length,
    idx_to_token = opt.idx_to_token,
    image_vector_dim=fc_dim,
  }
  self.nets.language_model = nn.LanguageModel(lm_opt)

  self.nets.recog_net = self:_buildRecognitionNet()
  self.net:add(self.nets.recog_net)

  --self.nets.languageEncoder = self:_buildLanguageEncoder()
  --self.net:add(self.nets.languageEncoder)

  --self.nets.fusingModel = self._buildFusingModel()
  --self.nets.languageEncoder = self:_buildLanguageEncoder()
  --self.net:add(self.nets.languageEncoder)

  -- Set up Criterions
  self.crits = {}
  self.crits.objectness_crit = nn.LogisticCriterion()
  self.crits.box_reg_crit = nn.BoxRegressionCriterion(opt.end_box_reg_weight)
  self.crits.score_crit = nn.CrossEntropyCriterion()
  --self.crits.IoUs_crit = nn.SmoothL1Criterion()
  --self.crits.lm_crit = nn.TemporalCrossEntropyCriterion()

  self:training()
  self.finetune_cnn = false
end

function DenseCapModel:_buildSelectModel(length)

  local objectness_scores = nn.Identity()()
  local pos_roi_boxes = nn.Identity()()
  local final_box_trans = nn.Identity()()
  local final_boxes = nn.Identity()()
  local lm_output = nn.Identity()()
  local gt_boxes = nn.Identity()()
  local gt_labels = nn.Identity()()
  local pos_roi_feat = nn.Identity()()

  local selectModel = nn.ConcatTable()
  for i = 1, length:size(1) do
    local sel = nn.Sequential()
    sel:add(nn.SelectTable(i))
    sel:add(nn.Select(1,length[i]))
    sel:add(nn.View(1,-1))
    selectModel:add(sel)
  end
  local lm_output_tem = nn.SplitTable(1)(lm_output)
  local lm_output_end = selectModel(lm_output_tem)
  lm_output_end = nn.JoinTable(1)(lm_output_end)
  --local lm_output_end = nn.SplitTable(1)(lm_output)
  local inputs = {
    objectness_scores,
    pos_roi_boxes, final_box_trans, final_boxes,
    lm_output,
    gt_boxes, gt_labels, pos_roi_feat
  }
  local outputs = {
    objectness_scores,
    pos_roi_boxes, final_box_trans, final_boxes,
    lm_output_end,
    gt_boxes, gt_labels, pos_roi_feat
  }

  
  local mod = nn.gModule(inputs, outputs)
  mod.name = 'selectModel'
  return mod

end
-- TODO input the mask
function DenseCapModel:_buildLanguageEncoder()

  local objectness_scores = nn.Identity()()
  local pos_roi_boxes = nn.Identity()()
  local final_box_trans = nn.Identity()()
  local final_boxes = nn.Identity()()
  local gt_boxes = nn.Identity()()
  local gt_labels = nn.Identity()()
  local pos_roi_feat = nn.Identity()()
  local phrase = nn.Identity()()

  local lm_output = self.nets.language_model(phrase) -- N,16,512

  local inputs = {
    objectness_scores,
    pos_roi_boxes, final_box_trans, final_boxes,
    gt_boxes, gt_labels , pos_roi_feat, phrase
  }

  local outputs = {
    objectness_scores,
    pos_roi_boxes, final_box_trans, final_boxes,
    lm_output,
    gt_boxes, gt_labels, pos_roi_feat
  }

  local mod = nn.gModule(inputs, outputs)
  mod.name = 'language_encoder'
  return mod

end

function DenseCapModel:_buildFusingModel(num_proposals)

  local objectness_scores = nn.Identity()()
  local pos_roi_boxes = nn.Identity()()
  local final_box_trans = nn.Identity()()
  local final_boxes = nn.Identity()()
  local lm_output_end = nn.Identity()()
  local gt_boxes = nn.Identity()()
  local gt_labels = nn.Identity()()
  local pos_roi_feat = nn.Identity()()

  -- take average
  --lm_output = nn.Mean(2)(lm_output)

  local ious_input = {lm_output_end, pos_roi_feat} 
  local pred_IoUs = nn.MM(false, true)(ious_input)
  -- Andrew
  local pred_scores = nn.Linear(num_proposals,num_proposals)
 
  local inputs = {
    objectness_scores,
    pos_roi_boxes, final_box_trans, final_boxes,
    lm_output_end,
    gt_boxes, gt_labels, pos_roi_feat
  }

  local outputs = {
    objectness_scores,
    pos_roi_boxes, final_box_trans, final_boxes,
    lm_output_end,
    gt_boxes, gt_labels , pred_IoUs
  }

  local mod = nn.gModule(inputs, outputs)
  mod.name = 'fusingModel'
  return mod
   
end

function DenseCapModel:_buildRecognitionNet()

  local roi_feats = nn.Identity()()
  local roi_boxes = nn.Identity()()
  local gt_boxes = nn.Identity()()
  local gt_labels = nn.Identity()()

  local roi_codes = self.nets.recog_base(roi_feats)
  local objectness_scores = self.nets.objectness_branch(roi_codes)

  local pos_roi_codes = nn.PosSlicer(){roi_codes, gt_labels}
  local pos_roi_boxes = nn.PosSlicer(){roi_boxes, gt_boxes}
  
  local final_box_trans = self.nets.box_reg_branch(pos_roi_codes)
  local final_boxes = nn.ApplyBoxTransform(){pos_roi_boxes, final_box_trans}

  local pos_roi_feat = nn.Linear(4096,512)(pos_roi_codes)

  -- Annotate nodes
  roi_codes:annotate{name='recog_base'}
  objectness_scores:annotate{name='objectness_branch'}
  pos_roi_codes:annotate{name='code_slicer'}
  pos_roi_boxes:annotate{name='box_slicer'}
  final_box_trans:annotate{name='box_reg_branch'}

  local inputs = {roi_feats, roi_boxes, gt_boxes, gt_labels}
  local outputs = {
    objectness_scores,
    pos_roi_boxes, final_box_trans, final_boxes,
    gt_boxes, gt_labels , pos_roi_feat
  }

  local mod = nn.gModule(inputs, outputs)
  mod.name = 'recognition_network'
  return mod
end


function DenseCapModel:training()
  parent.training(self)
  self.net:training()
end


function DenseCapModel:evaluate()
  parent.evaluate(self)
  self.net:evaluate()
end


--[[
Set test-time parameters for this DenseCapModel.

Input: Table with the following keys:
- rpn_nms_thresh: NMS threshold for region proposals in the RPN; default is 0.7.
- final_nms_thresh: NMS threshold for final predictions; default is 0.3.
- num_proposals: Number of proposals to use; default is 1000
--]]
function DenseCapModel:setTestArgs(kwargs)
  self.nets.localization_layer:setTestArgs{
    nms_thresh = utils.getopt(kwargs, 'rpn_nms_thresh', 0.7),
    max_proposals = utils.getopt(kwargs, 'num_proposals', 1000)
  }
  self.opt.final_nms_thresh = utils.getopt(kwargs, 'final_nms_thresh', 0.3)
end


--[[
Convert this DenseCapModel to a particular datatype, and convert convolutions
between cudnn and nn.
--]]
function DenseCapModel:convert(dtype, use_cudnn)
  self:type(dtype)
  if cudnn and use_cudnn ~= nil then
    local backend = nn
    if use_cudnn then
      backend = cudnn
    end
    cudnn.convert(self.net, backend)
    cudnn.convert(self.nets.localization_layer.nets.rpn, backend)
  end
end


--[[
Run the model forward.

Input:
- image: Pixel data for a single image of shape (1, 3, H, W)

After running the model forward, we will process N regions from the
input image. At training time we have access to the ground-truth regions
for that image, and assume that there are P ground-truth regions. We assume
that the language model has a vocabulary of V elements (including the END
token) and that all captions have been padded to a length of L.

Output: A table of
- objectness_scores: Array of shape (N, 1) giving (final) objectness scores
  for boxes.
- pos_roi_boxes: Array of shape (P, 4) at training time and (N, 4) at test-time
  giving the positions of RoI boxes in (xc, yc, w, h) format.
- final_box_trans: Array of shape (P, 4) at training time and (N, 4) at
  test-time giving the transformations applied on top of the region proposal
  boxes by the final box regression.
- final_boxes: Array of shape (P, 4) at training time and (N, 4) at test-time
  giving the coordinates of the final output boxes, in (xc, yc, w, h) format.
- lm_output: At training time, an array of shape (P, L+2, V) giving log
  probabilities (the +2 is two extra time steps for CNN input and END token).
  At test time, an array of shape (N, L) where each element is an integer in
  the range [1, V] giving sampled captions.
- gt_boxes: At training time, an array of shape (P, 4) giving ground-truth
  boxes corresponding to the sampled positives. At test time, an empty tensor.
- gt_labels: At training time, an array of shape (P, L) giving ground-truth
  sequences for sampled positives. At test-time, and empty tensor.
--]]
function DenseCapModel:updateOutput(input)
  -- Make sure the input is (1, 3, H, W)

  assert(input:dim() == 4 and input:size(1) == 1 and input:size(2) == 3)
  local H, W = input:size(3), input:size(4)
  self.nets.localization_layer:setImageSize(H, W)
  if self.train then
    assert(not self._called_forward,
      'Must call setGroundTruth before training-time forward pass')
    self._called_forward = true
  end

  if self.train then
    self.output = self.net:forward(input)
  else
    local out = self.net:forward(input)

    table.insert(out, self.gt_labels[1])
    self.nets.languageEncoder = self:_buildLanguageEncoder()

    out = self.nets.languageEncoder:forward(out)
    out[5] = out[5]:float()
    self.nets.selectModel = self:_buildSelectModel(self.gt_length)
    out = self.nets.selectModel:forward(out)
    out[8] = out[8]:float()
    self.nets.fusingModel = self:_buildFusingModel(out[2]:size(1))
    out = self.nets.fusingModel:forward(out)

    out[5] = out[5]:cuda()
    out[8] = out[8]:cuda()
    self.output = out
    --[[
    local cnn1_out = self.net:get(1):forward(input)
    local cnn2_out = self.net:get(2):forward(cnn1_out)
    local localization_out = self.net:get(3):forward(cnn2_out)
    local recognition_out = self.net:get(4):forward(localization_out)
    recognition_out[6] = self.gt_labels:view(self.gt_labels:size(2),-1)
    self.output = self.net:get(5):forward(recognition_out)
    --]]
  end
  -- At test-time, apply NMS to final boxes
  local verbose = false
  if verbose then
    print(string.format('before final NMS there are %d boxes', self.output[4]:size(1)))
    print(string.format('Using NMS threshold of %f', self.opt.final_nms_thresh))
  end
  if not self.train and self.opt.final_nms_thresh > 0 then
    -- We need to apply the same NMS mask to the final boxes, their
    -- objectness scores, and the output from the language model
    local final_boxes_float = self.output[4]:float()
    local class_scores_float = self.output[1]:float()
    --local lm_output_float = self.output[5]:float()
    local boxes_scores = torch.FloatTensor(final_boxes_float:size(1), 5)
    local boxes_x1y1x2y2 = box_utils.xcycwh_to_x1y1x2y2(final_boxes_float)
    boxes_scores[{{}, {1, 4}}]:copy(boxes_x1y1x2y2)
    boxes_scores[{{}, 5}]:copy(class_scores_float[{{}, 1}])
    local idx = box_utils.nms(boxes_scores, self.opt.final_nms_thresh)
    self.output[4] = final_boxes_float:index(1, idx):typeAs(self.output[4])
    self.output[1] = class_scores_float:index(1, idx):typeAs(self.output[1])
    --self.output[5] = lm_output_float:index(1, idx):typeAs(self.output[5])

    -- TODO: In the old StnDetectionModel we also applied NMS to the
    -- variables dumped by the LocalizationLayer. Do we want to do that?
  end

  return self.output
end


function DenseCapModel:extractFeatures(input)
  -- Make sure the input is (1, 3, H, W)
  assert(input:dim() == 4 and input:size(1) == 1 and input:size(2) == 3)
  local H, W = input:size(3), input:size(4)
  self.nets.localization_layer:setImageSize(H, W)

  local output = self.net:forward(input)
  local final_boxes_float = output[4]:float()
  local class_scores_float = output[1]:float()
  local boxes_scores = torch.FloatTensor(final_boxes_float:size(1), 5)
  local boxes_x1y1x2y2 = box_utils.xcycwh_to_x1y1x2y2(final_boxes_float)
  boxes_scores[{{}, {1, 4}}]:copy(boxes_x1y1x2y2)
  boxes_scores[{{}, 5}]:copy(class_scores_float)
  local idx = box_utils.nms(boxes_scores, self.opt.final_nms_thresh)

  local boxes_xcycwh = final_boxes_float:index(1, idx):typeAs(self.output[4])
  local feats = self.nets.recog_base.output:float():index(1, idx):typeAs(self.output[4])

  return boxes_xcycwh, feats
end


--[[
Run a test-time forward pass, plucking out only the relevant outputs.

Input: Tensor of shape (1, 3, H, W) giving pixels for an input image.

Returns:
- final_boxes: Tensor of shape (N, 4) giving coordinates of output boxes
  in (xc, yc, w, h) format.
- objectness_scores: Tensor of shape (N, 1) giving objectness scores of
  those boxes.
- captions: Array of length N giving output captions, decoded as strings.
--]]
function DenseCapModel:forward_test(input)
  self:evaluate()
  self:setGroundTruth(nil, input[2], input[3])

  local out = self:forward(input[1])
  --[[
  table.insert(out, input[2])
  self.nets.languageEncoder = self:_buildLanguageEncoder()
  debugger.enter()
  out = self.nets.languageEncoder:forward(out)
  out[5] = out[5]:float()
  self.nets.selectModel = self:_buildSelectModel(input[3])
  out = self.nets.selectModel:forward(out)
  out[8] = out[8]:float()
  self.nets.fusingModel = self:_buildFusingModel()
  out = self.nets.fusingModel:forward(out)

  out[5] = out[5]:cuda()
  out[8] = out[8]:cuda()
  --]]
  local final_boxes = out[4]
  local pos_roi_boxes = out[2]
  local objectness_scores = out[1]
  local pred_IoUs = out[8]
  --local captions = output[5]
  --local captions = self.nets.language_model:decodeSequence(captions)
  return final_boxes, objectness_scores, pred_IoUs, pos_roi_boxes
end


function DenseCapModel:setGroundTruth(gt_boxes, gt_labels, gt_length)
  self.gt_boxes = gt_boxes
  self.gt_labels = gt_labels
  self.gt_length = gt_length
  self.num_labels = gt_labels:size(2)
  self._called_forward = false
  if self.train then
    self.nets.localization_layer:setGroundTruth(gt_boxes, gt_labels)
  else 
    self.nets.localization_layer:setGroundTruth(nil, nil)
  end
end

function DenseCapModel:backward(input, gradOutput)
  -- Manually backprop through part of the network
  -- self.net has 4 elements:
  -- (1) CNN part 1        (2) CNN part 2
  -- (3) LocalizationLayer (4) Recognition network
  -- We always backprop through (3) and (4), and only backprop through
  -- (2) when finetuning; we never backprop through (1).
  -- Note that this means we break the module API in this method, and don't
  -- actually return gradients with respect to our input.

  local layer_input = self.nets.selectModel.output
  local dout = self.nets.fusingModel:backward(layer_input, gradOutput)


  layer_input = self.nets.languageEncoder.output
  dout = self.nets.selectModel:backward(layer_input, dout)
  dout[5] = dout[5]:cuda()
  dout[8] = dout[8]:cuda()
  dout = self.nets.languageEncoder:backward(self.net:get(4).output, dout)
  table.remove(dout, 8)

  -- Andrew
  -- not backward pass localization layer
  --dout = self.net:get(4):backward(self.net:get(3).output, dout)
  --[[
  local end_idx = 3
  if self.finetune_cnn then end_idx = 2 end
  for i = 4, end_idx, -1 do
    layer_input = self.net:get(i-1).output
    dout = self.net:get(i):backward(layer_input, dout)
  end
  --]]
  self.gradInput = dout
  return self.gradInput
end


--[[
We naughtily override the module's getParameters method, and return:
- params: Flattened parameters for the RPN and recognition network
- grad_params: Flattened gradients for the RPN and recognition network
- cnn_params: Flattened portion of the CNN parameters that will be fine-tuned
- grad_cnn_params: Flattened gradients for the portion of the CNN that will
  be fine-tuned.
--]]
function DenseCapModel:getParameters()
  local cnn_params, grad_cnn_params = self.net:get(2):getParameters()
  local fakenet = nn.Sequential()
  fakenet:add(self.net:get(3))
  fakenet:add(self.net:get(4))
  local params, grad_params = fakenet:getParameters()
  return params, grad_params, cnn_params, grad_cnn_params
end


function DenseCapModel:clearState()
  self.net:clearState()
  for k, v in pairs(self.crits) do
    if v.clearState then
      v:clearState()
    end
  end
end

--[[
Perform a (training-time) forward pass to compute output and loss,
and a backward pass to compute gradients.

This is a nonstandard method, but it allows the DenseCapModel to
have control over its own Criterions.

Input: data is table with the following keys:
- image: 1 x 3 x H x W array of pixel data
- gt_boxes: 1 x B x 4 array of ground-truth object boxes
  TODO: What format are the boxes?
- gt_labels: 1 x B x L array of ground-truth sequences for boxes
--]]
function DenseCapModel:getTarget(num_proposals, ious)

  local num_query = self.gt_labels:size(2)
  local num_proposals = num_proposals
  local max, idx = torch.max(ious[1],1)
  return idx[1]
end

function DenseCapModel:forward_backward(data)
  self:training()

  -- Run the model forward
  self:setGroundTruth(data.gt_boxes, data.gt_labels, data.gt_length)
  local out = self:forward(data.image)
  table.insert(out, data.gt_labels[1])
  self.nets.languageEncoder = self:_buildLanguageEncoder()

  out = self.nets.languageEncoder:forward(out)
  out[5] = out[5]:float()
  self.nets.selectModel = self:_buildSelectModel(data.gt_length)
  out = self.nets.selectModel:forward(out)
  out[8] = out[8]:float()
  local num_proposals = out[2]:size(1)
  self.nets.fusingModel = self:_buildFusingModel(num_proposals)
  out = self.nets.fusingModel:forward(out)
  out[5] = out[5]:cuda()
  out[8] = out[8]:cuda()

  -- Pick out the outputs we care about
  local objectness_scores = out[1]
  local pos_roi_boxes = out[2]
  local final_box_trans = out[3]
  local lm_output = out[5]
  local gt_boxes = out[6]
  local gt_labels = out[7]
  local pred_IoUs = out[8]

  local num_boxes = objectness_scores:size(1)
  local num_pos = pos_roi_boxes:size(1)

  local boxes_view = pos_roi_boxes:view(1, -1, 4)
  local gt_boxes_view = self.gt_boxes:view(1, -1, 4)
  local box_iou = nn.BoxIoU():type(gt_boxes:type())
  local IoUs = box_iou:forward{boxes_view, gt_boxes_view}  -- N x M

  -- Compute final objectness loss and gradient
  local objectness_labels = torch.LongTensor(num_boxes):zero()
  objectness_labels[{{1, num_pos}}]:fill(1)
  local end_objectness_loss = self.crits.objectness_crit:forward(
                                         objectness_scores, objectness_labels)
                                       
  end_objectness_loss = end_objectness_loss * self.opt.end_objectness_weight
  local grad_objectness_scores = self.crits.objectness_crit:backward(
                                      objectness_scores, objectness_labels)
  grad_objectness_scores:mul(self.opt.end_objectness_weight)
  -- Compute box regression loss; this one multiplies by the weight inside
  -- the criterion so we don't do it manually.
  local end_box_reg_loss = self.crits.box_reg_crit:forward(
                                {pos_roi_boxes, final_box_trans},
                                gt_boxes)
  local din = self.crits.box_reg_crit:backward(
                         {pos_roi_boxes, final_box_trans},
                         gt_boxes)
  local grad_pos_roi_boxes, grad_final_box_trans = unpack(din)
  --[[
  -- Compute captioning loss
  local target = self.nets.language_model:getTarget(gt_labels)
  local captioning_loss = self.crits.lm_crit:forward(lm_output, target)
  captioning_loss = captioning_loss * self.opt.captioning_weight
  local grad_lm_output = self.crits.lm_crit:backward(lm_output, target)
  grad_lm_output:mul(self.opt.captioning_weight)
  --]]


  local score_loss
  local grad_score
  if pos_roi_boxes:size(1) ~= 1 then
    local target = self:getTarget(num_proposals, IoUs):cuda()
    score_loss = self.crits.score_crit:forward(pred_IoUs, target)
    grad_score = self.crits.score_crit:backward(pred_IoUs, target)
  else 
    grad_loss = 0
    grad_score = torch.zeros(pred_IoUs:size()):cuda()
    print ('Only one proposal :(')
  end
  --[[
  debugger.enter()
  local IoUs_loss = self.crits.IoUs_crit:forward(pred_IoUs, IoUs)
  IoUs_loss = IoUs_loss * 1
  local grad_IoUs = self.crits.IoUs_crit:backward(pred_IoUs, IoUs)
  grad_IoUs:mul(1)
  --]]

  local ll_losses = self.nets.localization_layer.stats.losses
  local losses = {
    mid_objectness_loss=ll_losses.obj_loss_pos + ll_losses.obj_loss_neg,
    mid_box_reg_loss=ll_losses.box_reg_loss,
    end_objectness_loss=end_objectness_loss,
    end_box_reg_loss=end_box_reg_loss,
    score_loss = score_loss
    --captioning_loss=captioning_loss,
  }
  local sum = 0
  for k, v in pairs(losses) do
    sum = sum + v
  end

  losses.total_loss = sum
  -- Run the model backward
  local grad_out = {}
  grad_out[1] = grad_objectness_scores
  grad_out[2] = grad_pos_roi_boxes
  grad_out[3] = grad_final_box_trans
  grad_out[4] = out[4].new(#out[4]):zero()
  --grad_out[5] = grad_lm_output
  grad_out[5] = lm_output.new(#lm_output):zero():float()
  grad_out[6] = gt_boxes.new(#gt_boxes):zero()
  grad_out[7] = gt_labels.new(#gt_labels):zero()
  grad_out[8] = grad_score:float()

  self:backward(input, grad_out)
  return losses
end
