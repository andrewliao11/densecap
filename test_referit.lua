--[[
Main entry point for training a DenseCap model
]]--

-------------------------------------------------------------------------------
-- Includes
-------------------------------------------------------------------------------
require 'torch'
require 'nngraph'
require 'optim'
require 'image'
require 'lfs'
require 'nn'
local cjson = require 'cjson'

require 'densecap.DataLoader_referit'
require 'densecap.DenseCapModel'
require 'densecap.optim_updates'
local utils = require 'densecap.utils'
local opts = require 'test_referit_opts'
local models = require 'models'
local eval_utils = require 'eval.eval_utils_referit'
local debugger = require('fb.debugger')
-------------------------------------------------------------------------------
-- Initializations
-------------------------------------------------------------------------------
local opt = opts.parse(arg)
print(opt)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
if opt.gpu >= 0 then
  -- cuda related includes and settings
  require 'cutorch'
  require 'cunn'
  require 'cudnn'
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpu + 1) -- note +1 because lua is 1-indexed
end

-- initialize the data loader class
local loader = DataLoader(opt)
opt.seq_length = loader:getSeqLength()
opt.vocab_size = loader:getVocabSize()
opt.idx_to_token = loader.info.idx_to_token

-- initialize the DenseCap model object
-- pretrained model
--opt.checkpoint_start_from = 'checkpoint.t7'

local checkpoint = torch.load('./model/delete_pos/checkpoint.t7-10000')

local model = checkpoint.model
print 'Loaded model'

local dtype, use_cudnn = utils.setup_gpus(opt.gpu, opt.use_cudnn)
print(string.format('Using dtype "%s"', dtype))

model:convert(dtype, use_cudnn)
model:setTestArgs{
  rpn_nms_thresh=opt.rpn_nms_thresh,
  final_nms_thresh=opt.final_nms_thresh,
  max_proposals=opt.num_proposals,
}

-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
local loss0
local optim_state = {}
local cnn_optim_state = {}
local best_val_score = -1


-- Set test-time options for the model
    model.nets.localization_layer:setTestArgs{
      nms_thresh=opt.test_rpn_nms_thresh,
      max_proposals=opt.test_num_proposals,
    }
    model.opt.final_nms_thresh = opt.test_final_nms_thresh

    -- Evaluate validation performance
    local eval_kwargs = {
      model=model,
      loader=loader,
      split='val',
      max_images=opt.val_images_use,
      dtype=dtype,
      get_box=false
    }
    local results, result_boxes = eval_utils.eval_split(eval_kwargs, opt)
    -- local results = eval_split(1, opt.val_images_use) -- 1 = validation

    local checkpoint = {}
    checkpoint.opt = opt
    checkpoint.result_boxes = results_boxes
    cjson.encode_number_precision(4) -- number of sig digits to use in encoding
    cjson.encode_sparse_array(true, 2, 10)
    local text = cjson.encode(checkpoint)
    local file = io.open('test_boxes.json', 'w')
    file:write(text)
    file:close()
    print('wrote ' .. 'test_boxes' .. '.json')


--[[
    -- serialize a json file that has all info except the model
    local checkpoint = {}
    checkpoint.opt = opt
    checkpoint.iter = iter
    checkpoint.loss_history = loss_history
    checkpoint.results_history = results_history
    cjson.encode_number_precision(4) -- number of sig digits to use in encoding
    cjson.encode_sparse_array(true, 2, 10)
    local text = cjson.encode(checkpoint)
    local file = io.open(opt.checkpoint_path .. '.json', 'w')
    file:write(text)
    file:close()
    print('wrote ' .. opt.checkpoint_path .. '.json')
--]]
