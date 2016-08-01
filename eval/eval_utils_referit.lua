local image = require 'image'
local cjson = require 'cjson'
local utils = require 'densecap.utils'
local box_utils = require 'densecap.box_utils'
local eval_utils = {}
local debugger = require('fb.debugger')
require 'densecap.modules.BoxIoU'

--[[
Evaluate a DenseCapModel on a split of data from a DataLoader.

Input: An object with the following keys:
- model: A DenseCapModel object to evaluate; required.
- loader: A DataLoader object; required.
- split: Either 'val' or 'test'; default is 'val'
- max_images: Integer giving the number of images to use, or -1 to use the
  entire split. Default is -1.
- id: ID for cross-validation; default is ''.
- dtype: torch datatype to which data should be cast before passing to the
  model. Default is 'torch.FloatTensor'.
--]]
function eval_utils.eval_split(kwargs, opt)
  local model = utils.getopt(kwargs, 'model')
  local loader = utils.getopt(kwargs, 'loader')
  local split = utils.getopt(kwargs, 'split', 'val')
  local max_images = utils.getopt(kwargs, 'max_images', -1)
  local id = utils.getopt(kwargs, 'id', '')
  local dtype = utils.getopt(kwargs, 'dtype', 'torch.FloatTensor')
  assert(split == 'val' or split == 'test', 'split must be "val" or "test"')
  local split_to_int = {val=1, test=2}
  split = split_to_int[split]
  print('using split ', split)
  
  model:evaluate()
  loader:resetIterator(split)
  local evaluator = DenseCaptioningEvaluator{id=id}

  local ious = nn.BoxIoU()
  local total_query = 0
  local hit = 0
  local oracle = 0
  local counter = 0
  local all_losses = {}
  local result_boxes = {}
  while true do
    counter = counter + 1
    
    -- Grab a batch of data and convert it to the right dtype
    local data = {}
    local loader_kwargs = {split=split, iterate=true}
    while true do
      num_query, img, gt_boxes, gt_labels, info, raw_query = loader:getBatch(loader_kwargs)
      if num_query ~= 0 then
        break
      end
    end

    local data = {
      image = img:type(dtype),
      gt_boxes = gt_boxes:type(dtype),
      gt_labels = gt_labels:type(dtype),
    }
    --info = info[1] -- Since we are only using a single image

    -- Call forward_backward to compute losses
    model.timing = false
    model.dump_vars = false
    model.cnn_backward = false

    data.gt_length = torch.eq(torch.eq(data.gt_labels,0),0):sum(3)+1
    data.gt_length = data.gt_length:view(-1)

    local max_words = utils.getopt(opt, 'max_words')+1
    data.mask = torch.zeros(data.gt_labels:size(2), max_words, utils.getopt(opt, 'rnn_size'))
    for i = 1, data.gt_labels:size(2) do
      data.mask[i][data.gt_length[i]]:fill(1)
    end


    local valid, losses = model:forward_backward(data)
    table.insert(all_losses, losses)

    -- Call forward_test to make predictions, and pass them to evaluator

    local boxes, logprobs, pred_IoUs, pos_roi_boxes, lm_out, roi_feat, diff = model:forward_test(data)
    result_boxes[info[1]] = boxes:float()
    evaluator:addResult(logprobs, boxes, nil, gt_boxes[1], nil)

    y, i = torch.max(pred_IoUs:float(), 2)
    pos_roi_boxes = pos_roi_boxes:float()
    pred_boxes = pos_roi_boxes:index(1,i:view(-1)):view(1, -1, 4)
    img_dir = '/home/andrewliao11/Work/Natural-Language-Object-Retrieval-tensorflow/datasets/ReferIt/ImageCLEF/images/'
    -- draw boxes onto image

    if utils.getopt(kwargs, 'get_box', false) then
      for i = 1, pred_boxes:size(2) do
	local new_image = image.load(img_dir .. info[1] .. '.jpg')
	local img_size = new_image:size()
        local new_box = box_utils.xcycwh_to_x1y1x2y2(pred_boxes[1][i]:view(1,1,4)):int():view(-1)
        if new_box[1]-2 < 0 then new_box[1] = 2 end	-- x1
        if new_box[2]-2 < 0 then new_box[2] = 2 end	-- y1
        if new_box[3]+2 > img_size[3]-1 then new_box[3] = img_size[3]-2 end	-- x2
        if new_box[4]+2 > img_size[2]-1 then new_box[4] = img_size[2]-2 end	-- y2

	if math.abs(new_box[3]-new_box[1])>4 and math.abs(new_box[4]-new_box[2])>4 then 
          new_image = image.drawRect(new_image, new_box[1], new_box[2], new_box[3], new_box[4], {lineWidth = 2, color = {255, 0, 0}})
          new_image = image.drawText(new_image, raw_query[i], new_box[1], new_box[2],{color = {255, 0, 0}, size = 2})
          image.save('result/' .. info[1] .. '-' .. tostring(i) .. '.jpg',new_image)
	end
      end
    end


    local boxes_view = pos_roi_boxes:view(1, -1, 4):cuda()
    local gt_boxes_view = data.gt_boxes:view(1, -1, 4):cuda()
    local box_iou = nn.BoxIoU():type(gt_boxes_view:type())
    local IoUs = box_iou:forward{gt_boxes_view, boxes_view}  -- N x M
    local cur_oracle = torch.sum(torch.max(IoUs,3)[1]:ge(0.5))
    oracle = oracle + cur_oracle
    
    local cur_hit  = 0
    for i = 1,pred_boxes:size(2) do
      iou = ious:forward{pred_boxes[1][i]:view(1, -1, 4), gt_boxes[1][i]:float():view(1, -1, 4)}
      iou = iou:view(-1)
      if iou[1] > 0.5 then
        cur_hit = cur_hit+1
      end
    end
    hit = hit + cur_hit 
    print(info[1] .. 'get' .. tostring(cur_hit) .. 'hits!')    

    total_query = total_query + pred_boxes:size(2)

    -- Print a message to the console
    local msg = 'Processed image %s (%d / %d) of split %d, detected %d regions'
    --local num_images = info.split_bounds[2]
    --if max_images > 0 then num_images = math.min(num_images, max_images) end
    local num_boxes = boxes:size(1)
    print(string.format(msg, loader:getFilename(), counter, max_images, split, num_boxes))

    --[[
    if counter % 100 == 0 then 
      local loss_results = utils.dict_average(all_losses)
      print('Loss stats:')
      print(loss_results)
      print('Average loss: ', loss_results.total_loss)
      local precision = hit/total_query
      print('Precision: ', hit/total_query)
      print('Languange out:')
      print(lm_out)
      print('roi feature[1]:')
      print(roi_feat[1])
      local ap_results = evaluator:evaluate()
      print(string.format('mAP: %f', 100 * ap_results.map))

      local out = {
        loss_results=loss_results,
        ap_results=ap_results,
      }
    end
    --]]
    -- Break out if we have processed enough images
    if max_images > 0 and counter >= max_images then break end
    --if info.split_bounds[1] == info.split_bounds[2] then break end
    if loader:testIfOutOfBound() then break end
  end
  local loss_results = utils.dict_average(all_losses)
  print('Loss stats:')
  print(loss_results)
  print('Average loss: ', loss_results.total_loss)
  local precision = hit/total_query
  print('Precision: ', hit/total_query)
  print('Oracle: ', oracle/total_query)
  local ap_results = evaluator:evaluate()
  print(string.format('mAP: %f', 100 * ap_results.map))
  
  local out = {
    loss_results=loss_results,
    ap_results=ap_results,
  }
  
  return out, result_boxes, precision
end


function eval_utils.score_captions(records)
  -- serialize records to json file
  utils.write_json('eval/input.json', records)
  -- invoke python process 
  os.execute('python eval/meteor_bridge.py')
  -- read out results
  local blob = utils.read_json('eval/output.json')
  return blob
end


local function pluck_boxes(ix, boxes, text)
  -- ix is a list (length N) of LongTensors giving indices to boxes/text. Use them to do merge
  -- this is done because multiple ground truth annotations can be on top of each other, and
  -- we want to instead group many overlapping boxes into one, with multiple caption references.
  -- return boxes Nx4, and text[] of length N

  local N = #ix
  local new_boxes = torch.zeros(N, 4)
  local new_text = {}

  for i=1,N do
    
    local ixi = ix[i]
    local n = ixi:nElement()
    local bsub = boxes:index(1, ixi)
    local newbox = torch.mean(bsub, 1)
    new_boxes[i] = newbox
    
    --local texts = {}
   -- if text then
    --  for j=1,n do
    --    table.insert(texts, text[ixi[j]])
    --  end
    --end
    --table.insert(new_text, texts)
  end
  return new_boxes
end


local DenseCaptioningEvaluator = torch.class('DenseCaptioningEvaluator')
function DenseCaptioningEvaluator:__init(opt)
  self.all_logprobs = {}
  self.records = {}
  self.n = 1
  self.npos = 0
  self.id = utils.getopt(opt, 'id', '')
end

-- boxes is (B x 4) are xcycwh, logprobs are (B x 2), target_boxes are (M x 4) also as xcycwh.
-- these can be both on CPU or on GPU (they will be shipped to CPU if not already so)
-- predict_text is length B list of strings, target_text is length M list of strings.
function DenseCaptioningEvaluator:addResult(logprobs, boxes, text, target_boxes, target_text)
  assert(logprobs:size(1) == boxes:size(1))
  --assert(logprobs:size(1) == #text)
  --assert(target_boxes:size(1) == #target_text)
  assert(boxes:nDimension() == 2)

  -- convert both boxes to x1y1x2y2 coordinate systems
  boxes = box_utils.xcycwh_to_x1y1x2y2(boxes)
  target_boxes = box_utils.xcycwh_to_x1y1x2y2(target_boxes)

  -- make sure we're on CPU
  boxes = boxes:float()
  logprobs = logprobs[{ {}, 1 }]:double() -- grab the positives class (1)
  target_boxes = target_boxes:float()

  -- merge ground truth boxes that overlap by >= 0.7
  local mergeix = box_utils.merge_boxes(target_boxes, 0.7) -- merge groups of boxes together
  local merged_boxes= pluck_boxes(mergeix, target_boxes, nil)

  -- 1. Sort detections by decreasing confidence
  local Y,IX = torch.sort(logprobs,1,true) -- true makes order descending
  
  local nd = logprobs:size(1) -- number of detections
  local nt = merged_boxes:size(1) -- number of gt boxes
  local used = torch.zeros(nt)
  for d=1,nd do -- for each detection in descending order of confidence
    local ii = IX[d]
    local bb = boxes[ii]
    
    -- assign the box to its best match in true boxes
    local ovmax = 0
    local jmax = -1
    for j=1,nt do
      local bbgt = merged_boxes[j]
      local bi = {math.max(bb[1],bbgt[1]), math.max(bb[2],bbgt[2]),
                  math.min(bb[3],bbgt[3]), math.min(bb[4],bbgt[4])}
      local iw = bi[3]-bi[1]+1
      local ih = bi[4]-bi[2]+1
      if iw>0 and ih>0 then
        -- compute overlap as area of intersection / area of union
        local ua = (bb[3]-bb[1]+1)*(bb[4]-bb[2]+1)+
                   (bbgt[3]-bbgt[1]+1)*(bbgt[4]-bbgt[2]+1)-iw*ih
        local ov = iw*ih/ua
        if ov > ovmax then
          ovmax = ov
          jmax = j
        end
      end
    end

    local ok = 1
    if used[jmax] == 0 then
      used[jmax] = 1 -- mark as taken
    else
      ok = 0
    end

    -- record the best box, the overlap, and the fact that we need to score the language match
    local record = {}
    record.ok = ok -- whether this prediction can be counted toward a true positive
    record.ov = ovmax
    --record.candidate = text[ii]
    --record.references = merged_text[jmax] -- will be nil if jmax stays -1
    -- Replace nil with empty table to prevent crash in meteor bridge
    if record.references == nil then record.references = {} end
    record.imgid = self.n
    table.insert(self.records, record)
  end
  
  -- keep track of results
  self.n = self.n + 1
  self.npos = self.npos + nt
  table.insert(self.all_logprobs, Y:double()) -- inserting the sorted logprobs as double
end

function DenseCaptioningEvaluator:evaluate(verbose)
  if verbose == nil then verbose = true end
  local min_overlaps = {0.3, 0.4, 0.5, 0.6, 0.7}
  local min_scores = {-1, 0, 0.05, 0.1, 0.15, 0.2, 0.25}

  -- concatenate everything across all images
  local logprobs = torch.cat(self.all_logprobs, 1) -- concat all logprobs
  -- call python to evaluate all records and get their BLEU/METEOR scores
  -- local blob = eval_utils.score_captions(self.records, self.id) -- replace in place (prev struct will be collected)
  --local scores = blob.scores -- scores is a list of scores, parallel to records
  collectgarbage()
  collectgarbage()

  -- prints/debugging
   --[[
  if verbose then
    for k=1,#self.records do
      local record = self.records[k]
      if record.ov > 0 and record.ok == 1 and k % 1000 == 0 then
        local txtgt = ''
        assert(type(record.references) == "table")
        for kk,vv in pairs(record.references) do txtgt = txtgt .. vv .. '. ' end
	-- Andrew
        print(string.format('IMG %d PRED: %s, GT: %s, OK: %d, OV: %f SCORE: %f',
              record.imgid, nil, txtgt, record.ok, record.ov, scores[k]))
      end  
    end
  end
--]]
  -- lets now do the evaluation
  local y,ix = torch.sort(logprobs,1,true) -- true makes order descending

  local ap_results = {}
  local det_results = {}
  for foo, min_overlap in pairs(min_overlaps) do
    for foo2, min_score in pairs(min_scores) do

      -- go down the list and build tp,fp arrays
      local n = y:nElement()
      local tp = torch.zeros(n)
      local fp = torch.zeros(n)
      for i=1,n do
        -- pull up the relevant record
        local ii = ix[i]
        local r = self.records[ii]

        if not r.references then 
          fp[i] = 1 -- nothing aligned to this predicted box in the ground truth
        else
          -- ok something aligned. Lets check if it aligned enough, and correctly enough
          --local score = scores[ii]
          if r.ov >= min_overlap and r.ok == 1 then
            tp[i] = 1
          else
            fp[i] = 1
          end
        end
      end

      fp = torch.cumsum(fp,1)
      tp = torch.cumsum(tp,1)
      local rec = torch.div(tp, self.npos)
      local prec = torch.cdiv(tp, fp + tp)

      -- compute max-interpolated average precision
      local ap = 0
      local apn = 0
      for t=0,1,0.01 do
        local mask = torch.ge(rec, t):double()
        local prec_masked = torch.cmul(prec:double(), mask)
        local p = torch.max(prec_masked)
        ap = ap + p
        apn = apn + 1
      end
      ap = ap / apn

      -- store it
      if min_score == -1 then
        det_results['ov' .. min_overlap] = ap
      else
        ap_results['ov' .. min_overlap .. '_score' .. min_score] = ap
      end
    end
  end

  local map = utils.average_values(ap_results)
  local detmap = utils.average_values(det_results)

  -- lets get out of here
  local results = {map = map, ap_breakdown = ap_results, detmap = detmap, det_breakdown = det_results}
  return results
end

function DenseCaptioningEvaluator:numAdded()
  return self.n - 1
end

return eval_utils
