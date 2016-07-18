require 'hdf5'
local utils = require 'densecap.utils'
local box_utils = require 'densecap.box_utils'
local debugger = require('fb.debugger')
local DataLoader = torch.class('DataLoader')
local image = require 'image'

function DataLoader:__init(opt)
  self.h5_file = utils.getopt(opt, 'data_h5') -- required h5file with images and other (made with prepro script)
  self.json_file = utils.getopt(opt, 'data_json') -- required json file with vocab etc. (made with prepro script)
  self.debug_max_train_images = utils.getopt(opt, 'debug_max_train_images', -1)
  self.proposal_regions_h5 = utils.getopt(opt, 'proposal_regions_h5', '')
  
  -- load the json file which contains additional information about the dataset
  print('DataLoader loading json file: ', self.json_file)

  --[[
  self.info = utils.read_json(self.json_file)
  self.vocab_size = utils.count_keys(self.info.idx_to_token)

  self.idx_to_token = utils.buildVocab('/home/andrewliao11/Work/Natural-Language-Object-Retrieval-tensorflow/data/vocabulary.txt')

  debugger.enter()

  -- Convert keys in idx_to_token from string to integer
  local idx_to_token = {}
  for k, v in pairs(self.info.idx_to_token) do
    idx_to_token[tonumber(k)] = v
  end
  self.info.idx_to_token = idx_to_token
  --]]

  -- Build Dictionary
  self.dictionary_file = utils.getopt(opt, 'dictionary_path')
  self.info = {}
  self.info.idx_to_token = utils.read_txt(self.dictionary_file)
  self.info.idx_to_token[0] = '<end>'
  self.info.token_to_idx = {}
  for k,v in pairs(self.info.idx_to_token) do
    self.info.token_to_idx[v] = k
  end
  self.vocab_size = #self.info.idx_to_token+1	-- +1: the <end> 

  print('DataLoader loading')
  self.filename = nil
  self.max_words = utils.getopt(opt, 'max_words')
  self.trn_imlist_file = utils.getopt(opt, 'trn_imlist_file')
  self.trn_imlist = utils.read_txt(self.trn_imlist_file)
  self.test_imlist_file = utils.getopt(opt, 'test_imlist_file')
  self.test_imlist = utils.read_txt(self.test_imlist_file)

  self.imsize_file = utils.getopt(opt, 'imsize_file')
  self.imsize = utils.read_json(self.imsize_file)
  self.imcrop_file = utils.getopt(opt, 'imcrop_file')
  self.imcrop = utils.read_json(self.imcrop_file)
  self.imcrop_bbox_file = utils.getopt(opt, 'imcrop_bbox_file')
  self.imcrop_bbox = utils.read_json(self.imcrop_bbox_file)
  self.query_file = utils.getopt(opt, 'query_file')
  self.raw_query = utils.read_json(self.query_file)

  self.query = {}
  for k,v in pairs(self.raw_query) do
    self.query[k] = {}
    for i = 1,#v do
      self.query[k][i] = torch.zeros(self.max_words)
      local index = 1
      for value in string.gmatch(self.raw_query[k][i],"%w+") do 
	if index > self.max_words then break end
	if self.info.token_to_idx[value] ~= nil then
	  self.query[k][i][index] = self.info.token_to_idx[value]
	  index = index + 1
	end
	--[[
        if self.info.token_to_idx[value] == nil then
	  self.query[k][i][index] = 1	-- <unk>
	else 
    	  self.query[k][i][index] = self.info.token_to_idx[value]
	end
    	index = index + 1
	--]]
      end
    end
  end

  --[[
  -- open the hdf5 file
  print('DataLoader loading h5 file: ', self.h5_file)
  self.h5_file = hdf5.open(self.h5_file, 'r')
  local keys = {}
  table.insert(keys, 'box_to_img')
  table.insert(keys, 'boxes')
  table.insert(keys, 'image_heights')
  table.insert(keys, 'image_widths')
  table.insert(keys, 'img_to_first_box')
  table.insert(keys, 'img_to_last_box')
  table.insert(keys, 'labels')
  table.insert(keys, 'lengths')
  table.insert(keys, 'original_heights')
  table.insert(keys, 'original_widths')
  table.insert(keys, 'split')
  for k,v in pairs(keys) do
    print('reading ' .. v)
    self[v] = self.h5_file:read('/' .. v):all()
  end
  
  -- open region proposals file for reading. This is useful if we, e.g.
  -- want to use the ground truth boxes, or if we want to use external region proposals
  if string.len(self.proposal_regions_h5) > 0 then
    print('DataLoader loading objectness boxes from h5 file: ', self.proposal_regions_h5)
    self.obj_boxes_file = hdf5.open(self.proposal_regions_h5, 'r')
    self.obj_img_to_first_box = self.obj_boxes_file:read('/img_to_first_box'):all()
    self.obj_img_to_last_box = self.obj_boxes_file:read('/img_to_last_box'):all()
  end
  --]]
  -- extract image size from dataset
  --local images_size = self.h5_file:read('/images'):dataspaceSize()
  self.num_test_images = #self.test_imlist
  self.num_trn_images = #self.trn_imlist
  self.num_channels = 3
  local max_x = 0
  local max_y = 0
  for k,v in pairs(self.imsize) do 
   if v[1]>max_x then
     max_x = v[1]
   end
   if v[2]>max_y then
     max_y = v[2]
   end
  end
  assert(max_x == max_y, 'width and height must match')
  self.max_image_size = max_x

  -- extract some attributes from the data
  self.num_regions = 0
  for k,v in pairs(self.imcrop) do
    self.num_regions = self.num_regions + #v
  end
  self.vgg_mean = torch.FloatTensor{103.939, 116.779, 123.68} -- BGR order
  self.vgg_mean = self.vgg_mean:view(1,3,1,1)
  self.seq_length = self.max_words

  --[[
  -- set up index ranges for the different splits
  self.train_ix = {}
  self.val_ix = {}
  self.test_ix = {}
  for i=1,self.num_images do
    if self.split[i] == 0 then table.insert(self.train_ix, i) end
    if self.split[i] == 1 then table.insert(self.val_ix, i) end
    if self.split[i] == 2 then table.insert(self.test_ix, i) end
  end
  --]]
  self.iterators = {[0]=1,[1]=1,[2]=1} -- iterators (indices to split lists) for train/val/test
  print(string.format('assigned %d/%d images to train/test.', #self.trn_imlist, #self.test_imlist))
  print('initialized DataLoader:')
  print(string.format('#images: %d, #regions: %d, sequence max length: %d', 
                      self.num_trn_images+self.num_test_images, self.num_regions, self.seq_length))
end

function DataLoader:getImageMaxSize()
  return self.max_image_size
end

function DataLoader:getSeqLength()
  return self.seq_length
end

function DataLoader:getVocabSize()
  return self.vocab_size
end

function DataLoader:getVocab()
  return self.info.idx_to_token
end

--[[
take a LongTensor of size DxN with elements 1..vocab_size+1 
(where last dimension is END token), and decode it into table of raw text sentences
--]]
function DataLoader:decodeSequence(seq)
  local D,N = seq:size(1), seq:size(2)
  local out = {}
  local itow = self.info.idx_to_token
  for i=1,N do
    local txt = ''
    for j=1,D do
      local ix = seq[{j,i}]
      if ix >= 1 and ix <= self.vocab_size then
        -- a word, translate it
        if j >= 2 then txt = txt .. ' ' end -- space
        txt = txt .. itow[tostring(ix)]
      else
        -- END token
        break
      end
    end
    table.insert(out, txt)
  end
  return out
end

-- split is an integer: 0 = train, 1 = val, 2 = test
function DataLoader:resetIterator(split)
  assert(split == 0 or split == 1 or split == 2, 'split must be integer, either 0 (train), 1 (val) or 2 (test)')
  self.iterators[split] = 1
end

--[[
  split is an integer: 0 = train, 1 = val, 2 = test
  Returns a batch of data in two Tensors:
  - X (1,3,H,W) containing the image
  - B (1,R,4) containing the boxes for each of the R regions in xcycwh format
  - y (1,R,L) containing the (up to L) labels for each of the R regions of this image
  - info table of length R, containing additional information as dictionary (e.g. filename)
  The data is iterated linearly in order. Iterators for any split can be reset manually with resetIterator()
  Returning random examples is also supported by passing in .iterate = false in opt.
--]]


function DataLoader:getFilename()
  return self.filename
end

function DataLoader:testIfOutOfBound()

  if self.ri > #self.test_imlist then
    return true
  else
    return false
  end

end

function DataLoader:getBatch(opt)

  --TODO
  local split = utils.getopt(opt, 'split', 0)
  local iterate = utils.getopt(opt, 'iterate', true)
  --[[
  assert(split == 0 or split == 1 or split == 2, 'split must be integer, either 0 (train), 1 (val) or 2 (test)')
  local split_ix
  if split == 0 then split_ix = self.train_ix end
  if split == 1 then split_ix = self.val_ix end
  if split == 2 then split_ix = self.test_ix end
  assert(#split_ix > 0, 'split is empty?')
  --]]
  -- pick an index of the datapoint to load next
  local ri -- ri is iterator position in local coordinate system of split_ix for this split
  local max_index

  if split == 0 then
     max_index = self.num_trn_images
  else 
     max_index = self.num_test_images
  end

  -- pick the training sample
  --if self.debug_max_train_images > 0 then max_index = self.debug_max_train_images end
  if iterate then
    ri = self.iterators[split] -- get next index from iterator
    local ri_next = ri + 1 -- increment iterator
    if ri_next > max_index then ri_next = 1 end -- wrap back around
    self.iterators[split] = ri_next
  else
    -- pick an index randomly
    ri = torch.random(max_index)
  end
  self.ri = ri
  --[[
  ix = split_ix[ri]
  assert(ix ~= nil, 'bug: split ' .. split .. ' was accessed out of bounds with ' .. ri)
  
  -- fetch the image
  local  img = self.h5_file:read('/images'):partial({ix,ix},{1,self.num_channels},
                            {1,self.max_image_size},{1,self.max_image_size})
  --]]

  --TODO load the image
  img_dir = '/home/andrewliao11/Work/Natural-Language-Object-Retrieval-tensorflow/datasets/ReferIt/ImageCLEF/images/'
  local im_name
  if split == 0 then
    im_name = self.trn_imlist[ri]
  else
    im_name = self.test_imlist[ri]
  end
  self.filename = im_name
  img = image.load(img_dir .. im_name .. '.jpg')
  img = img:view(1, img:size(1), img:size(2), img:size(3))*255
  img:add(-1, self.vgg_mean:expandAs(img)) -- subtract vgg mean

  --[[
  -- crop image to its original width/height, get rid of padding, and dummy first dim
  img = img[{ 1, {}, {1,self.image_heights[ix]}, {1,self.image_widths[ix]} }]
  img = img:float() -- convert to float
  img = img:view(1, img:size(1), img:size(2), img:size(3)) -- batch the image
  img:add(-1, self.vgg_mean:expandAs(img)) -- subtract vgg mean
  --]]

  -- fetch the corresponding labels array
  local count = 0
  for k,v in pairs(self.imcrop[im_name]) do
    count = count + #self.query[v]
  end

  local box_batch = torch.zeros(1,count,4)
  local label_array = torch.zeros(1,count, self.max_words)
  local i = 1
  for k,v in pairs(self.imcrop[im_name]) do
    for j = 1,#self.query[v] do
      label_array[1][i] = self.query[v][j]
      box_batch[1][i][1] = (self.imcrop_bbox[v][1]+self.imcrop_bbox[v][3])/2
      box_batch[1][i][2] = (self.imcrop_bbox[v][2]+self.imcrop_bbox[v][4])/2
      box_batch[1][i][3] = (self.imcrop_bbox[v][3]-self.imcrop_bbox[v][1])
      box_batch[1][i][4] = (self.imcrop_bbox[v][4]+self.imcrop_bbox[v][2])
      i = i+1
    end
  end
  box_batch = box_batch:int()
  label_array = label_array:int()
  --[[ 
  local r0 = self.img_to_first_box[ix]
  local r1 = self.img_to_last_box[ix]
  local label_array = self.labels[{ {r0,r1} }]
  local box_batch = self.boxes[{ {r0,r1} }]

  -- batch the boxes and labels
  assert(label_array:nDimension() == 2)
  assert(box_batch:nDimension() == 2)
  label_array = label_array:view(1, label_array:size(1), label_array:size(2))
  box_batch = box_batch:view(1, box_batch:size(1), box_batch:size(2))
  -- finally pull the info from json file
  local filename = self.info.idx_to_filename[tostring(ix)] -- json is loaded with string keys
  assert(filename ~= nil, 'lookup for index ' .. ix .. ' failed in the json info table.')
  local w,h = self.image_widths[ix], self.image_heights[ix]
  local ow,oh = self.original_widths[ix], self.original_heights[ix]
  local info_table = { {filename = filename, 
                        split_bounds = {ri, #split_ix},
                        width = w, height = h, ori_width = ow, ori_height = oh} }

  -- read regions if applicable
  local obj_boxes -- contains batch of x,y,w,h,score objectness boxes for this image
  if self.obj_boxes_file then
    local r0 = self.obj_img_to_first_box[ix]
    local r1 = self.obj_img_to_last_box[ix]
    obj_boxes = self.obj_boxes_file:read('/boxes'):partial({r0,r1},{1,5})
    -- scale boxes (encoded as xywh) into coord system of the resized image
    local frac = w/ow -- e.g. if ori image is 800 and we want 512, then we need to scale by 512/800
    local boxes_scaled = box_utils.scale_boxes_xywh(obj_boxes[{ {}, {1,4} }], frac)
    local boxes_trans = box_utils.xywh_to_xcycwh(boxes_scaled)
    obj_boxes[{ {}, {1,4} }] = boxes_trans
    obj_boxes = obj_boxes:view(1, obj_boxes:size(1), obj_boxes:size(2)) -- make batch
  end

  -- TODO: start a prefetching thread to load the next image ?
  return img, box_batch, label_array, info_table, obj_boxes
  --]]
  return count, img, box_batch, label_array
end

