require 'nn'
require 'magic'

local debugger = require('fb.debugger')

local x1 = torch.rand(10,4)
local x2 = torch.rand(4,4)

local magic = nn.Magic(4)

local out = magic:forward({x1,x2})
debugger.enter()

magic:backward({x1,x2},out)

debugger.enter()
