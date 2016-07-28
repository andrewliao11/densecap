require 'nngraph'
require 'nn'

local mcb = {}

function mcb.twoD_mcb(outer_size,dropout)


    local inputs = {}
    local outputs = {}
    table.insert(inputs, nn.Identity()())
    table.insert(inputs, nn.Identity()())
 
    local ques_feat = inputs[1]         
    local img_feat = inputs[2]

    local fusing_feat = nn.CompactBilinearPooling(outer_size)({ques_feat,img_feat})
    local sqrt_feat = nn.SignedSquareRoot()(fusing_feat)
    local norm_feat = nn.Normalize(2)(sqrt_feat)
    local drop_feat = nn.Dropout(dropout)(norm_feat)

    table.insert(outputs,drop_feat)

    return nn.gModule(inputs, outputs)
end
