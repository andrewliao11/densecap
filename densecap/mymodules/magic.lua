local Magic, parent = torch.class('nn.Magic', 'nn.Module')
local debugger = require('fb.debugger')

function Magic:__init(dim)

   parent.__init(self)
   self.dimension = dim
   self.output = {}

end

--[[
expect input[1] is d1,dim
       input[2] is d2,dim
output is d1,d2,dim
]]--
function Magic:updateOutput(input)


   local a1, a2 = unpack(input)
   local d1 = a1:size(1)
   local d2 = a2:size(1)
   -- check if the dimension is the same
   assert(a1:size(2)==a2:size(2))
   assert(a1:size(2)==self.dimension)
   local dim = a1:size(2)
   local a1_exp = a1:view(d1,1,dim):expand(d1,d2,dim)
   local a2_exp = a2:view(1,d2,dim):expand(d1,d2,dim)

   self.output[1] = a1_exp
   self.output[2] = a2_exp
   return self.output

end

function Magic:updateGradInput(input, gradOutput)


   local a1_size = input[1]:size()
   local a2_size = input[2]:size()
   self.gradInput = {}
   self.gradInput[1] = gradOutput[1]:contiguous():sub(1,a1_size[1],1,1,1,a1_size[2]):squeeze()
   self.gradInput[2] = gradOutput[2]:contiguous():sub(1,1,1,a2_size[1],1,a2_size[2]):squeeze()
   return self.gradInput

end

function Magic:clearState()
   nn.utils.clear(self, '_output', 'gradInput')
   return parent.clearState(self)
end

function Magic:training()

   self.cmul.train = true
   self.train = true

end

function Magic:evaluate()

   self.cmul.train = false
   self.train = false

end
