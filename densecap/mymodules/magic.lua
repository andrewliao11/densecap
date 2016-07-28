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
local debugger = require('fb.debugger')

function Magic:updateOutput(input)


   local a1, a2 = unpack(input)
   self.d1 = a1:size(1)
   self.d2 = a2:size(1)
   -- check if the dimension is the same
   assert(a1:size(2)==a2:size(2))
   assert(a1:size(2)==self.dimension)
   self.dim = a1:size(2)
   local a1_exp = a1:view(self.d1,1,self.dim):expand(self.d1,self.d2,self.dim)
   local a2_exp = a2:view(1,self.d2,self.dim):expand(self.d1,self.d2,self.dim)


   --self.output[1] = a1_exp
   --self.output[2] = a2_exp
   self.output[1] = a1_exp:contiguous():view(self.d1*self.d2, self.dim) 
   self.output[2] = a2_exp:contiguous():view(self.d1*self.d2, self.dim)
   return self.output

end

function Magic:updateGradInput(input, gradOutput)

   gradOutput[1] = gradOutput[1]:view(self.d1,self.d2,self.dim)
   gradOutput[2] = gradOutput[2]:view(self.d1,self.d2,self.dim)
   self.gradInput = {}
   self.gradInput[1] = torch.sum(gradOutput[1],2):squeeze()
   self.gradInput[2] = torch.sum(gradOutput[2],1):squeeze()
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
