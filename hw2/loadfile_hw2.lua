
require 'nn'

local M={}

	do -- data augmentation module
	  local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

	  function BatchFlip:__init()
		parent.__init(self)
		self.train = true
	  end

	  function BatchFlip:updateOutput(input)
		if self.train then
		  local bs = input:size(1)
		  local flip_mask = torch.randperm(bs):le(bs/2)
		  for i=1,input:size(1) do
			if 1 == flip_mask[i] % 3 then image.hflip(input[i]) end
		  end
		end
		self.output:set(input:cuda())
		return self.output
	  end
	end

local  function forwardNet(model,data , labels )

	local confusion = optim.ConfusionMatrix(torch.range(0,9):totable())
	--local model:cuda()
	local criterion = nn.ClassNLLCriterion():cuda()
    local lossAcc = 0
    local numBatches = 0
	local batchSize = 128
	model:evaluate()
	--[[
    if train then
        --set network into training mode
        model:training()
    else
        model:evaluate() -- turn off drop-out
    end
	]]--
    for i = 1, data:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        local x = data:narrow(1, i, batchSize)
        local yt = labels:narrow(1, i, batchSize)
        local y = model:forward(x)
        local err = criterion:forward(y, yt)
        lossAcc = lossAcc + err
        confusion:batchAdd(y,yt)
        --[[
        if train then
            function feval()
                model:zeroGradParameters() --zero grads
                local dE_dy = criterion:backward(y,yt)
                model:backward(x, dE_dy) -- backpropagation
            
                return err, dE_dw
            end
        
            optim.adam(feval, w, optimState)
        end
		]]--
    end
    
    confusion:updateValids()
    local avgLoss = lossAcc / numBatches
    local avgError = 1 - confusion.totalValid
    
    return  avgError
end




function M.loadingModel()

	require 'torch'
	require 'image'
	require 'nn'
	require 'cunn'
	require 'cudnn'
	require 'optim'
	require 'xlua'
	--start
	
	--local redChannel = trainData[{ {}, {1}, {}, {}  }] -- this picks {all images, 1st channel, all vertical pixels, all horizontal pixels}
	--print(#redChannel)
	
	local function hflip(x)
		return torch.random(0,1) == 1 and x or image.hflip(x)
	end
	

	
	local trainset = torch.load('cifar.torch/cifar10-train.t7')
	local testset = torch.load('cifar.torch/cifar10-test.t7')
	
	local trainData = trainset.data:float() -- convert the data from a ByteTensor to a float Tensor.
	local trainLabels = trainset.label:float():add(1)
	local testData = testset.data:float()
	local testLabels = testset.label:float():add(1)

	local mean = {}  -- store the mean, to normalize the test set in the future
	local stdv  = {} -- store the standard-deviation for the future
	for i=1,3 do -- over each image channel
		mean[i] = trainData[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
		--print('Channel ' .. i .. ', Mean: ' .. mean[i])
		trainData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
		
		stdv[i] = trainData[{ {}, {i}, {}, {}  }]:std() -- std estimation
		--print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
		trainData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
	end

-- Normalize test set using same values

	for i=1,3 do -- over each image channel
		testData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
		testData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
	end
	--[[
	local trainData = mnist.traindataset().data:float();
	local trainLabels = mnist.traindataset().label:add(1);
	testData = mnist.testdataset().data:float();
	testLabels = mnist.testdataset().label:add(1);

	local mean = trainData:mean()
	local std = trainData:std()
	trainData:add(-mean):div(std); 
	testData:add(-mean):div(std);
	--finish
	]]--
	
	--[[local trainData = mnist.traindataset();
	local testData = mnist.testdataset();
	 trainData = trainData.data:float(); --turn data to float (originaly byte)
	testData = testData.data:float();
 -- centre data around 0
	testData:add(-127):div(128);
	]]--
	--require 'nn'
	--local testLabels = mnist.testdataset().label:add(1);
	local obj = torch.load('mycnn_adam+hflip_1.dat')
	local testError = forwardNet(obj, testData, testLabels)
	
	return testError
	--print('Test error: ' .. testError)
end

		
	return M	
		
		
		