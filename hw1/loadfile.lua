
local M={}


local function forwardNet(model,data , labels )

    local confusion = optim.ConfusionMatrix(torch.range(0,9):totable())
	local criterion = nn.ClassNLLCriterion():cuda()
    local lossAcc = 0
    local numBatches = 0
	local batchSize = 128
    for i = 1, data:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        local x = data:narrow(1, i, batchSize):cuda()
        local yt = labels:narrow(1, i, batchSize):cuda()
        local y = model:forward(x)
        local err = criterion:forward(y, yt)
        lossAcc = lossAcc + err
        confusion:batchAdd(y,yt)
	end
	confusion:updateValids()
	local avgError = 1 - confusion.totalValid
		
		
	return avgError
		
		
end


function M.loadingModel()
	require 'nn'
	require 'cunn'
	--luarocks install mnist
	--luarocks install image
	
	require 'image'
	local mnist = require 'mnist';
	local optim = require 'optim'
	--start
	local trainData = mnist.traindataset().data:float();
	local trainLabels = mnist.traindataset().label:add(1);
	testData = mnist.testdataset().data:float();
	testLabels = mnist.testdataset().label:add(1);

	local mean = trainData:mean()
	local std = trainData:std()
	trainData:add(-mean):div(std); 
	testData:add(-mean):div(std);
	--finish
	
	
	--[[local trainData = mnist.traindataset();
	local testData = mnist.testdataset();
	 trainData = trainData.data:float(); --turn data to float (originaly byte)
	testData = testData.data:float();
 -- centre data around 0
	testData:add(-127):div(128);
	]]--
	testLabels = mnist.testdataset().label:add(1);
	local obj = torch.load('mynn.dat')
	local testError = forwardNet(obj, testData, testLabels)
	
	return testError
	--print('Test error: ' .. testError)
end

		
	return M	
		
		
		