--[[
Due to interest of time, please prepared the data before-hand into a 4D torch
ByteTensor of size 50000x3x32x32 (training) and 10000x3x32x32 (testing) 

mkdir t5
cd t5/
git clone https://github.com/soumith/cifar.torch.git
cd cifar.torch/
th Cifar10BinToTensor.lua

]]--

require 'torch'
require 'image'
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'
require 'xlua'

local function hflip(x)
   return torch.random(0,1) == 1 and x or image.hflip(x)
end

local function randomcrop(im , pad, randomcrop_type)
   if randomcrop_type == 'reflection' then
      -- Each feature map of a given input is padded with the replication of the input boundary
      module = nn.SpatialReflectionPadding(pad,pad,pad,pad):float() 
   elseif randomcrop_type == 'zero' then
      -- Each feature map of a given input is padded with specified number of zeros.
	  -- If padding values are negative, then input is cropped.
      module = nn.SpatialZeroPadding(pad,pad,pad,pad):float()
   end
	
   local padded = module:forward(im:float())
   local x = torch.random(1,pad*2 + 1)
   local y = torch.random(1,pad*2 + 1)
   image.save('img2ZeroPadded.jpg', padded)

   return padded:narrow(3,x,im:size(3)):narrow(2,y,im:size(2))
end


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
        --if 1 == flip_mask[i] % 3 then image.hflip(input[i]) end

      end
    end
    self.output:set(input:cuda())
    return self.output
  end
end



function saveTensorAsGrid(tensor,fileName) 
	local padding = 1
	local grid = image.toDisplayTensor(tensor/255.0, padding)
	image.save(fileName,grid)
end

local trainset = torch.load('cifar.torch/cifar10-train.t7')
local testset = torch.load('cifar.torch/cifar10-test.t7')

local classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

local trainData = trainset.data:float() -- convert the data from a ByteTensor to a float Tensor.
local trainLabels = trainset.label:float():add(1)
local testData = testset.data:float()
local testLabels = testset.label:float():add(1)

--print(trainData:size())

--saveTensorAsGrid(trainData:narrow(1,100,36),'train_100-136.jpg') -- display the 100-136 images in dataset
--print(classes[trainLabels[100]]) -- display the 100-th image class


--  *****************************************************************
--  Let's take a look at a simple convolutional layer:
--  *****************************************************************

--[[
local img = trainData[100]:cuda()
print(img:size())

local conv = cudnn.SpatialConvolution(3, 16, 5, 5, 4, 4, 2, 2)
conv:cuda()
-- 3 input maps, 16 output maps
-- 5x5 kernels, stride 4x4, padding 0x0

print(conv)
print '12'
local output = conv:forward(img)
print(output:size())
saveTensorAsGrid(output, 'convOut.jpg')

local weights = conv.weight
saveTensorAsGrid(weights, 'convWeights.jpg')
print(weights:size())
]]--
--  ****************************************************************
--  Full Example - Training a ConvNet on Cifar10
--  ****************************************************************

-- Load and normalize data:

local redChannel = trainData[{ {}, {1}, {}, {}  }] -- this picks {all images, 1st channel, all vertical pixels, all horizontal pixels}
print(#redChannel)

local mean = {}  -- store the mean, to normalize the test set in the future
local stdv  = {} -- store the standard-deviation for the future
for i=1,3 do -- over each image channel
    mean[i] = trainData[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    
    stdv[i] = trainData[{ {}, {i}, {}, {}  }]:std() -- std estimation
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

-- Normalize test set using same values

for i=1,3 do -- over each image channel
    testData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
    testData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end



--  ****************************************************************
--  Define our neural network
--  ****************************************************************


local model = nn.Sequential()

local function Block(...)
  local arg = {...}
  model:add(nn.SpatialConvolution(...))
  model:add(nn.SpatialBatchNormalization(arg[2],1e-3))
  model:add(nn.ReLU(true))
  return model
end

model:add(nn.BatchFlip():float())
Block(3,64,5,5,1,1,2,2)
Block(64,32,1,1)
Block(32,26,1,1)
model:add(nn.SpatialMaxPooling(3,3,2,2):ceil())
model:add(nn.Dropout(0.3))
Block(26,48,5,5,1,1,2,2)
--Block(48,48,1,1)
Block(48,32,1,1)
model:add(nn.SpatialAveragePooling(3,3,2,2):ceil())
model:add(nn.Dropout(0.3))
Block(32,24,3,3,1,1,1,1)
--Block(64,24,1,1)
Block(24,10,1,1)
model:add(nn.SpatialAveragePooling(8,8,1,1):ceil())
model:add(nn.View(10))
model:add(nn.LogSoftMax()) 
--[[
model = nn.Sequential()
model:add(nn.BatchFlip())
Block(3,64,3,3,1,1,2,2)
Block(64,32,1,1)
Block(32,36,1,1)

model:add(nn.SpatialMaxPooling(3,3,2,2):ceil())
model:add(nn.Dropout(0.3))
Block(36,32,5,5,1,1,2,2)
Block(32,32,1,1)
Block(32,48,1,1)
model:add(nn.SpatialAveragePooling(3,3,2,2):ceil())
model:add(nn.Dropout(0.3))
--Block(48,10,3,3,1,1,1,1)
Block(48,48,1,1)
Block(48,64,1,1)
Block(64,10,1,1)
model:add(nn.SpatialAveragePooling(8,8,1,1):ceil())
model:add(nn.View(10))
model:add(nn.LogSoftMax()) 
]]--
--[[
model:add(nn.BatchFlip():float())
Block(3,64,5,5,1,1,2,2)
Block(64,32,1,1)
Block(32,32,1,1)
model:add(nn.SpatialMaxPooling(3,3,2,2):ceil())
model:add(nn.Dropout(0.3))
Block(32,48,5,5,1,1,2,2)
--Block(64,64,1,1)
Block(48,32,1,1)
model:add(nn.SpatialAveragePooling(3,3,2,2):ceil())
model:add(nn.Dropout(0.3))
Block(32,24,3,3,1,1,1,1)
--Block(64,64,1,1)
Block(24,10,1,1)
model:add(nn.SpatialAveragePooling(8,8,1,1):ceil())
model:add(nn.View(10))
model:add(nn.LogSoftMax())
]]--
--[[
local model = nn.Sequential()
model:add(nn.BatchFlip())
model:add(nn.SpatialConvolution(3, 16, 3, 3)) -- 3 input image channel, 32 output channels, 5x5 convolution kernel
model:add(nn.SpatialMaxPooling(2,2,2,2))      -- A max-pooling operation that looks at 2x2 windows and finds the max.
model:add(nn.ReLU(true))                          -- ReLU activation function
model:add(nn.SpatialBatchNormalization(16))    --Batch normalization will provide quicker convergence
model:add(nn.SpatialConvolution(16, 34, 3, 3))
model:add(nn.SpatialMaxPooling(2,2,2,2))
model:add(nn.ReLU(true))
model:add(nn.SpatialBatchNormalization(34))
model:add(nn.SpatialConvolution(34, 32, 3, 3))
model:add(nn.View(32*4*4):setNumInputDims(3))  -- reshapes from a 3D tensor of 32x4x4 into 1D tensor of 32*4*4
model:add(nn.Linear(32*4*4,64 ))             -- fully connected layer (matrix multiplication between input and weights)
model:add(nn.ReLU(true))
model:add(nn.Dropout(0.5))                      --Dropout layer with p=0.5
model:add(nn.Linear(64, #classes))            -- 10 is the number of outputs of the network (in this case, 10 digits)
model:add(nn.LogSoftMax())                     -- converts the output to a log-probability. Useful for classificati
]]--
model:cuda()
criterion = nn.ClassNLLCriterion():cuda()


w, dE_dw = model:getParameters()
print('Number of parameters:', w:nElement())
print(model)

function shuffle(data,ydata) --shuffle data function
    local RandOrder = torch.randperm(data:size(1)):long()
    return data:index(1,RandOrder), ydata:index(1,RandOrder)
end

--  ****************************************************************
--  Training the network
--  ****************************************************************
require 'optim'

local batchSize = 128
local optimState = {}

function forwardNet(data,labels, train)
    --another helpful function of optim is ConfusionMatrix
    local confusion = optim.ConfusionMatrix(classes)
    local lossAcc = 0
    local numBatches = 0
    if train then
        --set network into training mode
        model:training()
    else
        model:evaluate() -- turn off drop-out
    end
    for i = 1, data:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        local x = data:narrow(1, i, batchSize)
        local yt = labels:narrow(1, i, batchSize)
        local y = model:forward(x)
        local err = criterion:forward(y, yt)
        lossAcc = lossAcc + err
        confusion:batchAdd(y,yt)
        
        if train then
            function feval()
                model:zeroGradParameters() --zero grads
                local dE_dy = criterion:backward(y,yt)
                model:backward(x, dE_dy) -- backpropagation
            
                return err, dE_dw
            end
        
            optim.adam(feval, w, optimState)
        end
    end
    
    confusion:updateValids()
    local avgLoss = lossAcc / numBatches
    local avgError = 1 - confusion.totalValid
    
    return avgLoss, avgError, tostring(confusion)
end

function plotError(trainError, testError, title)
	require 'gnuplot'
	local range = torch.range(1, trainError:size(1))
	gnuplot.pngfigure('testVsTrainError.png')
	gnuplot.plot({'trainError',trainError},{'testError',testError})
	gnuplot.xlabel('epochs')
	gnuplot.ylabel('Error')
	gnuplot.plotflush()
end

---------------------------------------------------------------------

epochs = 1
trainLoss = torch.Tensor(epochs)
testLoss = torch.Tensor(epochs)
trainError = torch.Tensor(epochs)
testError = torch.Tensor(epochs)

--reset net weights
model:apply(function(l) l:reset() end)

timer = torch.Timer()
for e = 1, epochs do

    trainData, trainLabels = shuffle(trainData, trainLabels) --shuffle training data

    trainLoss[e], trainError[e] = forwardNet(trainData, trainLabels, true)
    testLoss[e], testError[e], confusion = forwardNet(testData, testLabels, false)
   
	
    if e % 5 == 0 then
        print('Epoch ' .. e .. ':')
        print('Training error: ' .. trainError[e], 'Training Loss: ' .. trainLoss[e])
        print('Test error: ' .. testError[e], 'Test Loss: ' .. testLoss[e])
        print(confusion)
    end
end

plotError(trainError, testError, 'Classification Error')


--  ****************************************************************
--  Network predictions
--  ****************************************************************


--model:evaluate()   --turn off dropout

--print(classes[testLabels[10]])
--print(testData[10]:size())
--saveTensorAsGrid(testData[10],'testImg10.jpg')
--local predicted = model:forward(testData[10]:view(1,3,32,32):cuda())
--print(predicted:exp()) -- the output of the network is Log-Probabilities. To convert them to probabilities, you have to take e^x 

-- assigned a probability to each classes
--for i=1,predicted:size(2) do
  --  print(classes[i],predicted[1][i])
--end


require 'gnuplot'
local range = torch.range(1, epochs)
gnuplot.pngfigure('test.png')
gnuplot.plot({'trainLoss',trainLoss},{'testLoss',testLoss})
gnuplot.xlabel('epochs')
gnuplot.ylabel('Loss')
gnuplot.plotflush()


local range = torch.range(1, epochs)
gnuplot.pngfigure('test1.png')
gnuplot.plot({'trainError',trainError},{'testError',testError})
gnuplot.xlabel('epochs')
gnuplot.ylabel('Loss')
gnuplot.plotflush()

torch.save('mycnn_adam+hflip_2.dat', model)

--  ****************************************************************
--  Visualizing Network Weights+Activations
--  ****************************************************************

--[[
local Weights_1st_Layer = model:get(1).weight
local scaledWeights = image.scale(image.toDisplayTensor({input=Weights_1st_Layer,padding=2}),200)
saveTensorAsGrid(scaledWeights,'Weights_1st_Layer.jpg')


print('Input Image')
saveTensorAsGrid(testData[100],'testImg100.jpg')
model:forward(testData[100]:view(1,3,32,32):cuda())
for l=1,9 do
  print('Layer ' ,l, tostring(model:get(l)))
  local layer_output = model:get(l).output[1]
  saveTensorAsGrid(layer_output,'Layer'..l..'-'..tostring(model:get(l))..'.jpg')
  if ( l == 5 or l == 9 )then
	local Weights_lst_Layer = model:get(l).weight
	local scaledWeights = image.scale(image.toDisplayTensor({input=Weights_lst_Layer[1],padding=2}),200)
	saveTensorAsGrid(scaledWeights,'Weights_'..l..'st_Layer.jpg')
  end 
end
]]--
